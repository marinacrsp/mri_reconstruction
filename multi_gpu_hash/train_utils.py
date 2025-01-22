import os
from pathlib import Path
from typing import Optional

import fastmri
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils import *
from fastmri.data.transforms import tensor_to_complex_np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        dataloader,
        embeddings_vol,
        embeddings_coil,
        embeddings_coil_idx,
        model,
        loss_fn,
        optimizer,
        scheduler,
        device,
        config,
    ) -> None:
        self.device = device
        self.n_epochs = config["n_epochs"]

        self.dataloader = dataloader
        self.embeddings_vol, self.embeddings_coil = embeddings_vol.to(self.device), embeddings_coil.to(self.device)
        
        # Wrap the model in the data distributed parallelism
        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.device])
        
        # Wrap the embeddings also in the data distributed parallelism
        self.embeddings_vol, self.embeddings_coil = DDP(embeddings_vol, device_ids=[self.device]), DDP(embeddings_coil, device_ids=[self.device])
        self.start_idx = embeddings_coil_idx.to(self.device)

        # If stateful loss function, move its "parameters" to `device`.
        if hasattr(loss_fn, "to"):
            self.loss_fn = loss_fn.to(self.device)
        else:
            self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Only one process does the logging (to avoid redundancy).
        if self.device == 0:
            self.log_interval = config["log_interval"]
            self.checkpoint_interval = config["checkpoint_interval"]
            self.path_to_out = Path(config["path_to_outputs"])
            self.timestamp = config["timestamp"]
            self.writer = SummaryWriter(self.path_to_out / self.timestamp)

            # Ground truth (used to compute the evaluation metrics).
            self.ground_truth = []
            for vol_id in self.dataloader.dataset.metadata.keys():
                file = self.dataloader.dataset.metadata[vol_id]["file"]
                with h5py.File(file, "r") as hf:
                    self.ground_truth.append(
                        hf["reconstruction_rss"][()][: config["dataset"]["n_slices"]]
                    )

            # Scientific and nuissance hyperparameters.
            self.hparam_info = config["hparam_info"]
            self.hparam_info["learning_rate"] = self.scheduler.get_last_lr()[0]
            self.hparam_info["loss"] = config["loss"]["id"]
            self.hparam_info["acceleration"] = config["dataset"]["acceleration"]
            self.hparam_info["center_frac"] = config["dataset"]["center_frac"]
            # self.hparam_info["embedding_dim"] = self.embeddings.module.embedding_dim
            self.hparam_info["sigma"] = config["loss"]["params"]["sigma"]
            self.hparam_info["gamma"] = config["loss"]["params"]["gamma"]

            # Evaluation metrics for the last log.
            self.last_nmse = [0] * len(
                self.dataloader.dataset.metadata
            )  # Used during the last log.
            self.last_psnr = [0] * len(
                self.dataloader.dataset.metadata
            )  # Used during the last log.
            self.last_ssim = [0] * len(
                self.dataloader.dataset.metadata
            )  # Used during the last log.

    ###########################################################################
    ###########################################################################
    ###########################################################################

    def train(self):
        """Train the model across multiple epochs and log the performance."""
        empirical_risk = 0
        for epoch_idx in range(self.n_epochs):
            # In distributed mode, calling the set_epoch() method before creating the DataLoader iterator is necessary to make shuffling work properly.
            # Otherwise, the same ordering will be always used.
            self.dataloader.sampler.set_epoch(epoch_idx)
            
            empirical_risk = self._run_epoch(epoch_idx)

            if self.device == 0:
                print(f"EPOCH {epoch_idx}    avg loss: {empirical_risk}\n")
                self.writer.add_scalar("Loss/train", empirical_risk, epoch_idx)
                # TODO: UNCOMMENT WHEN USING LR SCHEDULER.
                # self.writer.add_scalar("Learning Rate", self.scheduler.get_last_lr()[0], epoch_idx)

                if (epoch_idx + 1) % self.log_interval == 0:
                    self._log_performance(epoch_idx)
                    self._log_weight_info(epoch_idx)

                if (epoch_idx + 1) % self.checkpoint_interval == 0:
                    # Takes ~3 seconds.
                    self._save_checkpoint(epoch_idx)

        if self.device == 0:
            self._log_information(empirical_risk)
            self.writer.close()

    def _run_epoch(self, epoch_idx):
        # Also known as "empirical risk".
        avg_loss = 0.0
        n_obs = 0

        self.model.train()
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Get the index for the coil latent embedding
            coords = inputs[:, 1:-1] #kx, ky, kz
            vol_ids = inputs[:,0].long()
            coil_ids = inputs[:,-1].long() 
            
            latent_vol = self.embeddings_vol(vol_ids)
            latent_coil = self.embeddings_coil(self.start_idx[vol_ids] + coil_ids)
            
            self.optimizer.zero_grad(set_to_none=True)
    
            outputs = self.model(coords, latent_vol, latent_coil)
            
            # Can be thought as a moving average (with "stride" `batch_size`) of the loss.
            batch_loss = self.loss_fn(outputs, targets, latent_vol)

            batch_loss.backward()
            self.optimizer.step()

            avg_loss += batch_loss.item() * len(inputs)
            n_obs += len(inputs)

        self.scheduler.step()
        avg_loss = avg_loss / n_obs
        return avg_loss

    ###########################################################################
    ###########################################################################
    ###########################################################################

    @torch.no_grad()
    def predict(self, vol_id, shape, left_idx, right_idx, center_vals):
        """Reconstruct MRI volume (k-space)."""
        self.model.eval()
        n_slices, n_coils, height, width = shape

        # Create tensors of indices for each dimension
        kx_ids = torch.cat([torch.arange(left_idx), torch.arange(right_idx, width)])
        ky_ids = torch.arange(height)
        kz_ids = torch.arange(n_slices)
        coil_ids = torch.arange(n_coils)

        # Use meshgrid to create expanded grids
        kspace_ids = torch.meshgrid(kx_ids, ky_ids, kz_ids, coil_ids, indexing="ij")
        kspace_ids = torch.stack(kspace_ids, dim=-1).reshape(-1, len(kspace_ids))

        dataset = TensorDataset(kspace_ids)
        dataloader = DataLoader(
            dataset, batch_size=60_000, shuffle=False, num_workers=3
        )
        vol_embeddings = self.embeddings_vol(
            torch.tensor([vol_id] * 60_000, dtype=torch.long, device=self.device)
        )

        volume_kspace = torch.zeros(
            (n_slices, n_coils, height, width, 2),
            device=self.device,
            dtype=torch.float32,
        )
        for point_ids in dataloader:
            point_ids = point_ids[0].to(self.device, dtype=torch.long)
            coords = torch.zeros_like(
                point_ids, dtype=torch.float32, device=self.device
            )
            # Normalize the necessary coordinates for hash encoding to work
            coords[:, :2] = point_ids[:, :2]
            coords[:, 2] = (2 * point_ids[:, 2]) / (n_slices - 1) - 1
            coords[:, 3] = point_ids[:, 3]
            
            coil_embeddings = self.embeddings_coil(self.start_idx[vol_id] + coords[:,3].long())

            # Need to add `:len(coords)` because the last batch has a different size (than 60_000).
            outputs = self.model(coords, vol_embeddings[: len(coords)], coil_embeddings)
            
            # "Fill in" the unsampled region.
            volume_kspace[
                point_ids[:, 2], point_ids[:, 3], point_ids[:, 1], point_ids[:, 0]
            ] = outputs

        # Multiply by the normalization constant.
        volume_kspace = (
            volume_kspace * self.dataloader.dataset.metadata[vol_id]["norm_cste"]
        )

        volume_kspace = tensor_to_complex_np(volume_kspace.detach().cpu())

        # "Fill-in" center values.
        volume_kspace[..., left_idx:right_idx] = center_vals

        volume_img = rss(inverse_fft2_shift(volume_kspace))
        vol_c0 = np.abs(inverse_fft2_shift(volume_kspace)[:,0])
        vol_c1 = np.abs(inverse_fft2_shift(volume_kspace)[:,1]) 
        vol_c2 = np.abs(inverse_fft2_shift(volume_kspace)[:,2]) 
        vol_c3 = np.abs(inverse_fft2_shift(volume_kspace)[:,3])

        self.model.train()
        return volume_img, vol_c0, vol_c1, vol_c2, vol_c3

    ###########################################################################
    ###########################################################################
    ###########################################################################

    @torch.no_grad()
    def _log_performance(self, epoch_idx):
        for vol_id in self.dataloader.dataset.metadata.keys():

            # Predict volume image.
            shape = self.dataloader.dataset.metadata[vol_id]["shape"]
            center_data = self.dataloader.dataset.metadata[vol_id]["center"]
            left_idx, right_idx, center_vals = (
                center_data["left_idx"],
                center_data["right_idx"],
                center_data["vals"],
            )

            volume_img, vol_c0, vol_c1, vol_c2, vol_c3 = self.predict(vol_id, shape, left_idx, right_idx, center_vals)

            volume_kspace = fft2_shift(volume_img)  # To get "single-coil" k-space.
            # volume_kspace[..., left_idx:right_idx] = 0

            ##################################################
            # Log kspace values.
            ##################################################
            # Plot modulus and argument.
            modulus = np.abs(volume_kspace)
            cste_mod = self.dataloader.dataset.metadata[vol_id]["plot_cste"]

            argument = np.angle(volume_kspace)
            cste_arg = np.pi / 180

            ##################################################
            # Log image space values
            ##################################################
            volume_img = np.abs(volume_img)

            for slice_id in range(shape[0]):
                self._plot_info(
                    modulus[slice_id],
                    argument[slice_id],
                    cste_mod,
                    cste_arg,
                    "Modulus",
                    "Argument",
                    epoch_idx,
                    f"prediction/vol_{vol_id}/slice_{slice_id}/kspace_v1")
                
                
                # Plot 4 coils image
                fig = plt.figure(figsize=(20, 10))
                plt.subplot(1,4,1)
                plt.imshow(vol_c0[slice_id], cmap='gray')
                plt.axis('off')
                
                plt.subplot(1,4,2)
                plt.imshow(vol_c1[slice_id], cmap='gray')
                plt.axis('off')
            
                plt.subplot(1,4,3)
                plt.imshow(vol_c2[slice_id], cmap='gray')
                plt.axis('off')
                
                plt.subplot(1,4,4)
                plt.imshow(vol_c3[slice_id], cmap='gray')
                plt.axis('off')
                
                self.writer.add_figure(
                    f"prediction/vol_{vol_id}/slice_{slice_id}/coils_img",
                    fig,
                    global_step=epoch_idx,
                )
                plt.close(fig)
                
                # Plot image.
                fig = plt.figure(figsize=(8, 8))
                plt.imshow(volume_img[slice_id], cmap='gray')
                self.writer.add_figure(
                    f"prediction/vol_{vol_id}/slice_{slice_id}/volume_img",
                    fig,
                    global_step=epoch_idx,
                )
                plt.close(fig)

            # Log evaluation metrics.
            ssim_val = ssim(self.ground_truth[vol_id], volume_img)
            self.writer.add_scalar(f"eval/vol_{vol_id}/ssim", ssim_val, epoch_idx)

            psnr_val = psnr(self.ground_truth[vol_id], volume_img)
            self.writer.add_scalar(f"eval/vol_{vol_id}/psnr", psnr_val, epoch_idx)

            nmse_val = nmse(self.ground_truth[vol_id], volume_img)
            self.writer.add_scalar(f"eval/vol_{vol_id}/nmse", nmse_val, epoch_idx)

            # Update.
            self.last_nmse[vol_id] = nmse_val
            self.last_psnr[vol_id] = psnr_val
            self.last_ssim[vol_id] = ssim_val

    def _plot_info(
        self, data_1, data_2, cste_1, cste_2, title_1, title_2, epoch_idx, tag
    ):
        fig = plt.figure(figsize=(20, 20))

        plt.subplot(2, 2, 1)
        plt.imshow(np.log(data_1 / cste_1))
        plt.colorbar()
        plt.title(f"{title_1} kspace")

        plt.subplot(2, 2, 2)
        plt.hist(data_1.flatten(), log=True, bins=100)

        max_val = np.max(data_1)
        min_val = np.min(data_1)
        # ignoring zero data
        non_zero = data_1 > 0
        mean = np.mean(data_1[non_zero])
        median = np.median(data_1[non_zero])
        q05 = np.quantile(data_1[non_zero], 0.05)
        q95 = np.quantile(data_1[non_zero], 0.95)

        plt.axvline(
            mean, color="r", linestyle="dashed", linewidth=2, label=f"Mean: {mean:.2e}"
        )
        plt.axvline(
            median,
            color="g",
            linestyle="dashed",
            linewidth=2,
            label=f"Median: {median:.2e}",
        )
        plt.axvline(
            q05, color="b", linestyle="dotted", linewidth=2, label=f"Q05: {q05:.2e}"
        )
        plt.axvline(
            q95, color="b", linestyle="dotted", linewidth=2, label=f"Q95: {q95:.2e}"
        )
        plt.axvline(
            min_val,
            color="orange",
            linestyle="solid",
            linewidth=2,
            label=f"Min: {min_val:.2e}",
        )
        plt.axvline(
            max_val,
            color="purple",
            linestyle="solid",
            linewidth=2,
            label=f"Max: {max_val:.2e}",
        )
        plt.legend()
        plt.title(f"{title_1} histogram")

        plt.subplot(2, 2, 3)
        plt.imshow(data_2 / cste_2)
        plt.colorbar()
        plt.title(f"{title_2} kspace")

        plt.subplot(2, 2, 4)
        plt.hist(data_2.flatten(), log=True, bins=100)

        max_val = np.max(data_2)
        min_val = np.min(data_2)
        # ignoring zero data
        non_zero = data_2 > 0
        mean = np.mean(data_2[non_zero])
        median = np.median(data_2[non_zero])
        q05 = np.quantile(data_2[non_zero], 0.05)
        q95 = np.quantile(data_2[non_zero], 0.95)

        plt.axvline(
            mean, color="r", linestyle="dashed", linewidth=2, label=f"Mean: {mean:.2e}"
        )
        plt.axvline(
            median,
            color="g",
            linestyle="dashed",
            linewidth=2,
            label=f"Median: {median:.2e}",
        )
        plt.axvline(
            q05, color="b", linestyle="dotted", linewidth=2, label=f"Q05: {q05:.2e}"
        )
        plt.axvline(
            q95, color="b", linestyle="dotted", linewidth=2, label=f"Q95: {q95:.2e}"
        )
        plt.axvline(
            min_val,
            color="orange",
            linestyle="solid",
            linewidth=2,
            label=f"Min: {min_val:.2e}",
        )
        plt.axvline(
            max_val,
            color="purple",
            linestyle="solid",
            linewidth=2,
            label=f"Max: {max_val:.2e}",
        )
        plt.legend()
        plt.title(f"{title_2} histogram")

        self.writer.add_figure(tag, fig, global_step=epoch_idx)
        plt.close(fig)

    @torch.no_grad()
    def _log_weight_info(self, epoch_idx):
        """Log weight values and gradients."""
        for module, case in zip(
            [self.model.module, self.embeddings_vol.module, self.embeddings_coil.module], ["network", "embeddings"]
        ):
            for name, param in module.named_parameters():
                subplot_count = 1 if param.data is None else 2
                fig = plt.figure(figsize=(8 * subplot_count, 5))

                plt.subplot(1, subplot_count, 1)
                plt.hist(param.data.cpu().numpy().flatten(), bins=100, log=True)
                # plt.hist(param.data.cpu().numpy().flatten(), bins='auto', log=True)
                plt.title("Values")

                if param.grad is not None:
                    plt.subplot(1, subplot_count, 2)
                    # plt.hist(param.grad.cpu().numpy().flatten(), bins='auto', log=True)
                    plt.hist(param.grad.cpu().numpy().flatten(), bins=100, log=True)
                    plt.title("Gradients")

                tag = name.replace(".", "/")
                self.writer.add_figure(
                    f"params/{case}/{tag}", fig, global_step=epoch_idx
                )
                plt.close(fig)

    @torch.no_grad()
    # Save current model gradients (model is wraped in DDP so .module is needed)
    def _save_checkpoint(self, epoch_idx):
        """Save current state of the training process."""
        # Ensure the path exists.
        path = self.path_to_out / self.timestamp / "checkpoints"
        os.makedirs(path, exist_ok=True)

        path_to_file = path / f"epoch_{epoch_idx:04d}.pt"

        # Prepare state to save.
        save_dict = {
            "model_state_dict": self.model.module.state_dict(),
            "embedding_coil_state_dict": self.embeddings_coil.module.state_dict(),
            "embedding_vol_state_dict": self.embeddings_vol.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        # Save trainer state.
        torch.save(save_dict, path_to_file)

    @torch.no_grad()
    def _log_information(self, loss):
        """Log 'scientific' and 'nuissance' hyperparameters."""

        if hasattr(self.model.module, "activation"):
            self.hparam_info["hidden_activation"] = type(
                self.model.module.activation
            ).__name__
        elif type(self.model.module).__name__ == "Siren":
            self.hparam_info["hidden_activation"] = "Sine"

        if hasattr(self.model.module, "out_activation"):
            self.hparam_info["output_activation"] = type(
                self.model.module.out_activation
            ).__name__
        else:
            self.hparam_info["output_activation"] = "None"

        hparam_metrics = {"hparam/loss": loss}
        hparam_metrics["hparam/eval_metric/nmse"] = np.mean(self.last_nmse)
        hparam_metrics["hparam/eval_metric/psnr"] = np.mean(self.last_psnr)
        hparam_metrics["hparam/eval_metric/ssim"] = np.mean(self.last_ssim)
        self.writer.add_hparams(self.hparam_info, hparam_metrics)


###########################################################################
###########################################################################
###########################################################################


##################################################
# Loss Functions
##################################################


class MAELoss:
    """Mean Absolute Error Loss Function."""

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, predictions, targets):
        loss = torch.sum(torch.abs(predictions - targets), dim=-1)
        return self.gamma * torch.mean(loss)


class DMAELoss:
    """Dynamic Mean Absolute Error Loss Function."""

    def __init__(self, gamma=100):
        self.gamma = gamma

    def __call__(self, predictions, targets, epoch_id):
        loss = torch.sum(torch.abs(predictions - targets), dim=-1)
        return (epoch_id / self.gamma + 1) * torch.mean(loss)


class MSELoss:
    """Mean Squared Error Loss Function."""

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, predictions, targets):
        predictions = torch.view_as_complex(predictions)
        targets = torch.view_as_complex(targets)

        loss = ((predictions - targets).abs()) ** 2

        return self.gamma * torch.mean(loss)


class MSEL2Loss:
    """Mean Squared Error Loss Function with L2 (latent embedding) Regularization."""

    def __init__(self, sigma, gamma):
        self.sigma_squared = sigma**2
        self.gamma = gamma

    def __call__(self, predictions, targets, embeddings):
        predictions = torch.view_as_complex(predictions)
        targets = torch.view_as_complex(targets)

        loss = ((predictions - targets).abs()) ** 2

        reg = (embeddings**2).sum(axis=-1) / self.sigma_squared

        return torch.mean(loss) + self.gamma * torch.mean(reg)


class MSEDistLoss:
    """Mean Squared Error Loss Function."""

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, predictions, targets, kcoords):
        predictions = torch.view_as_complex(predictions)
        targets = torch.view_as_complex(targets)

        distance_to_center = torch.sqrt(kcoords[:, 0] ** 2 + kcoords[:, 1] ** 2)

        loss = (1 + 1 / distance_to_center) * ((predictions - targets).abs()) ** 2

        return self.gamma * torch.mean(loss)


class HDRLoss:
    def __init__(self, sigma, epsilon, factor):
        self.sigma = sigma
        self.epsilon = epsilon
        self.factor = factor

    def __call__(self, predictions, targets, kcoords):
        predictions = torch.view_as_complex(predictions)
        targets = torch.view_as_complex(targets)

        loss = ((predictions - targets).abs() / (targets.abs() + self.epsilon)) ** 2

        if self.factor > 0.0:
            dist_to_center2 = kcoords[:, 0] ** 2 + kcoords[:, 1] ** 2
            filter_value = torch.exp(-dist_to_center2 / (2 * self.sigma**2))

            reg_error = predictions - predictions * filter_value
            reg = self.factor * (reg_error.abs() / (targets.abs() + self.epsilon)) ** 2

            return loss.mean() + reg.mean()

        else:
            return loss.mean()


class LogL2Loss(torch.nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        log_y_pred = torch.log(fastmri.complex_abs(y_pred) + self.epsilon)
        log_y_true = torch.log(fastmri.complex_abs(y_true) + self.epsilon)
        loss = torch.mean((log_y_pred - log_y_true) ** 2)
        return loss


###########################################################################
###########################################################################
###########################################################################

##################################################
# Evaluation Metrics
##################################################


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array(0.0)
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]
