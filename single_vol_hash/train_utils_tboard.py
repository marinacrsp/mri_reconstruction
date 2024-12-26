import os
from pathlib import Path
from typing import Optional

import fastmri
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils import *
from fastmri.data.transforms import tensor_to_complex_np, to_tensor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, TensorDataset
from pisco import *
from helper_functions import *
from torch.utils.tensorboard import SummaryWriter # To print to tensorboard


class Trainer:
    def __init__(
        self, dataloader_consistency, dataloader_pisco, model, loss_fn, optimizer, scheduler, config
    ) -> None:
        self.device = torch.device(config["device"])
        self.n_epochs = config["n_epochs"]

        self.dataloader_consistency = dataloader_consistency
        self.dataloader_pisco = dataloader_pisco
        
        self.model = model.to(self.device)
        
        # If stateful loss function, move its "parameters" to `device`.
        if hasattr(loss_fn, "to"):
            self.loss_fn = loss_fn.to(self.device)
        else:
            self.loss_fn = loss_fn
            
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.log_interval = config["log_interval"]
        self.checkpoint_interval = config["n_epochs"] # Initialize the checkpoint interval as this value
        self.path_to_out = Path(config["path_to_outputs"])
        self.timestamp = config["timestamp"]
        
        self.add_pisco = config["l_pisco"]["addpisco"]
        self.E_epoch = config["l_pisco"]["E_epoch"]
        self.alpha = config["l_pisco"]["alpha"]
        self.factor = config["l_pisco"]["factor"]
        self.minibatch = config["l_pisco"]["minibatch"]
        self.patch_size = config["l_pisco"]["patch_size"]
        
        self.writer = SummaryWriter(self.path_to_out / self.timestamp)

        # Ground truth (used to compute the evaluation metrics).
        file = self.dataloader_consistency.dataset.metadata[0]["file"]
        with h5py.File(file, "r") as hf:
            self.ground_truth = hf["reconstruction_rss"][()][
                : config["dataset"]["n_slices"]
            ]
            self.kspace_gt = to_tensor(preprocess_kspace(hf["kspace"][()][: config["dataset"]["n_slices"]]))


        # Scientific and nuissance hyperparameters.
        self.hparam_info = config["hparam_info"]
        self.hparam_info["n_layer"] = config["model"]["params"]["n_layers"]
        self.hparam_info["hidden_dim"] = config["model"]["params"]["hidden_dim"]
        # self.hparam_info["resolution_levels"] = config["model"]["params"]["levels"]
        self.hparam_info["batch_size"] = config["dataloader"]["batch_size"]
        # self.hparam_info["pisco_weightfactor"] = config["l_pisco"]["factor"]
        
        print(self.hparam_info)

        # Evaluation metrics for the last log.
        self.last_nmse = [0] * len(self.dataloader_consistency.dataset.metadata)
        self.last_psnr = [0] * len(self.dataloader_consistency.dataset.metadata)
        self.last_ssim = [0] * len(self.dataloader_consistency.dataset.metadata)

    ###########################################################################
    ###########################################################################
    ###########################################################################

    def train(self):
        """Train the model across multiple epochs and log the performance."""
        empirical_risk = 0
        empirical_pisco = 0
        for epoch_idx in range(self.n_epochs):
            empirical_risk = self._train_one_epoch()
            
            print(f"EPOCH {epoch_idx}    avg loss: {empirical_risk}\n")
            
            # Log the errors
            self.writer.add_scalar("Loss/train", empirical_risk, epoch_idx)
            self.writer.add_scalar("Loss/Pisco", empirical_pisco, epoch_idx)
            
            # Log the average residuals
            # TODO: UNCOMMENT WHEN USING LR SCHEDULER.
            # self.writer.add_scalar("Learning Rate", self.scheduler.get_last_lr()[0], epoch_idx)

            if (epoch_idx + 1) % self.log_interval == 0:
                self._log_performance(epoch_idx)
                self._log_weight_info(epoch_idx)
            
            if (epoch_idx + 1) % self.checkpoint_interval == 0:
                # Takes ~3 seconds.
                self._save_checkpoint(epoch_idx)
                

        self._log_information(empirical_risk)
        self.writer.close()
        
    def _train_one_epoch(self):
        # Also known as "empirical risk".
        avg_loss = 0.0
        n_obs = 0

        self.model.train()
        for inputs, targets in self.dataloader_consistency:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            
            # Can be thought as a moving average (with "stride" `batch_size`) of the loss.
            batch_loss = self.loss_fn(outputs, targets)
            # NOTE: Uncomment for some of the loss functions (e.g. 'MSEDistLoss').
            # batch_loss = self.loss_fn(outputs, targets, inputs)

            batch_loss.backward()
            self.optimizer.step()

            avg_loss += batch_loss.item() * len(inputs)
            n_obs += len(inputs)

        avg_loss = avg_loss / n_obs
        return avg_loss
    
    ##########################################################################
    ##########################################################################
    ##########################################################################
    
    @torch.no_grad()
    def predict(self, vol_id, shape, left_idx, right_idx, center_vals, epoch_idx):
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
            
            # For hash encodings normalize only the last two coordinates
            coords[:, :2] = point_ids[:, :2]
            coords[:, 2] = (2 * point_ids[:, 2]) / (n_slices - 1) - 1
            coords[:, 3] = (2 * point_ids[:, 3]) / (n_coils - 1) - 1

            outputs = self.model(coords)
            # "Fill in" the unsampled region.
            volume_kspace[
                point_ids[:, 2], point_ids[:, 3], point_ids[:, 1], point_ids[:, 0]
            ] = outputs
            
    
        # Multiply by the normalization constant.
        volume_kspace = (
            volume_kspace * self.dataloader_consistency.dataset.metadata[vol_id]["norm_cste"]
        )

        volume_kspace = tensor_to_complex_np(volume_kspace.detach().cpu())

        self.model.train()
        return volume_kspace

    ###########################################################################
    ###########################################################################
    ###########################################################################

    @torch.no_grad() 
    def _log_performance(self, epoch_idx, vol_id = 0):
            # Predict volume image.
            
            shape = self.dataloader_consistency.dataset.metadata[vol_id]["shape"]
            center_data = self.dataloader_consistency.dataset.metadata[vol_id]["center"]
            left_idx, right_idx, center_vals = (
                center_data["left_idx"],
                center_data["right_idx"],
                center_data["vals"],
            )

            volume_kspace = self.predict(vol_id, shape, left_idx, right_idx, center_vals, epoch_idx)
            cste_mod = self.dataloader_consistency.dataset.metadata[vol_id]["norm_cste"]
            
            y_kspace_data = tensor_to_complex_np(self.kspace_gt)
            
            mask = self.dataloader_consistency.dataset.metadata[vol_id]["mask"].squeeze(-1).expand(shape).numpy()

            y_kspace_data_u = y_kspace_data  * (mask)
            y_kspace_prediction_u = volume_kspace * (1-mask)
            y_kspace_final = y_kspace_data_u + y_kspace_prediction_u

            y_kspace_data_rss = rss(y_kspace_data_u)
            y_kspace_prediction_rss = rss(y_kspace_prediction_u)
            y_kspace_final_rss = rss(y_kspace_final)
            
            y_kspace_final[..., left_idx:right_idx] = center_vals
            y_img_final = np.abs(rss(inverse_fft2_shift(y_kspace_final)))
            y_kspace_final_wcenter_rss = rss(y_kspace_final)
            
            ###### predict the edges - image
            y_img_edges = np.abs(rss(inverse_fft2_shift(volume_kspace)))
            
            ###### predict the center - image
            volume_kspace[..., left_idx:right_idx] = center_vals
            y_img_edges_center = np.abs(rss(inverse_fft2_shift(volume_kspace)))
            volume_kspace_rss = rss(volume_kspace)
            
            ###### raw img w/o 
            y_kspace_data[..., left_idx:right_idx] = 0
            raw_img_edges = np.abs(rss(inverse_fft2_shift(y_kspace_data)))
            

            for slice_id in range(shape[0]):
                self._plot_3subplots(y_img_edges, 'Edges',
                                    y_img_edges_center, 'Edges + centre',
                                    y_img_final, 'Edges + centre + acquisitions', 
                                    slice_id, 
                                    epoch_idx, 
                                    f"prediction/vol_{vol_id}_slice_{slice_id}/volume_img",
                                    'gray')

                self._plot_3subplots(np.abs(y_kspace_data_rss/cste_mod), 'M · ydata',
                                    np.abs(y_kspace_prediction_rss/cste_mod), '(1-M) · ypred', 
                                    np.abs(y_kspace_final_rss/cste_mod), 'M · ydata + (1-M)·ypred',
                                    slice_id, epoch_idx, 
                                    f"prediction/vol_{vol_id}_slice_{slice_id}/kspace composition",
                                    'viridis')

                self._plot_2subplots(self.ground_truth, 'groundtruth vol',
                                    raw_img_edges, 'groundtruth edges', 
                                    slice_id, epoch_idx, 
                                    f"groundtruth/vol_{vol_id}_slice_{slice_id}", 'gray')
                
                self._plot_2subplots(np.log(volume_kspace_rss/cste_mod + 1.e-45), 'kspace predicted',
                                    np.log(y_kspace_final_wcenter_rss/ cste_mod + 1.e-45), 'kspace predicted + acquired', 
                                    slice_id, epoch_idx, 
                                    f"prediction/vol_{vol_id}_slice_{slice_id}/kspace logarigthm", 'viridis')
            
            
                # Plot 4 coils image
                # fig = plt.figure(figsize=(20, 10))
                
                # for i in range(4):
                #     plt.subplot(1,4,i+1)
                #     plt.imshow(coils_img[slice_id], cmap='gray')
                #     plt.axis('off')
                
                # self.writer.add_figure(
                #     f"prediction/vol_{vol_id}/slice_{slice_id}/coils_img",
                #     fig,
                #     global_step=epoch_idx,
                # )
                # plt.close(fig)
                
                

            # self._log_coil_embeddings(epoch_idx, f"embeddings/coil")

            ############################################################
            # Log evaluation metrics.
            nmse_val = nmse(raw_img_edges, y_img_edges)
            self.writer.add_scalar(f"eval/vol_{vol_id}/nmse_edges", nmse_val, epoch_idx)

            psnr_val = psnr(raw_img_edges, y_img_edges)
            self.writer.add_scalar(f"eval/vol_{vol_id}/psnr_edges", psnr_val, epoch_idx)

            ssim_val = ssim(raw_img_edges, y_img_edges)
            self.writer.add_scalar(f"eval/vol_{vol_id}/ssim_edges", ssim_val, epoch_idx)
            
            # ############################################################
            # # # Comparison metrics for the volume image w center and the groundtruth
            nmse_val = nmse(self.ground_truth, y_img_edges_center)
            self.writer.add_scalar(f"eval/vol_{vol_id}/nmse_wcenter", nmse_val, epoch_idx)

            psnr_val = psnr(self.ground_truth, y_img_edges_center)
            self.writer.add_scalar(f"eval/vol_{vol_id}/psnr_wcenter", psnr_val, epoch_idx)

            ssim_val = ssim(self.ground_truth, y_img_edges_center)
            self.writer.add_scalar(f"eval/vol_{vol_id}/ssim_wcenter", ssim_val, epoch_idx)

            # ############################################################
            # # Comparison metrics for the volume image w center + predictions and the groundtruth
            nmse_val = nmse(self.ground_truth, y_img_final)
            self.writer.add_scalar(f"eval/vol_{vol_id}/nmse_acq_pred", nmse_val, epoch_idx)

            psnr_val = psnr(self.ground_truth, y_img_final)
            self.writer.add_scalar(f"eval/vol_{vol_id}/psnr_acq_pred", psnr_val, epoch_idx)

            ssim_val = ssim(self.ground_truth, y_img_final)
            self.writer.add_scalar(f"eval/vol_{vol_id}/ssim_acq_pred", ssim_val, epoch_idx)

            # Update.
            self.last_nmse = nmse_val
            self.last_psnr = psnr_val
            self.last_ssim = ssim_val

    @torch.no_grad()
    def _plot_3subplots(
        self, data_1, title1, data_2, title2, data_3, title3, slice_id, epoch_idx, tag, map
        ):
        fig = plt.figure(figsize=(20,30))
        plt.subplot(1,3,1)
        plt.imshow(data_1[slice_id], cmap=map)
        plt.title(title1)
        plt.axis('off')
        
        plt.subplot(1,3,2)
        plt.imshow(data_2[slice_id], cmap=map)
        plt.title(title2)
        plt.axis('off')
        
        plt.subplot(1,3,3)                
        plt.imshow(data_3[slice_id], cmap=map)
        plt.title(title3)
        plt.axis('off')
        
            
        self.writer.add_figure(
            tag,
            fig,
            global_step=epoch_idx,
        )
        plt.close(fig)

        
    @torch.no_grad()
    def _plot_2subplots(
        self, data_1, title1, data_2, title2, slice_id, epoch_idx, tag, map
        ):
        fig = plt.figure(figsize=(20,20))
        plt.subplot(1,2,1)
        plt.imshow(data_1[slice_id], cmap=map)
        plt.title(title1)
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(data_2[slice_id], cmap=map)
        plt.title(title2)
        plt.axis('off')
            
        self.writer.add_figure(
            tag,
            fig,
            global_step=epoch_idx,
        )
        plt.close(fig)
    @torch.no_grad()
    def _log_weight_info(self, epoch_idx):
        """Log weight values and gradients."""
        for name, param in self.model.named_parameters():
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
            self.writer.add_figure(f"params/{tag}", fig, global_step=epoch_idx)
            plt.close(fig)

    @torch.no_grad()
    def _save_checkpoint(self, epoch_idx):
        """Save current state of the training process."""
        # Ensure the path exists.
        path = self.path_to_out / self.timestamp / "checkpoints"
        os.makedirs(path, exist_ok=True)

        path_to_file = path / f"epoch_{epoch_idx:04d}.pt"

        # Prepare state to save.
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        # Save trainer state.
        torch.save(save_dict, path_to_file)

    @torch.no_grad()
    def _log_information(self, loss):
        """Log 'scientific' and 'nuissance' hyperparameters."""

        # if hasattr(self.model, "activation"):
        #     self.hparam_info["hidden_activation"] = type(self.model.activation).__name__
        # elif type(self.model).__name__ == "Siren":
        #     self.hparam_info["hidden_activation"] = "Sine"
        # if hasattr(self.model, "out_activation"):
        #     self.hparam_info["output_activation"] = type(
        #         self.model.out_activation
        #     ).__name__
        # else:
        #     self.hparam_info["output_activation"] = "None"

        hparam_metrics = {"hparam/loss": loss}
        hparam_metrics["hparam/eval_metric/nmse"] = np.mean(self.last_nmse)
        hparam_metrics["hparam/eval_metric/psnr"] = np.mean(self.last_psnr)
        hparam_metrics["hparam/eval_metric/ssim"] = np.mean(self.last_ssim)
        
        self.writer.add_hparams(self.hparam_info, hparam_metrics)

        # Log model's architecture.
        inputs, _ = next(iter(self.dataloader_consistency))
        inputs = inputs.to(self.device)
        self.writer.add_graph(self.model, inputs)


###########################################################################
###########################################################################
##########################################################################