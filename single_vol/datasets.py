import os
import random
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from data_utils import *
from fastmri.data.subsample import EquiSpacedMaskFunc, RandomMaskFunc
from fastmri.data.transforms import tensor_to_complex_np, to_tensor
from torch.utils.data import Dataset


class KCoordDataset(Dataset):
    def __init__(
        self,
        path_to_data: Union[str, Path, os.PathLike],
        n_volumes: int = 1,
        n_slices: int = 3,
        with_mask: bool = True,
        acceleration: int = 4,
        mask_type = 'Equispaced',
        center_frac: float = 0.15,
    ):
        self.metadata = {}
        self.inputs = []
        self.targets = []

        path_to_data = Path(path_to_data)
        if path_to_data.is_dir():
            files = sorted(
                [
                    file
                    for file in path_to_data.iterdir()
                    if file.suffix == ".h5" and "AXT1POST_205" in file.name
                ]
            )[:n_volumes]
        else:
            files = [path_to_data]

        # For each MRI volume in the dataset...
        for vol_id, file in enumerate(files):
            # Load MRI volume
            with h5py.File(file, "r") as hf:
                volume_kspace = to_tensor(preprocess_kspace(hf["kspace"][()]))[
                    :n_slices
                ]

            ##################################################
            # Mask creation
            ##################################################
            if mask_type == "Random":
                mask_func = RandomMaskFunc(center_fractions=[center_frac], accelerations=[acceleration])
                
            else: 
                mask_func = EquiSpacedMaskFunc(center_fractions=[center_frac], accelerations=[acceleration])
                
            shape = (1,) * len(volume_kspace.shape[:-3]) + tuple(
                volume_kspace.shape[-3:]
            )
            mask, _ = mask_func(
                shape, None, vol_id
            )  # use the volume index as random seed.

            mask, left_idx, right_idx = remove_center(mask)
            # _, left_idx, right_idx = remove_center(mask)  # NOTE: Uncomment to include the center region in the training data. Note that 'left_idx' and 'right_idx' are still needed.

            ##################################################
            # Computing the indices
            ##################################################
            n_slices, n_coils, height, width = volume_kspace.shape[:-1]
            if with_mask:
                print(f'Training with {mask_type} mask')
                kx_ids = torch.where(mask.squeeze())[0]
            else:
                kx_ids = torch.from_numpy(np.setdiff1d(np.arange(width), np.arange(left_idx, right_idx))) # NOTE: Uncomment to include all the datapoints (fully-sampled volume), with the exception of the center region.
            ky_ids = torch.arange(height)
            kz_ids = torch.arange(n_slices)
            coil_ids = torch.arange(n_coils)

            kspace_ids = torch.meshgrid(kx_ids, ky_ids, kz_ids, coil_ids, indexing="ij")
            kspace_ids = torch.stack(kspace_ids, dim=-1).reshape(-1, len(kspace_ids))

            ##################################################
            # Computing the inputs
            ##################################################
            # Convert indices into normalized coordinates in [-1, 1].
            kspace_coords = torch.zeros((kspace_ids.shape[0], 4), dtype=torch.float)
            kspace_coords[:, 0] = (2 * kspace_ids[:, 0]) / (width - 1) - 1
            kspace_coords[:, 1] = (2 * kspace_ids[:, 1]) / (height - 1) - 1
            kspace_coords[:, 2] = (2 * kspace_ids[:, 2]) / (n_slices - 1) - 1
            kspace_coords[:, 3] = (2 * kspace_ids[:, 3]) / (n_coils - 1) - 1

            self.inputs.append(kspace_coords)

            ##################################################
            # Computing the targets
            ##################################################
            targets = volume_kspace[
                kspace_ids[:, 2], kspace_ids[:, 3], kspace_ids[:, 1], kspace_ids[:, 0]
            ]

            # Compute 0.999 quantile of modulus.
            quant_mod = torch.quantile(torch.abs(torch.view_as_complex(targets)), 0.999)
            targets = targets / quant_mod  # Normalize targets

            self.targets.append(targets)

            ##################################################
            # Add metadata from the current volume
            ##################################################
            # Values from the center region (used in 'predict' method).
            center_vals = tensor_to_complex_np(
                volume_kspace[..., left_idx:right_idx, :]
            )
            # Constant (used in '_plot_info' method).
            plot_cste = quant_mod

            self.metadata[vol_id] = {
                "file": file,
                "mask": mask,
                "shape": (n_slices, n_coils, height, width),
                "plot_cste": plot_cste,
                "norm_cste": quant_mod,
                "center": {
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "vals": center_vals,
                },
            }

        self.inputs = torch.cat(self.inputs, dim=0).float()
        self.targets = torch.cat(self.targets, dim=0).float()

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)


def seed_worker(worker_id):
    """
    Controlling randomness in multi-process data loading. The RNGs are used by
    the RandomSampler to generate random indices for data shuffling.
    """
    # Use `torch.initial_seed` to access the PyTorch seed set for each worker.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)