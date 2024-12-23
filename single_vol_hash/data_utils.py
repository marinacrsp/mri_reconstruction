from typing import Tuple

import numpy as np
import torch


def center_crop(data: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
    """Perform a center crop on the MRI volume."""
    assert (
        data.shape[-2] >= new_shape[0] and data.shape[-1] >= new_shape[1]
    ), "New shape must be smaller than the original one."

    w_from = (data.shape[-2] - new_shape[0]) // 2
    h_from = (data.shape[-1] - new_shape[1]) // 2

    w_to = w_from + new_shape[0]
    h_to = h_from + new_shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def inverse_fft2_shift(kspace: np.ndarray) -> np.ndarray:
    """Perform inverse 2D FFT with appropriate shifting."""
    # Shift zero-frequency component to the expected position for the FFT algorithm.
    kspace_shifted = np.fft.ifftshift(kspace, axes=(-2, -1))
    # Perform the inverse 2D FFT with orthogonal normalization.
    image = np.fft.ifft2(kspace_shifted, norm="ortho")
    # Shift zero-frequency component to the center of the image.
    image_shifted = np.fft.fftshift(image, axes=(-2, -1))

    return image_shifted


def fft2_shift(img: np.ndarray) -> np.ndarray:
    """Perform 2D FFT with appropriate shifting."""
    img_shifted = np.fft.fftshift(img, axes=(-2, -1))
    kspace = np.fft.fft2(img_shifted, norm="ortho")
    kspace_shifted = np.fft.ifftshift(kspace, axes=(-2, -1))

    return kspace_shifted


def preprocess_kspace(volume_kspace: np.ndarray) -> np.ndarray:
    """Preprocess MRI volume by performing cropping (in image space)."""
    # Compute inverse fourier transform (go to image space).
    volume_img = inverse_fft2_shift(volume_kspace)

    # Crop the image.
    shape = (volume_img.shape[-1], volume_img.shape[-1])
    cropped_img = center_crop(volume_img, shape)

    # Compute fourier transform of the cropped image (go back to kspace).
    cropped_kspace = fft2_shift(cropped_img)

    return cropped_kspace


def rss(data: np.ndarray) -> np.ndarray:
    """Compute Root Sum of Squares (RSS) along the 'coil' axis."""
    return np.sqrt((np.abs(data) ** 2).sum(1))


def remove_center(mask: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Remove the center region of the undersampling mask."""
    flat_mask = mask.flatten()
    center_idx = len(flat_mask) // 2

    # Falling edge detector to find the center region boundaries.
    left_idx = right_idx = -1
    for shift in range(center_idx):
        if flat_mask[center_idx + shift].item() == 0.0 and right_idx == -1:
            right_idx = center_idx + shift  # Right edge of center region

        if flat_mask[center_idx - shift].item() == 0.0 and left_idx == -1:
            left_idx = center_idx - shift  # Left edge of center region

        if (left_idx != -1) and (right_idx != -1):
            break  # Both edges found, exit loop

    # Create new mask by zeroing out the detected center region.
    new_mask = torch.ones_like(mask)
    new_mask[..., left_idx:right_idx, :] = 0
    # new_mask[..., left_idx+15:right_idx-15, :] = 0    # NOTE: Uncomment to include parts of the center.
    new_mask = mask * new_mask

    return new_mask, left_idx, right_idx

def normalize_hist (data):
    """Normalize the histogram to [0,1]"""
    data_normalized = (data - np.min(data)) / (np.max(data) -  np.min(data))
    return data_normalized

def rescale_hist (data, min, max):
    """Rescale the normalized and equalized histogram back to its original range [min, max]"""
    data_rescaled = data * (max - min) + min
    return data_rescaled