import torch
import fastmri
import numpy as np
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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
    """Mean Squared Error Loss Function of the magnitude of the complex values."""

    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, predictions, targets):
        predictions = torch.view_as_complex(predictions)
        targets = torch.view_as_complex(targets)

        loss = ((predictions - targets).abs()) ** 2

        return self.gamma * torch.mean(loss)

def transform_magnitude(complex_tensor, epsilon):
    """Function that applies the logarithmic transformation to the magnitudes of the complex values
    The input should be a tensor -> complex values
    """
    Tx = torch.zeros_like(complex_tensor, dtype = torch.float)
    
    transformed_mgn = torch.abs(torch.log(torch.abs(torch.view_as_complex(complex_tensor)) + epsilon))
    angle = torch.angle(torch.view_as_complex(complex_tensor))
    
    # Fill in the tensor with the transformed magnitude values
    Tx[...,0] = transformed_mgn*torch.cos(angle)
    Tx[...,1] = transformed_mgn*torch.sin(angle)
    
    return Tx

def inv_transform_magnitude(Tx, epsilon):
    """Function that computes the inverse logarithmic magnitude transformation and rescales back
    The input should be a tensor -> complex values
    """
    iTx = torch.zeros_like(Tx, dtype = torch.float)
    
    T_modulus = torch.abs(torch.view_as_complex(Tx))
    T_angle = torch.angle(torch.view_as_complex(Tx))
    
    iTmodulus = torch.exp(-T_modulus) - epsilon
    
    iTx[...,0] = iTmodulus*torch.cos(T_angle)
    iTx[...,1] = iTmodulus*torch.sin(T_angle)
    
    return iTx
    

class MSELoss_transformed:
    """Mean Squared Error Loss Function with logarithmic transformation."""

    def __init__(self, gamma, epsilon):
        self.gamma = gamma
        self.epsilon = epsilon

    def __call__(self, predictions, targets):
        
        predictions = transform_magnitude(predictions, self.epsilon)
        targets = transform_magnitude(targets, self.epsilon)
        
        loss = ((predictions - targets).abs()) ** 2

        return self.gamma * torch.mean(loss)
    


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
