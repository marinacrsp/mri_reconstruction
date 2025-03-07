import torch
import numpy as np

class DiagonalGaussianDistribution(object):
    """
    Compute the stdev from half the channels, clamping the values (i guess to avoid exploding gradients before introducing it into the exp)

    """
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters # the input batch; torch.Size([4, 128, 64, 64])
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1) # splits the input into 2 tensors of this shape: torch.Size([4, 64, 64, 64])
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self): # Parametrization trick of VAEs
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, mean = False, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if mean:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar, # formula for KL divergence btw sampled posterior and gaussian (stdev 1.0) 
                                       dim=[1, 2, 3]) #sums the estimated KL divergence for the batch
                # result is of size n batches

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean