import torch
import numpy as np
import torch.nn as nn

class hash_encoder(nn.Module):
    """
    Class that computes the encoding for a batch of points, based on concatenation of bounding box embeddings at different resolution levels L.
    """
    def __init__(self, levels=10, log2_hashmap_size=12, n_features_per_level=2, n_max=320, n_min=16):
        super(hash_encoder, self).__init__()
        self.l_max = levels
        self.log2_hashmap_size = log2_hashmap_size
        self.n_features_per_level = n_features_per_level
        self.n_max = n_max
        self.n_min = n_min
        self.b = np.exp((np.log(self.n_max) - np.log(self.n_min)) / (self.l_max - 1))

        # Initialize embeddings for each level
        self.embeddings = nn.ModuleList([
            nn.Embedding(self._get_number_of_embeddings(i), self.n_features_per_level)
            for i in range(self.l_max)
        ])
        
    
    def _get_number_of_embeddings(self, level_idx: int) -> int:
        max_size = 2 ** self.log2_hashmap_size
        n_l = int(self.n_min * (self.b ** level_idx).item())
        n_l_embeddings = (n_l + 5) ** 2
        return min(max_size, n_l_embeddings)

    def bilinear_interp(self, x: torch.Tensor, box_indices: torch.Tensor, box_embedds: torch.Tensor) -> torch.Tensor:
        device = x.device
        
        if box_indices.shape[1] > 2:
            weights = torch.norm(box_indices - x[:, None, :], dim=2)
            den = weights.sum(dim=1, keepdim=True)
            
            weights /= den # Normalize weights
            weights = 1-weights # NOTE: More weight is given to vertex closer to the point of interest
            
            weights = weights.to(device)
            box_embedds = box_embedds.to(device)

            Npoints = len(den)
            xi_embedding = torch.zeros((Npoints, self.n_features_per_level), device = device)
            
            for i in range(4): # For each corner of the box
                xi_embedding += weights[:,i].unsqueeze(1) * box_embedds[:,i,:]
                
        else:
            xi_embedding = box_embedds
            
        return xi_embedding
    
    def _get_box_idx(self, points: torch.Tensor, n_l: int) -> tuple:
        
        # Get bounding box indices for a batch of points
        if points.dim() > 1:
            x = points[:,0]
            y = points[:,1]
        else:
            x = points[0]
            y = points[1]

        if self.n_max == n_l:
            box_idx = points
            hashed_box_idx = self._hash(points)
        else:
            # Calculate box size based on the total boxes
            box_width = self.n_max // n_l  # Width of each box
            box_height = self.n_max // n_l  # Height of each box

            x_min = torch.maximum(torch.zeros_like(x), (x // box_width) * box_width)
            y_min = torch.maximum(torch.zeros_like(y), (y // box_height) * box_height)
            x_max = torch.minimum(torch.full_like(x, self.n_max), x_min + box_width)
            y_max = torch.minimum(torch.full_like(y, self.n_max), y_min + box_height)
            
            # Stack to create four corners per point, maintaining the batch dimension
            box_idx = torch.stack([
                torch.stack([x_min, y_min], dim=1),
                torch.stack([x_max, y_min], dim=1),
                torch.stack([x_min, y_max], dim=1),
                torch.stack([x_max, y_max], dim=1)
            ], dim=1)  # Shape: (batch_size, 4, 2)
            
            # Determine if the coordinates can be directly mapped or need hashing
            max_hashtable_size = 2 ** self.log2_hashmap_size
            if max_hashtable_size >= (n_l + 5) ** 2:
                hashed_box_idx, _ = self._to_1D(box_idx, n_l)
            else:
                hashed_box_idx = self._hash(box_idx)
                
        return box_idx, hashed_box_idx
    
    ## Hash encoders
    def _to_1D(self, coors, n_l):

        scale_factor = self.n_max // n_l
        scaled_coords = torch.div(coors, scale_factor, rounding_mode="floor").int()    
        x = scaled_coords[...,0]
        y = scaled_coords[...,1]
        
        return (y * n_l + x), scaled_coords
    
    
    def _hash(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: this function can process upto 7 dim coordinates
        log2T:  logarithm of T w.r.t 2
        """
        device = coords.device
        primes = torch.tensor([
            1,
            2654435761,
            805459861,
            3674653429,
            2097192037,
            1434869437,
            2165219737,
        ], dtype = torch.int64, device=device
        )

        xor_result = torch.zeros(coords.shape[:-1], dtype=torch.int64, device=device)

        for i in range(coords.shape[-1]): # Loop around all possible dimensions of the vector containing the bounding box positions
            xor_result ^= coords[...,i].to(torch.int64)*primes[i]
            
        hash_mask = (1 << self.log2_hashmap_size) - 1
        return xor_result & hash_mask
    
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # Process a batch of points
        self.device = points.device
        
        xy_embedded_all = []
        xy = points[:,:2]
        
        for i in range(self.l_max):
            n_l = int(self.n_min * self.b ** i)
            
            box_idx, hashed_box_idx = self._get_box_idx(xy, n_l)
            
            box_embedds = self.embeddings[i](hashed_box_idx)
            
            xy_embedded = self.bilinear_interp(xy, box_idx, box_embedds)
            xy_embedded_all.append(xy_embedded)
            
            
        xy_embeddings_all = torch.cat(xy_embedded_all, dim=1)
        full_embedding = torch.cat((xy_embeddings_all, points[:,2].unsqueeze(-1)), dim=1)
        return full_embedding
    
    
    def __call__(self, point_coors: torch.Tensor) -> torch.Tensor:
        return self.forward(point_coors)