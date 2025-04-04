import torch
import numpy as np
import torch.nn as nn

class hash_encoder(nn.Module):
    """
    Class that computes the encoding for a batch of points, based on concatenation of bounding box embeddings at different resolution levels L.
    """
    def __init__(self, levels=10, log2_hashmap_size=12, n_features_per_level=2, n_max=320, n_min=16, n_volumes = 5):
        super(hash_encoder, self).__init__()
        self.l_max = levels
        self.log2_hashmap_size = log2_hashmap_size
        self.n_features_per_level = n_features_per_level
        self.n_max = n_max
        self.n_min = n_min
        self.b = np.exp((np.log(self.n_max) - np.log(self.n_min)) / (self.l_max - 1))
        self.n_volumes = n_volumes

        self.vol_embeddings = nn.ModuleList([
            nn.ModuleList([
                nn.Embedding(self._get_number_of_embeddings(i), self.n_features_per_level) for i in range(levels)
            ]) for _ in range(n_volumes)
        ])
        
    
    def _get_number_of_embeddings(self, level_idx: int) -> int:
        max_size = 2 ** self.log2_hashmap_size
        n_l = int(self.n_min * (self.b ** level_idx).item())
        n_l_embeddings = (n_l + 5) ** 2
        return min(max_size, n_l_embeddings)

    def _bilinear_interp(self, x: torch.Tensor, box_indices: torch.Tensor, box_embedds: torch.Tensor) -> torch.Tensor:
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
    
    
    # def forward(self, points: torch.Tensor) -> torch.Tensor:
    #     # Process a batch of points
    #     self.device = points.device
        
    #     xy_embedded_all = []
    #     xy = points[:,:2]
        
    #     for i in range(self.l_max):
    #         n_l = int(self.n_min * self.b ** i)
            
    #         box_idx, hashed_box_idx = self._get_box_idx(xy, n_l)
            
    #         box_embedds = self.embeddings[i](hashed_box_idx)
            
    #         xy_embedded = self.bilinear_interp(xy, box_idx, box_embedds)
    #         xy_embedded_all.append(xy_embedded)
            
            
    #     xy_embeddings_all = torch.cat(xy_embedded_all, dim=1)
    #     full_embedding = torch.cat((xy_embeddings_all, points[:,2].unsqueeze(-1)), dim=1)
    #     return full_embedding
    
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        self.device = points.device
        hash_feature_size = self.n_features_per_level * self.l_max + 2 # (i.e. 2 * 15 + 2), coil and zcoor need to be considered
        model_input = torch.zeros(points.shape[0], hash_feature_size, device=self.device) #final input to the model after computing the embeddings
        for vol in range(self.n_volumes): # do this per volume, identify all points in batch from same vol
            # mask the points in that volume
            mask_vol = (points[:,0] == vol) 
            reduced_batch = points[mask_vol] 
            
            # select the x,y coordinates to perform the hash encoding
            xy = reduced_batch[:,1:-2] #Input coordinates have dimension 5 (volID, x, y, z, coilID)
            xy_embedded_all = []
            
            for i in range(self.l_max):
                n_l = int(self.n_min * self.b ** i)
                
                box_indices, hashed_box_idx = self._get_box_idx(xy, n_l)
                box_embedds = self.vol_embeddings[vol][i](hashed_box_idx)
                    
                xy_embedded = self._bilinear_interp(xy, box_indices, box_embedds)
                xy_embedded_all.append(xy_embedded) # list of embeddings for -xy- coordinates in vol_id
                
            xy_embeddings_all = torch.cat(xy_embedded_all, dim=1) # transform into single -xy- embedding vector
            full_embedding = torch.cat((xy_embeddings_all, reduced_batch[:,3:]), dim=1) # append the coilID and zcoordinate
            ## Introduce the coordinate corresponding to the kz normalized previously// ignore the coil coordinate
            model_input[mask_vol] = full_embedding
    
        return model_input
    
    def __call__(self, point_coors: torch.Tensor) -> torch.Tensor:
        return self.forward(point_coors)