import sys
import torch

sys.path.append("../align-transformers/")
from models.layers import LowRankRotateLayer, RotateLayer
from models.interventions import (
    TrainbleIntervention,
    VanillaIntervention
)
from models.utils import sigmoid_boundary

class LowRankRotatedSpaceIntervention(TrainbleIntervention):
    
    """Intervention in the rotated space."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        rotate_layer = LowRankRotateLayer(embed_dim, kwargs["proj_dim"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.interchange_dim = None
        self.embed_dim = embed_dim
        
    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

    def forward(self, base, source):
        batch_size = base.shape[0]
        base = base.reshape(batch_size, -1, self.embed_dim)
        source = source.reshape(batch_size, -1, self.embed_dim)
        rotated_base = self.rotate_layer(base)
        low_rank_approx_base = base - torch.matmul(rotated_base, self.rotate_layer.weight.T)
        rotated_source = self.rotate_layer(source)
        # interchange
        inv_value = rotated_source
        # inverse base
        output = torch.matmul(inv_value, self.rotate_layer.weight.T) + low_rank_approx_base
        output = output.reshape(batch_size, -1)
        return output.to(base.dtype)
    
    def __str__(self):
        return f"LowRankRotatedSpaceIntervention(embed_dim={self.embed_dim})"

class BoundlessRotatedSpaceIntervention(TrainbleIntervention):
    
    """Intervention in the rotated space with boundary mask."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        rotate_layer = RotateLayer(embed_dim)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(
            rotate_layer)
        self.subspace_partition = kwargs["subspace_partition"] \
            if "subspace_partition" in kwargs else None
        # TODO: in case there are subspace partitions, we 
        #       need to initialize followings differently.
        self.intervention_boundaries = torch.nn.Parameter(
            torch.tensor([0.5]), requires_grad=True)
        self.temperature = torch.nn.Parameter(torch.tensor(50.0)) 
        self.embed_dim = embed_dim
        self.intervention_population = torch.nn.Parameter(
            torch.arange(0, self.embed_dim), requires_grad=False)

    def get_boundary_parameters(self):
        return self.intervention_boundaries

    def get_temperature(self):
        return self.temperature

    def set_temperature(self, temp: torch.Tensor):
        self.temperature.data = temp
        
    def set_interchange_dim(self, interchange_dim):
        """interchange dim is learned and can not be set"""
        assert False

    def forward(self, base, source, subspace=None):
        batch_size = base.shape[0]
        base = base.reshape(batch_size, -1, self.embed_dim)
        source = source.reshape(batch_size, -1, self.embed_dim)
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)
        # get boundary
        intervention_boundaries = torch.clamp(
            self.intervention_boundaries, 1e-3, 1)
        boundary_mask = sigmoid_boundary(
            self.intervention_population.repeat(batch_size, base.shape[1], 1), 
            0.,
            intervention_boundaries[0] * int(self.embed_dim),
            self.temperature
        )
        boundary_mask = torch.ones((batch_size, base.shape[1]), device=base.device).unsqueeze(dim=-1)*boundary_mask
        boundary_mask = boundary_mask.to(rotated_base.dtype)
        # interchange
        rotated_output = (1. - boundary_mask)*rotated_base + boundary_mask*rotated_source
        # inverse output
        output = torch.matmul(rotated_output, self.rotate_layer.weight.T)
        output = output.reshape(batch_size, -1)
        return output.to(base.dtype)
    
    def __str__(self):
        return f"BoundlessRotatedSpaceIntervention(embed_dim={self.embed_dim})"