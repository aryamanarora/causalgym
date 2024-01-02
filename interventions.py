import sys
import torch

sys.path.append("../align-transformers/")
from models.layers import LowRankRotateLayer, RotateLayer
from models.configuration_alignable_model import (
    AlignableRepresentationConfig,
    AlignableConfig,
)
from models.alignable_base import AlignableModel
from models.interventions import (
    TrainableIntervention,
    VanillaIntervention,
    Intervention,
    BasisAgnosticIntervention,
    _do_intervention_by_swap
)
from models.basic_utils import sigmoid_boundary


# INTERVENTION CONFIGS


def intervention_config(
    model_type, intervention_type,
    layer, num_dims, intervention_obj=None
):
    """Generate intervention config."""

    # which intervention class to use
    intervention_class = VanillaIntervention
    if intervention_obj is None:
        if num_dims == -1:
            intervention_class = BoundlessRotatedSpaceIntervention
        elif num_dims is None:
            intervention_class = CollectActivation
        elif num_dims > 0:
            intervention_class = LowRankRotatedSpaceIntervention
    else:
        intervention_class = type(intervention_obj)

    # init
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,                                  # layer
                intervention_type,                      # intervention type
                "pos",                                  # intervention unit
                1,                                      # max number of unit
                alignable_low_rank_dimension=num_dims,  # low rank dimension
            ),
        ],
        alignable_interventions_type=intervention_class,
        alignable_interventions=[intervention_obj]
    )
    return alignable_config


def activation_addition_position_config(
    model_type, intervention_type, 
    layer
):
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,             # layer
                intervention_type, # intervention type
                "pos",             # intervention unit
                1                  # max number of unit
            ),
        ],
        alignable_interventions_type=AdditionIntervention,
    )
    return alignable_config


# INTERVENTIONS


class AdditionIntervention(BasisAgnosticIntervention):
    
    """Modified."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.interchange_dim = None
        self.embed_dim = embed_dim
        self.subspace_partition = kwargs["subspace_partition"] \
            if "subspace_partition" in kwargs else None
        
    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

    def forward(self, base, source, subspaces=None):
        result = base + source
        return result

    def __str__(self):
        return f"AdditionIntervention(embed_dim={self.embed_dim})"
    

class LowRankRotatedSpaceIntervention(TrainableIntervention):
    
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
        self.base_val = rotated_base.clone()
        low_rank_approx_base = base - torch.matmul(rotated_base, self.rotate_layer.weight.T)
        rotated_source = self.rotate_layer(source)
        self.src_val = rotated_source.clone()
        # interchange
        inv_value = rotated_source
        # inverse base
        output = torch.matmul(inv_value, self.rotate_layer.weight.T) + low_rank_approx_base
        output = output.reshape(batch_size, -1)
        return output.to(base.dtype)
    
    def __str__(self):
        return f"LowRankRotatedSpaceIntervention(embed_dim={self.embed_dim})"


class BoundlessRotatedSpaceIntervention(TrainableIntervention):
    
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


class CollectActivation(Intervention):
    
    """Collect activations."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.interchange_dim = None
        self.subspace_partition = kwargs["subspace_partition"] \
            if "subspace_partition" in kwargs else None
        self.stored_base = None
        self.stored_src = None
        
    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

    def forward(self, base, source):
        self.stored_base = base
        self.stored_src = source
        return base
    
    def get_stored_vals(self):
        return self.stored_base, self.stored_src
    
    def __str__(self):
        return f"CollectActivation()"