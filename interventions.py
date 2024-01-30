import pyvene as pv
from pyvene.models.layers import LowRankRotateLayer
from pyvene.models.modeling_utils import b_sd_to_bsd, bsd_to_b_sd
import torch

class PooledLowRankRotatedSpaceIntervention(pv.TrainableIntervention, pv.DistributedRepresentationIntervention):

    """Intervention in the rotated space."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        # TODO: put them into a parent class
        self.register_buffer('embed_dim', torch.tensor(self.embed_dim))
        self.register_buffer('interchange_dim', torch.tensor(self.embed_dim))

    def forward(self, base, source, subspaces=None):
        num_unit = (base.shape[1] // int(self.embed_dim))
        base = b_sd_to_bsd(base, num_unit)
        source = b_sd_to_bsd(source, num_unit)
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source).mean(dim=1).unsqueeze(1).repeat(1, base.shape[1], 1)
        output = base + torch.matmul(
            (rotated_source - rotated_base), self.rotate_layer.weight.T
        )
        output = bsd_to_b_sd(output)
        return output.to(base.dtype)

    def __str__(self):
        return f"LowRankRotatedSpaceIntervention(embed_dim={self.embed_dim})"


class CollectIntervention(pv.CollectIntervention):

    """Collect activations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, base, source=None, subspaces=None):
        return base

    def __str__(self):
        return f"CollectIntervention(embed_dim={self.embed_dim})"


def intervention_config(intervention_site, intervention_type, layer, num_dims):
    """Generate intervention config."""
    intervenable_config = pv.IntervenableConfig([
        {
            "layer": layer,
            "component": intervention_site,
            "intervention_type": CollectIntervention
        },
        {
            "layer": layer,
            "component": intervention_site,
            "intervention_type": intervention_type,
            "low_rank_dimension": num_dims
        }
    ])
    return intervenable_config