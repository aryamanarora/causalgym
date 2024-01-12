import sys
import torch

from pyvene.models.layers import LowRankRotateLayer, RotateLayer
from pyvene.models.configuration_intervenable_model import (
    IntervenableRepresentationConfig,
    IntervenableConfig,
)
from pyvene.models.intervenable_base import IntervenableModel
from pyvene.models.interventions import (
    BoundlessRotatedSpaceIntervention,
    VanillaIntervention,
    LowRankRotatedSpaceIntervention,
    CollectIntervention,
    AdditionIntervention,
)
from pyvene.models.basic_utils import sigmoid_boundary


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
            intervention_class = CollectIntervention
        elif num_dims > 0:
            intervention_class = LowRankRotatedSpaceIntervention
    else:
        intervention_class = type(intervention_obj)

    # init
    intervenable_config = IntervenableConfig(
        intervenable_model_type=model_type,
        intervenable_representations=[
            IntervenableRepresentationConfig(
                layer,                                  # layer
                intervention_type,                      # intervention type
                "pos",                                  # intervention unit
                1,                                      # max number of unit
                intervenable_low_rank_dimension=num_dims,  # low rank dimension
            ),
        ],
        intervenable_interventions_type=intervention_class,
        intervenable_interventions=[intervention_obj]
    )
    return intervenable_config


def activation_addition_position_config(
    model_type, intervention_type, 
    layer
):
    intervenable_config = IntervenableConfig(
        intervenable_model_type=model_type,
        intervenable_representations=[
            IntervenableRepresentationConfig(
                layer,             # layer
                intervention_type, # intervention type
                "pos",             # intervention unit
                1                  # max number of unit
            ),
        ],
        intervenable_interventions_type=AdditionIntervention,
    )
    return intervenable_config