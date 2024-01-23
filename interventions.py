import pyvene as pv

def intervention_config(
    model_type, intervention_site,
    layer, num_dims, intervention_obj=None
):
    """Generate intervention config."""

    # which intervention class to use
    intervention_class = pv.VanillaIntervention
    if num_dims == -1:
        intervention_class = pv.BoundlessRotatedSpaceIntervention
    elif num_dims > 0:
        intervention_class = pv.LowRankRotatedSpaceIntervention

    # init
    intervenable_config = pv.IntervenableConfig([
        {
            "layer": layer,
            "component": intervention_site,
            "intervention_type": pv.CollectIntervention
        },
        {
            "layer": layer,
            "component": intervention_site,
            "intervention_type": intervention_class,
            "low_rank_dimension": num_dims
        }
    ])
    return intervenable_config