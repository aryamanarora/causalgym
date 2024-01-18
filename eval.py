import torch
from torch.nn import CrossEntropyLoss
from utils import get_last_token
from interventions import IntervenableModel
from data import Batch

loss_fct = CrossEntropyLoss()


def calculate_loss(logits: torch.tensor, label: torch.tensor):
    """Calculate cross entropy between logits and a single target label (can be batched)"""
    shift_logits = logits.contiguous()
    shift_labels = label.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


@torch.no_grad()
def eval(intervenable: IntervenableModel, evalset: list[Batch], layer_i: int, pos_i: int):
    """Evaluate an intervention on an evalset."""

    data = []
    for batch in evalset:

        # inference
        pos_interv = batch.pos[:, :, pos_i].tolist()
        _, counterfactual_outputs = intervenable(
            batch.base,
            [None, batch.src],
            {"sources->base": ([None, pos_interv[1]], pos_interv)},
        )

        # get last token probs
        logits = get_last_token(counterfactual_outputs.logits, batch.base['attention_mask'])
        loss = calculate_loss(logits, batch.src_labels)

        # get probs
        for batch_i in range(len(batch.pairs)):
            src_label = batch.src_labels[batch_i]
            base_label = batch.base_labels[batch_i]

            # store stats
            data.append({
                "src_label": src_label,
                "base_label": base_label,
                "loss": loss.item(),
                "iia": 1 if logits[batch_i][src_label].item() > logits[batch_i][base_label].item() else 0,
                "layer": layer_i,
                "pos": pos_i
            })
    
    # summary metrics
    summary = {
        "iia": f"{sum([d['iia'] for d in data]) / len(data):.3f}",
        "eval_loss": f"{sum([d['loss'] for d in data]) / len(data):.3f}",
    }
    
    # update iterator
    return data, summary


def augment_data(data, information):
    """Add information to a list of dicts."""
    for d in data:
        d.update(information)
    return data