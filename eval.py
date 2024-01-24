import torch
from torch.nn import CrossEntropyLoss
from utils import get_last_token
import pyvene as pv
from data import Batch

loss_fct = CrossEntropyLoss()


def calculate_loss(logits: torch.tensor, label: torch.tensor):
    """Calculate cross entropy between logits and a single target label (can be batched)"""
    shift_logits = logits.contiguous()
    shift_labels = label.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


@torch.no_grad()
def eval(intervenable: pv.IntervenableModel, evalset: list[Batch], layer_i: int, pos_i: int, strategy: str):
    """Evaluate an intervention on an evalset."""

    data = []
    for batch in evalset:

        # inference
        pos_interv = [[x[pos_i] for x in y] for y in batch.compute_pos(strategy)]
        base_outputs, counterfactual_outputs = intervenable(
            batch.base,
            [None, batch.src],
            {"sources->base": ([None, pos_interv[1]], pos_interv)},
        )

        # get last token probs
        logits = get_last_token(counterfactual_outputs.logits, batch.base['attention_mask'])
        probs = logits.log_softmax(dim=-1)
        base_logits = get_last_token(base_outputs[0].logits, batch.base['attention_mask'])
        base_probs = base_logits.log_softmax(dim=-1)
        loss = calculate_loss(logits, batch.src_labels)

        # get probs
        for batch_i in range(len(batch.pairs)):
            src_label = batch.src_labels[batch_i]
            base_label = batch.base_labels[batch_i]
            riia = 1 if logits[batch_i][src_label].item() > logits[batch_i][base_label].item() else 0
            odds_ratio = (base_probs[batch_i][base_label] - base_probs[batch_i][src_label]) + (probs[batch_i][src_label] - probs[batch_i][base_label])

            # store stats
            data.append({
                "src_label": src_label.item(),
                "base_label": base_label.item(),
                "loss": loss.item(),
                "iia": riia,
                "odds_ratio": odds_ratio.item(),
                "layer": layer_i,
                "pos": pos_i
            })
    
    # summary metrics
    summary = {
        "iia": f"{sum([d['iia'] for d in data]) / len(data):.3f}",
        "odds_ratio": f"{sum([d['odds_ratio'] for d in data]) / len(data):.3f}",
        "eval_loss": f"{sum([d['loss'] for d in data]) / len(data):.3f}",
    }
    
    # update iterator
    return data, summary


def augment_data(data, information):
    """Add information to a list of dicts."""
    for d in data:
        d.update(information)
    return data