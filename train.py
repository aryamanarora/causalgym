import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from eval import augment_data, calculate_loss, eval
from utils import get_last_token
from interventions import intervention_config, PooledLowRankRotatedSpaceIntervention
import pyvene as pv
from diff_methods import method_to_class_mapping
from data import Batch

def train_das(
        intervenable: pv.IntervenableModel, trainset: list[Batch], evalset: list[Batch],
        layer_i: int, pos_i: int, strategy: str, eval_steps: int, grad_steps: int):
    """Train DAS or Boundless DAS on a model."""

    # setup
    data, activations, stats = [], [], {}
    total_steps = len(trainset)
    warm_up_steps = 0.1 * total_steps

    # optimizer
    optimizer_params = []
    for k, v in intervenable.interventions.items():
        if isinstance(v[0], pv.LowRankRotatedSpaceIntervention) or isinstance(v[0], PooledLowRankRotatedSpaceIntervention):
            optimizer_params.append({"params": v[0].rotate_layer.parameters()})
        elif isinstance(v[0], pv.BoundlessRotatedSpaceIntervention):
            optimizer_params.append({"params": v[0].rotate_layer.parameters()})
            optimizer_params.append({"params": v[0].intervention_boundaries, "lr": 1e-2})
    optimizer = torch.optim.Adam(optimizer_params, lr=1e-2)
    # print("model trainable parameters: ", count_parameters(intervenable.model))
    # print("intervention trainable parameters: ", intervenable.count_parameters())

    # scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warm_up_steps,
        num_training_steps=total_steps,
    )

    # temperature for boundless
    total_step = 0
    temperature_start = 50.0
    temperature_end = 0.1
    temperature_schedule = (
        torch.linspace(temperature_start, temperature_end, total_steps)
        .to(torch.bfloat16)
        .to(intervenable.get_device())
    )
    intervenable.set_temperature(temperature_schedule[total_step])

    # train
    iterator = tqdm(trainset)
    total_loss = torch.tensor(0.0).to(intervenable.get_device())

    for step, batch in enumerate(iterator):

        # inference
        pos_interv = [[x[pos_i] for x in y] for y in batch.compute_pos(strategy)]
        base_outputs, counterfactual_outputs = intervenable(
            base=batch.base,
            sources=[None, batch.src],
            unit_locations={"sources->base": ([None, pos_interv[1]], pos_interv)},
        )

        # store activations/labels for training non-causal methods
        for batch_i in range(len(batch.pairs)):
            for unit_i in range(base_outputs[-1][batch_i].shape[0]):
                activation = base_outputs[-1][batch_i][unit_i].detach().cpu()
                activations.append((activation, batch.base_types[batch_i]))

        # get last token logits
        logits = get_last_token(counterfactual_outputs.logits, batch.base['attention_mask'])

        # loss and backprop
        loss = calculate_loss(logits, batch.src_labels)
        total_loss += loss

        # gradient accumulation
        if total_step % grad_steps == 0:

            # print stats
            stats["lr"] = scheduler.optimizer.param_groups[0]['lr']
            stats["loss"] = f"{total_loss.item():.3f}"
            for k, v in intervenable.interventions.items():
                if isinstance(v[0], pv.BoundlessRotatedSpaceIntervention):
                    stats["bound"] = f"{v[0].intervention_boundaries.sum() * v[0].embed_dim:.3f}"
            iterator.set_postfix(stats)

            # backward
            if not (grad_steps > 1 and total_step == 0):
                total_loss.backward()
                total_loss = torch.tensor(0.0).to(intervenable.get_device())
                optimizer.step()
                scheduler.step()
                intervenable.set_zero_grad()
                intervenable.set_temperature(temperature_schedule[total_step])

        # eval
        if (step % eval_steps == 0 or step == total_steps - 1):
            more_data, summary = eval(intervenable, evalset, layer_i, pos_i, strategy)
            stats.update(summary)
            iterator.set_postfix(stats)
            data.extend(augment_data(more_data, {"method": "das", "step": step}))

        total_step += 1
    
    # return data
    return intervenable, data, activations


def train_feature_direction(
        method: str, intervenable: pv.IntervenableModel, activations: list[tuple[torch.tensor, str]],
        evalset: list[Batch], layer_i: int, pos_i: int, strategy: str, intervention_site: str):
    """Train/compute and evaluate an intervention direction on some activations."""

    # get diff vector based on method
    labels = [label for _, label in activations]
    activations = [activation.type(torch.float32) for activation, _ in activations]
    diff_vector, accuracy = method_to_class_mapping[method](activations, labels)
    diff_vector = diff_vector.to(intervenable.get_device()).unsqueeze(1)

    # new config
    eval_config = intervention_config(
        intervention_site,
        pv.LowRankRotatedSpaceIntervention if strategy != "all" else PooledLowRankRotatedSpaceIntervention,
        layer_i, 1
    )
    intervenable2 = pv.IntervenableModel(eval_config, intervenable.model)
    intervenable2.set_device(intervenable.get_device())
    intervenable2.disable_model_gradients()
    for k, v in intervenable2.interventions.items():
        if isinstance(v[0], pv.LowRankRotatedSpaceIntervention) or isinstance(v[0], PooledLowRankRotatedSpaceIntervention):
            v[0].rotate_layer.weight = diff_vector

    # eval
    data, summary = eval(intervenable2, evalset, layer_i, pos_i, strategy)
    if accuracy is not None:
        summary["accuracy"] = f"{accuracy:.3%}"

    # done
    intervenable2._cleanup_states()
    return augment_data(data, {"method": method, "step": -1, "accuracy": accuracy}), summary