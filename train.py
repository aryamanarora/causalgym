import torch
from eval import calculate_loss, eval, eval_sentence
from tqdm import tqdm
from utils import get_last_token
import sys

sys.path.append("../align-transformers/")
from models.basic_utils import format_token, sm, count_parameters

def train_das(
    alignable,
    tokenizer,
    trainset,
    evalset,
    layer_i,
    pos_i,
    num_dims,
    steps,
    warmup,
    eval_steps,
    grad_steps,
    store_weights,
    tokens,
):
    """Train DAS or Boundless DAS on a model."""

    # stuff to store
    weights, data = [], []
    stats = {}

    # optimizer
    if num_dims != 0:
        total_steps = steps
        warm_up_steps = 0.1 * total_steps if warmup else 0
        optimizer_params = []
        for k, v in alignable.interventions.items():
            try:
                optimizer_params.append({"params": v[0].rotate_layer.parameters()})
                optimizer_params.append({"params": v[0].intervention_boundaries, "lr": 1e-2})
            except:
                print("some trainable params not found")
                pass
        optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=warm_up_steps,
        #     num_training_steps=total_steps,
        # )
        print("model trainable parameters: ", count_parameters(alignable.model))
        print("intervention trainable parameters: ", alignable.count_parameters())

        # temperature for boundless
        total_step = 0
        temperature_start = 50.0
        temperature_end = 0.1
        temperature_schedule = (
            torch.linspace(temperature_start, temperature_end, total_steps)
            .to(torch.bfloat16)
            .to(alignable.get_device())
        )
        alignable.set_temperature(temperature_schedule[total_step])
    else:
        total_steps = 1
        warm_up_steps = 0.1 * total_steps if warmup else 0

    # train
    iterator = tqdm(range(total_steps))
    total_loss = torch.tensor(0.0).to(alignable.get_device())

    for step in iterator:
        # get weights
        if store_weights:
            for k, v in alignable.interventions.items():
                try:
                    w = v[0].rotate_layer.weight
                    for i in range(w.shape[0]):
                        for j in range(w.shape[1]):
                            weights.append([step, layer_i, pos_i, w[i, j].item(), i, j])
                except:
                    pass

        # make pair
        (pair, src_label, base_label, pos_interv) = trainset[step]
        for i in range(2):
            # inference
            _, counterfactual_outputs = alignable(
                pair[0],
                [pair[1]],
                {"sources->base": (pos_interv, pos_interv)},
            )

            # get last token logits
            logits = get_last_token(counterfactual_outputs.logits, pair[0].attention_mask)

            # loss and backprop
            loss = calculate_loss(logits, src_label, step, alignable, warm_up_steps)
            total_loss += loss
            
            # swap
            pair[0], pair[1] = pair[1], pair[0]
            src_label, base_label = base_label, src_label

        # gradient accumulation
        if total_step % grad_steps == 0 and num_dims != 0:
            # print stats
            stats["lr"] = scheduler.optimizer.param_groups[0]['lr']
            stats["loss"] = f"{total_loss.item():.3f}"
            if num_dims == -1:
                for k, v in alignable.interventions.items():
                    stats["bound"] = f"{v[0].intervention_boundaries.sum() * v[0].embed_dim:.3f}"
            iterator.set_postfix(stats)

            if not (grad_steps > 1 and total_step == 0):
                total_loss.backward()
                total_loss = torch.tensor(0.0).to(alignable.get_device())
                optimizer.step()
                scheduler.step()
                alignable.set_zero_grad()
                alignable.set_temperature(temperature_schedule[total_step])

        # eval
        if (step % eval_steps == 0 or step == total_steps - 1):
            more_data, more_stats = eval(
                alignable, tokenizer, evalset, step=step,
                layer_i=layer_i, num_dims=num_dims, tokens=tokens
            )
            data.extend(more_data)
            stats.update(more_stats)
            iterator.set_postfix(stats)
            prefix = str(step).zfill(4)

        total_step += 1
    
    # return data
    return alignable, data, weights
