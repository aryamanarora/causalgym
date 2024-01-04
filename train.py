from re import A
from turtle import forward
from typing import Any, Counter
from sympy import N
import torch
from eval import calculate_loss, eval, eval_sentence
from tqdm import tqdm
from utils import get_last_token
import sys
from collections import defaultdict
from interventions import activation_addition_position_config, intervention_config, AlignableModel, LowRankRotatedSpaceIntervention

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

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
        (pair, src_label, base_label, pos_interv, _, _) = trainset[step]
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


# mean diff
def mean_diff(activations):
    means, counts = {}, defaultdict(int)

    # accumulate
    for activation, label in activations:
        if label not in means:
            means[label] = torch.zeros_like(activation)
        means[label] += activation
        counts[label] += 1

    # calc means
    for k in means:
        means[k] /= counts[k]
    
    # make vector
    vecs = list(means.values())
    vec = vecs[1] - vecs[0]
    return vec / torch.norm(vec), None


def kmeans_diff(activations):
    # fit kmeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit([activation for activation, _ in activations])
    
    # make vector
    vecs = kmeans.cluster_centers_
    vec = torch.tensor(vecs[0] - vecs[1], dtype=torch.float32)
    return vec / torch.norm(vec), None


def pca_diff(activations):
    # fit pca
    pca = PCA(n_components=1).fit([activation for activation, _ in activations])

    # get first component
    vec = torch.tensor(pca.components_[0], dtype=torch.float32)
    return vec / torch.norm(vec), None


def probing_diff(activations):
    labels = set([label for _, label in activations])
    assert len(labels) == 2
    label_0 = list(labels)[0]

    # fit linear model
    weight = torch.nn.Parameter(torch.zeros_like(activations[0][0]))
    weight.requires_grad = True

    # train
    optimizer = torch.optim.Adam([weight], lr=1e-3)
    for epoch in range(1):
        for i, (activation, label) in enumerate(activations):
            logit = torch.matmul(weight, activation)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(1.0 if label == label_0 else 0.0))
            loss.backward()
            optimizer.step()
    
        # print acc
        acc = 0
        for activation, label in activations:
            logit = torch.matmul(weight, activation)
            prob = torch.sigmoid(logit).item()
            if (prob > 0.5 and label == label_0) or (prob < 0.5 and label != label_0):
                acc += 1
        print("linear probing acc:", acc / len(activations))

    weight_vector = weight.detach() / torch.norm(weight.detach())
    weight_vector.requires_grad = False
    return weight_vector, acc / len(activations)


def probing_diff_sklearn(activations):
    # fit lr
    labels = [label for _, label in activations]
    lr = LogisticRegression(random_state=0, max_iter=1000, fit_intercept=False).fit([activation for activation, _ in activations], labels)
    accuracy = lr.score([activation for activation, _ in activations], labels)
    print("accuracy:", accuracy)

    # extract weight
    vec = torch.tensor(lr.coef_[0], dtype=torch.float32)
    return vec / torch.norm(vec), accuracy


method_to_class_mapping = {
    "mean_diff": mean_diff,
    "kmeans": kmeans_diff,
    "probe": probing_diff,
    "probe_sklearn": probing_diff_sklearn,
    "pca": pca_diff
}


def train_feature_direction(method, alignable, tokenizer, trainset, evalset, layer_i, pos_i, intervention_site, tokens):
    """Train/compute an intervention direction on some activations."""
    # collect activations
    activations = []
    with torch.no_grad():
        for (pair, src_label, base_label, pos_interv, src_label_o, base_label_o) in tqdm(trainset):
            # inference
            alignable(
                pair[0],
                [pair[1]],
                {"sources->base": (pos_interv, pos_interv)},
            )

            # get activations
            activations_base, activations_src = list(alignable.interventions.values())[0][0].get_stored_vals()

            # per-batch
            for i in range(2):
                for batch_i in range(activations_base.shape[0]):
                    key = base_label_o[batch_i]
                    activations.append((activations_base[batch_i], key))
                src_label, base_label = base_label, src_label
                src_label_o, base_label_o = base_label_o, src_label_o
                activations_base, activations_src = activations_src, activations_base
    
    # get means
    diff_vector, accuracy = method_to_class_mapping[method](activations)

    # set up addition config
    alignable._cleanup_states()
    intervention = LowRankRotatedSpaceIntervention(activations[0][0].shape[-1], proj_dim=1)
    intervention.set_rotate_layer_weight(diff_vector.unsqueeze(1))
    eval_config = intervention_config(type(alignable.model), intervention_site, layer_i, None, intervention)
    alignable2 = AlignableModel(eval_config, alignable.model)

    # eval
    data, stats = eval(alignable2, tokenizer, evalset, layer_i, 0, tokens, None, accuracy=accuracy)

    # done
    alignable2._cleanup_states()
    return data, stats