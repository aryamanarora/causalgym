import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import pandas as pd
from plotnine import ggplot, geom_point, aes, facet_grid, geom_line, ggtitle, geom_tile, theme, element_text, facet_wrap
from plotnine.scales import scale_x_continuous, scale_fill_cmap, scale_y_reverse, scale_fill_gradient2, scale_fill_gradient
import os
from utils import format_token, get_last_token, top_vals
from collections import defaultdict

loss_fct = CrossEntropyLoss()

def save_layer_objs(layer_objs, name):
    layers = defaultdict(dict)
    for layer, alignable in layer_objs.items():
        for key in alignable.interventions:
            layers[layer][key] = alignable.interventions[key][0].state_dict()
    torch.save(layers, f"saved/{name}")

def get_intervention_vals(alignable):
    """Get intervention activations."""
    values = {}
    for key in alignable.interventions:
        intervention_object = alignable.interventions[key][0]
        values[key] = {"base": intervention_object.base_val, "src": intervention_object.src_val}
    # shape: [batch_size, seq_len, proj_dim]
    return values

def calculate_loss(logits, label, step, alignable, warm_up_steps=-1):
    """Calculate cross entropy between logits and a single target label (can be batched)"""
    shift_logits = logits.contiguous()
    shift_labels = label.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    if step >= warm_up_steps:
        try:
            for k, v in alignable.interventions.items():
                boundary_loss = 1.0 * v[0].intervention_boundaries.sum()
            loss += boundary_loss
        except:
            pass

    return loss

@torch.no_grad()
def eval(alignable, tokenizer, evalset, layer_i, step, tokens, num_dims, accuracy=None, method="das"):
    # get boundary
    data, stats = [], {}
    eval_loss = 0.0
    iia = 0
    boundary = num_dims
    if num_dims == -1:
        for k, v in alignable.interventions.items():
            boundary = (v[0].intervention_boundaries.sum() * v[0].embed_dim).item()

    for (pair, src_label, base_label, pos_i, _, _) in evalset:
        # inference
        _, counterfactual_outputs = alignable(
            pair[0],
            [None, pair[1]],
            {"sources->base": ([None, pos_i[0]], [pos_i[0], pos_i[0]])},
        )

        # get last token probs
        logits = get_last_token(counterfactual_outputs.logits, pair[0].attention_mask)
        loss = calculate_loss(logits, src_label, step, alignable)
        eval_loss += loss.item()

        # get probs
        batch_size = logits.shape[0]
        for batch_i in range(batch_size):
            probs = logits[batch_i].softmax(-1)
            if probs[src_label[batch_i]].item() > probs[base_label[batch_i]].item():
                iia += 1

            # store stats
            src_label_p = format_token(tokenizer, src_label[batch_i])
            base_label_p = format_token(tokenizer, base_label[batch_i])
            for i, tok in enumerate(tokens):
                prob = probs[tok].item()
                if len(tokens) <= 3: stats[format_token(tokenizer, tok)] = f"{prob:.3f}"
                data.append(
                    {
                        "method": method,
                        "step": step,
                        "src_label": src_label_p,
                        "base_label": base_label_p,
                        "label": src_label_p + " > " + base_label_p,
                        "loss": loss.item(),
                        "token": format_token(tokenizer, tok),
                        "label_token": src_label_p + " > " + base_label_p + ": " + format_token(tokenizer, tok),
                        "prob": probs[tok].item(),
                        "iia": 1 if probs[src_label[batch_i]].item() > probs[base_label[batch_i]].item() else 0,
                        "logit": logits[batch_i][tok].item(),
                        "bound": boundary,
                        "layer": layer_i,
                        "pos": pos_i[0][batch_i][0] if len(pos_i[0][batch_i]) == 1 else None,
                        "acc": accuracy
                    }
                )
    
    # update iterator
    stats["eval_loss"] = f"{eval_loss / (len(evalset) * batch_size):.3f}"
    stats["iia"] = f"{iia / (len(evalset) * batch_size):.3f}"
    if accuracy is not None:
        stats["acc"] = f"{accuracy:.3%}"
    return data, stats

@torch.no_grad()
def eval_sentence(
    layer_objs,
    df,
    evalset,
    tokenizer,
    tokens,
    sentence,
    step,
    dataset,
    model,
    prefix="",
    plots=False,
    scores=[],
    layer=-1
):
    # sentence
    device = evalset[0].pair[0].input_ids.device
    test = tokenizer(sentence, return_tensors="pt").to(device)

    # check each layer
    for layer_i in layer_objs:
        if layer != -1 and layer_i != layer:
            continue
        
        # get per-token logit ranges
        layer_data = df[df["layer"] == layer_i]
        layer_data = layer_data[layer_data["step"] == step]

        for (pair, src_label, base_label, pos_base, _, _) in evalset[:1]:
            src_label = format_token(tokenizer, src_label[0])
            base_label = format_token(tokenizer, base_label[0])

            sublayer_data = layer_data[layer_data["label"] == src_label + " > " + base_label]
            min_logit_per_token = sublayer_data.groupby("token")["logit"].min()
            max_logit_per_token = sublayer_data.groupby("token")["logit"].max()

            # get interventions
            alignable = layer_objs[layer_i]
            alignable.set_device(device)
            
            # run per-pos
            for pair_i in range(2):
                base_logits = alignable.model(**pair[pair_i]).logits[0, -1]
                for pos_i in range(1, len(test.input_ids[0])):
                    location = torch.zeros_like(torch.tensor(pos_base)) + pos_i

                    _, counterfactual_outputs = alignable(
                        pair[pair_i], [test], {"sources->base": (location.tolist(), pos_base)}
                    )
                    # intervention_vals = get_intervention_vals(alignable)
                    # src_val = list(intervention_vals.values())[0]["src"].mean().item()
                    # base_val = list(intervention_vals.values())[0]["base"].mean().item()

                    logits = counterfactual_outputs.logits[0, -1]
                    probs = logits.softmax(dim=-1)
                    partial_probs = probs[tokens]
                    partial_probs = partial_probs / partial_probs.sum()

                    for i, tok in enumerate(tokens):
                        token = format_token(tokenizer, tok)
                        scores.append({
                            "pos": pos_i,
                            "token": f"p({token})",
                            "partial_prob": partial_probs[i].item(),
                            "prob": probs[tok].item(),
                            "adjusted_logitdiff": max(0, min(1, (logits[tok].item() - min_logit_per_token.loc[token].item()) / (max_logit_per_token.loc[token] - min_logit_per_token.loc[token]))),
                            "iia": 1.0 if partial_probs[i] > partial_probs[1 - i] else 0.0,
                            "logit": logits[tok].item(),
                            "logitdiff": logits[tok].item() - base_logits[tok].item(),
                            "layer": layer_i,
                            "src_label": src_label,
                            # "src_val": src_val,
                            "base_label": base_label,
                            # "base_val": base_val,
                            "label": src_label + " > " + base_label,
                            # "valdiff": src_val - base_val,
                            "step": step
                        })
                src_label, base_label = base_label, src_label
    
    # plot
    if plots:
        scores_df = pd.DataFrame(scores)
        scores_df = scores_df[scores_df["step"] == step]
        ticks = list(range(len(test.input_ids[0])))
        labels = [format_token(tokenizer, test.input_ids[0][i]) for i in ticks]
        if prefix != "":
            prefix = prefix + "_"
        ext = "png" if prefix != "" else "pdf"
        plot = (ggplot(scores_df, aes(x="pos", y="layer", fill="prob"))
                + facet_grid("label ~ token") + geom_tile()
                + scale_x_continuous(breaks=ticks, labels=labels)
                + scale_fill_gradient(low="white", high="purple", limits=(0, 1))
                + theme(axis_text_x=element_text(rotation=90), figure_size=(10, 10))
                + ggtitle(f"{dataset}, {model}: Step {step}"))
        plot.save(f"figs/das/{prefix}prob_per_pos.{ext}")
        
        plot = (ggplot(scores_df, aes(x="pos", y="layer", fill="valdiff"))
                + facet_grid("label ~ token") + geom_tile()
                + scale_x_continuous(breaks=ticks, labels=labels)
                + scale_fill_gradient2(low="purple", high="orange", mid="white", midpoint=0.5)
                + theme(axis_text_x=element_text(rotation=90), figure_size=(10, 10))
                + ggtitle(f"{dataset}, {model}: Step {step}"))
        plot.save(f"figs/das/{prefix}val_per_pos.{ext}")

        # print avg score per pos, token
        if prefix == "":
            for pos_i in range(1, len(test.input_ids[0])):
                print(f"pos {pos_i}: {format_token(tokenizer, test.input_ids[0][pos_i])}")
                for src_label in df["src_label"].unique():
                    print(src_label)
                    df = scores_df
                    df = df[df["pos"] == pos_i]
                    df = df[df["src_label"] == src_label]
                    df = df.groupby(["src_label", "base_label", "token"]).mean(numeric_only=True)
                    df = df.reset_index()

                    # convert df to dict where key is label
                    data = {}
                    for _, row in df.iterrows():
                        data[row["token"]] = (row["prob"], row["src_val"])
                    
                    # print
                    for src_label in data:
                        print(f"{src_label:<10} {data[src_label][0]:>15.3%} {data[src_label][1]:>15.3}")
                print("\n")
    
    return scores, test