import torch
from torch.nn import CrossEntropyLoss
from utils import format_token
from tqdm import tqdm
import pandas as pd

loss_fct = CrossEntropyLoss()

def calculate_loss(logits, label, step, alignable, warm_up_steps=-1):
    """loss fxn: cross entropy between logits and a single target label"""
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
def eval(alignable, tokenizer, evalset, layer_i, step, tokens, num_dims):
    # get boundary
    data, stats = [], {}
    eval_loss = 0.0
    boundary = num_dims
    if num_dims == -1:
        for k, v in alignable.interventions.items():
            boundary = (v[0].intervention_boundaries.sum() * v[0].embed_dim).item()

    for (pair, src_label, base_label, pos_i) in evalset:
        # inference
        _, counterfactual_outputs = alignable(
            pair[0],
            [pair[1]],
            {"sources->base": (pos_i, pos_i)},
        )

        # get last token probs
        logits = counterfactual_outputs.logits[:, -1]
        loss = calculate_loss(logits, src_label, step, alignable)
        logits = logits[0]
        eval_loss += loss.item()
        probs = logits.softmax(-1)

        # store stats
        src_label = format_token(tokenizer, src_label)
        base_label = format_token(tokenizer, base_label)
        for i, tok in enumerate(tokens):
            prob = probs[tok].item()
            stats[format_token(tokenizer, tok)] = f"{prob:.3f}"
            data.append(
                {
                    "step": step,
                    "src_label": src_label,
                    "base_label": base_label,
                    "label": src_label + " > " + base_label,
                    "loss": loss.item(),
                    "token": format_token(tokenizer, tok),
                    "label_token": src_label + " > " + base_label + ": " + format_token(tokenizer, tok),
                    "prob": probs[tok].item(),
                    "logit": logits[tok].item(),
                    "bound": boundary,
                    "layer": layer_i,
                    "pos": pos_i,
                }
            )
    
    # update iterator
    stats["eval_loss"] = f"{eval_loss / len(evalset):.3f}"
    return data, stats

@torch.no_grad
def eval_sentence(
    layer_objs,
    df,
    evalset,
    tokenizer,
    alignable,
    tokens,
    sentence,
    prefix=""
):
    # sentence
    device = evalset[0][0][0].input_ids.device
    test = tokenizer(sentence, return_tensors="pt").to(device)
    scores = []

    # check each layer
    for layer_i in tqdm(layer_objs):
        
        # get per-token logit ranges
        layer_data = df[df["layer"] == layer_i]
        layer_data = layer_data[layer_data["step"] == layer_data["step"].max()]

        for (pair, src_label, base_label, pos_base) in evalset[:1]:
            src_label = format_token(tokenizer, src_label[0])
            base_label = format_token(tokenizer, base_label[0])

            sublayer_data = layer_data[layer_data["label"] == src_label + " > " + base_label]
            min_logit_per_token = sublayer_data.groupby("token")["logit"].min()
            max_logit_per_token = sublayer_data.groupby("token")["logit"].max()

            # get interventions
            alignable = layer_objs[layer_i][0]
            alignable.set_device(device)
            
            # run per-pos
            for pair_i in range(2):
                base_logits = alignable.model(**pair[pair_i]).logits[0, -1]
                for pos_i in range(1, len(test.input_ids[0])):
                    location = torch.zeros_like(pos_base) + pos_i
                    _, counterfactual_outputs = alignable(
                        pair[pair_i], [test], {"sources->base": (location, pos_base)}
                    )
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
                            "iit": 1.0 if partial_probs[i] > partial_probs[1 - i] else 0.0,
                            "logit": logits[tok].item(),
                            "logitdiff": logits[tok].item() - base_logits[tok].item(),
                            "layer": layer_i,
                            "src_label": src_label,
                            "base_label": base_label,
                        })
                src_label, base_label = base_label, src_label
    
    # plot
    ticks = list(range(len(test.input_ids[0])))
    labels = [format_token(tokenizer, test.input_ids[0][i]) for i in ticks]
    plot = (ggplot(scores_df, aes(x="pos", y="layer", fill="prob"))
            + facet_grid("src_label ~ token") + geom_tile()
            + scale_x_continuous(breaks=ticks, labels=labels)
            + scale_fill_gradient(low="white", high="purple")
            + theme(axis_text_x=element_text(rotation=90), figure_size=(10, 10)))
    plot.save(f"figs/das/{prefix}_prob_per_pos.pdf")
    
    plot = (ggplot(scores_df, aes(x="pos", y="layer", fill="adjusted_logitdiff"))
            + facet_grid("src_label ~ token") + geom_tile()
            + scale_x_continuous(breaks=ticks, labels=labels)
            + scale_fill_gradient2(low="purple", high="orange", mid="white", midpoint=0.5, limits=(0, 1))
            + theme(axis_text_x=element_text(rotation=90), figure_size=(10, 10)))
    plot.save(f"figs/das/{prefix}_logitdiff_per_pos.pdf")

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
                    data[row["token"]] = (row["prob"], row["logitdiff"])
                
                # print
                for src_label in data:
                    print(f"{src_label:<10} {data[src_label][0]:>15.3%} {data[src_label][1]:>15.3}")
            print("\n")
    
    return pd.DataFrame(scores), test