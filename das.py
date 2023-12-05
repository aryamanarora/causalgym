import torch
import os
import random
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from plotnine import ggplot, geom_point, aes, facet_grid, geom_line, ggtitle, geom_tile, theme, element_text, facet_wrap
from plotnine.scales import scale_x_continuous, scale_fill_cmap, scale_y_reverse, scale_fill_gradient2, scale_fill_gradient
from utils import MODELS, WEIGHTS, Sentence, get_options, make_sentence

# add align-transformers to path
sys.path.append("../align-transformers/")
from models.utils import format_token, sm, count_parameters
from models.configuration_alignable_model import (
    AlignableRepresentationConfig,
    AlignableConfig,
)
from models.alignable_base import AlignableModel
from models.interventions import (
    VanillaIntervention,
    RotatedSpaceIntervention,
    BoundlessRotatedSpaceIntervention,
    TrainbleIntervention
)
from models.layers import LowRankRotateLayer

names = {
    "He": ["John", "Bill", "Joseph", "Patrick", "Ken", "Geoff", "Simon", "Richard", "David", "Michael"],
    "She": ["Sarah", "Mary", "Elizabeth", "Jane", "Kate", "Jennifer", "Susan", "Karen", "Nancy", "Lisa"]
}

completions = [
    "walked.",
    "is tired.", "is excited.", "is ready.", "went home.",
    "is walking.", "ran.", "is running.", "works there.",
    "joined the army.", "plays soccer.", "likes playing games.",
    "said no to me."
]

class LowRankRotatedSpaceIntervention(TrainbleIntervention):
    
    """Intervention in the rotated space."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        rotate_layer = LowRankRotateLayer(embed_dim, kwargs["proj_dim"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.interchange_dim = None
        self.embed_dim = embed_dim
        
    def set_interchange_dim(self, interchange_dim):
        self.interchange_dim = interchange_dim

    def forward(self, base, source):
        batch_size = base.shape[0]
        base = base.reshape(batch_size, -1, self.embed_dim)
        source = source.reshape(batch_size, -1, self.embed_dim)
        rotated_base = self.rotate_layer(base)
        low_rank_approx_base = base - torch.matmul(rotated_base, self.rotate_layer.weight.T)
        rotated_source = self.rotate_layer(source)
        # interchange
        inv_value = rotated_source
        # inverse base
        output = torch.matmul(inv_value, self.rotate_layer.weight.T) + low_rank_approx_base
        output = output.reshape(batch_size, -1)
        return output.to(base.dtype)
    
    def __str__(self):
        return f"LowRankRotatedSpaceIntervention(embed_dim={self.embed_dim})"

def intervention_config(model_type, intervention_type, layer, num_dims, intervention_obj=None):
    if intervention_obj is None:
        intervention_obj = BoundlessRotatedSpaceIntervention if num_dims == -1 else LowRankRotatedSpaceIntervention
    else:
        def func(args, proj_dim):
            return intervention_obj
        intervention_obj = func
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,  # layer
                intervention_type,  # intervention type
                "pos",  # intervention unit
                1,  # max number of unit
            ),
        ],
        alignable_interventions_type=intervention_obj,
        alignable_low_rank_dimension=num_dims
    )
    return alignable_config

def get_last_token(logits, attention_mask):
    last_token_indices = attention_mask.sum(1) - 1
    batch_indices = torch.arange(logits.size(0)).unsqueeze(1)
    return logits[batch_indices, last_token_indices.unsqueeze(1)].squeeze(1)

def experiment(
    model: str,
    steps: int,
    num_dims: int,
    warmup: bool,
    eval_steps: int,
    grad_steps: int,
    batch_size: int
):
    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(
        model,
        revision="main",
        torch_dtype=WEIGHTS.get(model, torch.bfloat16) if device == "cuda:0" else torch.float32,
    ).to(device)
    print(gpt.config.num_hidden_layers)

    # filter names
    for key in names:
        names[key] = [name for name in names[key] if len(tokenizer(name)['input_ids']) == 1]
    print(names)

    def make_pair(batch_size: int=1):
        base, src, labels, base_labels, pos_i = [], [], [], [], []

        # make sentences
        for _ in range(batch_size):
            output = list(names.keys())
            random.shuffle(output)
            choices = [random.choice(names[output[0]]), random.choice(names[output[1]])]
            completion = random.choice(completions)
            base.append(f"<|endoftext|>{choices[0]} {completion}")
            src.append(f"<|endoftext|>{choices[1]} {completion}")
            labels.append(tokenizer.encode(" " + output[1])[0])
            base_labels.append(tokenizer.encode(" " + output[0])[0])

        # tokenize
        pair = [
            tokenizer(base, return_tensors="pt", padding=True).to(device),
            tokenizer(src, return_tensors="pt", padding=True).to(device),
        ]

        # get pos_i
        for i in range(batch_size):
            pos_i.append([1, len(pair[0].input_ids[i]) - 1])

        return pair, torch.tensor(labels), torch.tensor(base_labels), [pos_i,]

    # tokens to log
    tokens = tokenizer.encode(" " + " ".join(list(names.keys())))

    # make evalset
    evalset = []
    for i in range(20):
        evalset.append(make_pair())

    # intervene on each layer
    data = []
    stats = {}
    layer_objs = {}

    for layer_i in tqdm(range(gpt.config.num_hidden_layers)):

        # set up alignable model
        alignable_config = intervention_config(
            type(gpt), "block_output", layer_i, num_dims
        )
        alignable = AlignableModel(alignable_config, gpt)
        alignable.set_device(device)
        alignable.disable_model_gradients()

        # optimizer
        t_total = steps
        warm_up_steps = 0.1 * t_total if warmup else 0
        optimizer_params = []
        for k, v in alignable.interventions.items():
            optimizer_params.append({"params": v[0].rotate_layer.parameters()})
            try:
                optimizer_params.append({"params": v[0].intervention_boundaries, "lr": 1e-2})
            except:
                pass
        optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=t_total,
        )
        print("model trainable parameters: ", count_parameters(alignable.model))
        print("intervention trainable parameters: ", alignable.count_parameters())

        # temperature for boundless
        total_step = 0
        temperature_start = 50.0
        temperature_end = 0.1
        temperature_schedule = (
            torch.linspace(temperature_start, temperature_end, t_total)
            .to(torch.bfloat16)
            .to(device)
        )
        alignable.set_temperature(temperature_schedule[total_step])

        # loss fxn: cross entropy between logits and a single target label
        def calculate_loss(logits, label, step):
            shift_logits = logits.contiguous()
            loss_fct = CrossEntropyLoss()
            shift_labels = label
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            if step >= warm_up_steps:
                try:
                    for k, v in alignable.interventions.items():
                        boundary_loss = 1.0 * v[0].intervention_boundaries.sum()
                    loss += boundary_loss
                except:
                    pass

            return loss

        # train
        iterator = tqdm(range(t_total))
        total_loss = torch.tensor(0.0).to(device)

        for step in iterator:

            # make pair
            pair, label, _, pos_i = make_pair(batch_size=batch_size)

            # inference
            _, counterfactual_outputs = alignable(
                pair[0],
                [pair[1]],
                {"sources->base": (pos_i, pos_i)},
            )

            # get last token logits
            logits = get_last_token(counterfactual_outputs.logits, pair[0].attention_mask)

            # loss and backprop
            loss = calculate_loss(logits, label, step)
            total_loss += loss

            # gradient accumulation
            if total_step % grad_steps == 0:
                # print stats
                stats["loss"] = f"{total_loss.item():.3f}"
                if num_dims == -1:
                    for k, v in alignable.interventions.items():
                        stats["bound"] = f"{v[0].intervention_boundaries.sum() * v[0].embed_dim:.3f}"
                iterator.set_postfix(stats)

                if not (grad_steps > 1 and total_step == 0):
                    total_loss.backward()
                    total_loss = torch.tensor(0.0).to(device)
                    optimizer.step()
                    scheduler.step()
                    alignable.set_zero_grad()
                    alignable.set_temperature(temperature_schedule[total_step])

            # eval
            if step % eval_steps == 0:
                with torch.no_grad():

                    # get boundary
                    eval_loss = 0.0
                    boundary = num_dims
                    if num_dims == -1:
                        for k, v in alignable.interventions.items():
                            boundary = (v[0].intervention_boundaries.sum() * v[0].embed_dim).item()

                    for (pair, label, base_label, pos_i) in evalset:
                        
                        # inference
                        _, counterfactual_outputs = alignable(
                            pair[0],
                            [pair[1]],
                            {"sources->base": (pos_i, pos_i)},
                        )

                        # get last token probs
                        logits = counterfactual_outputs.logits[:, -1]
                        loss = calculate_loss(logits, label, step)
                        eval_loss += loss.item()
                        probs = logits.softmax(-1)

                        # store stats
                        label = format_token(tokenizer, label)
                        base_label = format_token(tokenizer, base_label)
                        for tok in tokens:
                            prob = probs[0, tok].item()
                            stats[format_token(tokenizer, tok)] = f"{prob:.3f}"
                            data.append(
                                {
                                    "step": step,
                                    "src_label": label,
                                    "base_label": base_label,
                                    "label": label + " > " + base_label,
                                    "loss": loss.item(),
                                    "token": format_token(tokenizer, tok),
                                    "label_token": label + " > " + base_label + " " + format_token(tokenizer, tok),
                                    "prob": prob,
                                    "bound": boundary,
                                    "temperature": temperature_schedule[total_step],
                                    "layer": layer_i,
                                    "pos": pos_i,
                                }
                            )
                    
                    # update iterator
                    stats["eval_loss"] = f"{eval_loss / len(evalset):.3f}"
                    iterator.set_postfix(stats)

            total_step += 1
        if layer_i not in layer_objs:
            layer_objs[layer_i] = (alignable, loss.item())
        elif loss.item() < layer_objs[layer_i][1]:
            layer_objs[layer_i] = (alignable, loss.item())

    # make das subdir
    if not os.path.exists("figs/das"):
        os.makedirs("figs/das")

    # print plots
    df = pd.DataFrame(data)
    
    plot = (
        ggplot(df, aes(x="step", y="bound", color="factor(layer)"))
        + geom_line()
        + ggtitle("intervention boundary")
    )
    plot.save("figs/das/bound.pdf")

    plot = (
        ggplot(df, aes(x="step", y="loss", color="factor(label)"))
        + facet_wrap("layer")
        + geom_point(alpha=0.1)
        + geom_line(stat='summary', fun_y=lambda x: x.mean())
        + ggtitle("per-label loss")
    )
    plot.save("figs/das/loss.pdf")

    plot = (
        ggplot(df, aes(x="step", y="prob", color="factor(label_token)"))
        + facet_wrap("layer")
        + geom_point(alpha=0.1)
        + geom_line(stat='summary', fun_y=lambda x: x.mean())
        + ggtitle("per-label probs")
    )
    plot.save("figs/das/prob.pdf")

    # test probe on a sentence
    with torch.no_grad():
        test = tokenizer(
            "<|endoftext|>Jane went home because she was beautiful. My buddy John is my girlfriend's brother and he wants to be a nurse.",
            #  Spain is a beautiful, cute, and handsome country.
            return_tensors="pt",
        ).to(device)

        scores = []
        for layer_i in tqdm(layer_objs):
            for (pair, label, base_label, pos_base) in evalset[:1]:
                label = format_token(tokenizer, label[0])
                base_label = format_token(tokenizer, base_label[0])

                # get interventions
                alignable = layer_objs[layer_i][0]
                alignable.set_device(device)
                
                # run per-pos
                for pair_i in range(2):
                    base_logits = gpt(**pair[pair_i]).logits[0, -1]
                    for pos_i in range(1, len(test.input_ids[0])):
                        location = torch.zeros_like(torch.tensor(pos_base)) + pos_i
                        _, counterfactual_outputs = alignable(
                            pair[pair_i], [test], {"sources->base": (location, pos_base)}
                        )
                        logits = counterfactual_outputs.logits[0, -1]
                        probs = logits.softmax(dim=-1)
                        partial_probs = probs[tokens]
                        partial_probs = partial_probs / partial_probs.sum()
                        for i, token in enumerate(tokens):
                            scores.append({
                                "pos": pos_i,
                                "token": format_token(tokenizer, token),
                                "partial_prob": partial_probs[i].item(),
                                "prob": probs[token].item(),
                                "iit": 1.0 if partial_probs[i] > partial_probs[1 - i] else 0.0,
                                "logit": logits[token].item(),
                                "logitdiff": logits[token].item() - base_logits[token].item(),
                                "layer": layer_i,
                                "label": label,
                                "base_label": base_label,
                            })
                    label, base_label = base_label, label
        
        # plot
        df = pd.DataFrame(scores)
        ticks = list(range(len(test.input_ids[0])))
        labels = [format_token(tokenizer, test.input_ids[0][i]) for i in ticks]
        plot = (ggplot(df, aes(x="pos", y="layer", fill="prob"))
                + facet_grid("label ~ token") + geom_tile()
                + scale_x_continuous(breaks=ticks, labels=labels)
                + scale_fill_gradient(low="white", high="purple")
                + theme(axis_text_x=element_text(rotation=90), figure_size=(10, 10)))
        plot.save("figs/das/prob_per_pos.pdf")
        
        # plot
        plot = (ggplot(df, aes(x="pos", y="layer", fill="logitdiff"))
                + facet_grid("label ~ token") + geom_tile()
                + scale_x_continuous(breaks=ticks, labels=labels)
                + scale_fill_gradient2(low="purple", high="orange", mid="white", midpoint=0)
                + theme(axis_text_x=element_text(rotation=90), figure_size=(10, 10)))
        plot.save("figs/das/logitdiff_per_pos.pdf")

        # print avg score per pos, token
        for pos_i in range(1, len(test.input_ids[0])):
            print(f"pos {pos_i}: {format_token(tokenizer, test.input_ids[0][pos_i])}")
            for label in df["label"].unique():
                print(label)
                df = pd.DataFrame(scores)
                df = df[df["pos"] == pos_i]
                df = df[df["label"] == label]
                df = df.groupby(["label", "base_label", "token"]).mean(numeric_only=True)
                df = df.reset_index()

                # convert df to dict where key is label
                data = {}
                for _, row in df.iterrows():
                    data[row["token"]] = (row["prob"], row["logitdiff"])
                
                # print
                for label in data:
                    print(f"{label:>5}: {data[label][0]:>15.3%} {data[label][1]:>15.3}")
            print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--num_dims", type=int, default=-1)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--grad_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    experiment(**vars(args))


if __name__ == "__main__":
    main()
