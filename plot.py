from plotnine import ggplot, aes, geom_line, geom_point, ggtitle, geom_tile, geom_text, facet_wrap
from plotnine.scales import scale_x_log10
import json
import pandas as pd
import torch


def plot_benchmark():
    data = json.load(open("logs/benchmark.json", "r"))
    df = pd.DataFrame(data)
    plot = (
        ggplot(df, aes(x="factor(parameters)", y="iia", group="dataset"))
        + geom_line()
        + geom_point(fill="black", color="white", size=3)
        + scale_x_log10() + facet_wrap("dataset")
    )
    plot.save("logs/benchmark.pdf", width=10, height=10)


def plot_bounds(df: pd.DataFrame, title="intervention boundary", loc="figs/das/bound.pdf"):
    """Plot boundless DAS boundary dim."""
    plot = (
        ggplot(df, aes(x="step", y="bound", color="factor(layer)"))
        + geom_line()
        + ggtitle(title)
    )
    plot.save(loc)


def plot_label_loss(df: pd.DataFrame, title="per-label loss", loc="figs/das/loss.pdf"):
    """Plot per-label loss for DAS."""
    plot = (
        ggplot(df, aes(x="step", y="loss", color="factor(label)"))
        + facet_wrap("layer")
        + geom_point(alpha=0.1)
        + geom_line(stat='summary', fun_y=lambda x: x.mean())
        + ggtitle(title)
    )
    plot.save(loc)


def plot_label_prob(df: pd.DataFrame, title="per-label prob", loc="figs/das/prob.pdf"):
    """Plot per-label probability for DAS."""
    plot = (
        ggplot(df, aes(x="step", y="prob", color="factor(label_token)"))
        + facet_wrap("layer")
        + geom_point(alpha=0.1)
        + geom_line(stat='summary', fun_y=lambda x: x.mean())
        + ggtitle(title)
    )
    plot.save(loc)


def plot_label_logit(df: pd.DataFrame, title="per-label logit", loc="figs/das/logit.pdf"):
    """Plot per-label logits for DAS."""
    plot = (
        ggplot(df, aes(x="step", y="logit", color="factor(label_token)"))
        + facet_wrap("layer")
        + geom_point(alpha=0.1)
        + geom_line(stat='summary', fun_y=lambda x: x.mean())
        + ggtitle(title)
    )
    plot.save(loc)


def plot_das_cos_sim(layer_objs, title="DAS cosine similarity", loc="figs/das/cos_sim.pdf"):
    # collect data
    directions = {}
    cos_sims = []
    for layer in layer_objs:
        alignable = layer_objs[layer]
        for key in alignable.interventions:
            intervention_object = alignable.interventions[key][0]
            direction = intervention_object.rotate_layer.weight.detach().cpu().reshape(-1)
            directions[layer] = direction
    for layer in directions:
        direction = directions[layer]
        for other_layer in directions:
            cos_sim = torch.nn.functional.cosine_similarity(direction, directions[other_layer], dim=0).mean().abs().item()
            cos_sims.append({"layer": layer, "other_layer": other_layer, "cos_sim": cos_sim})
        directions[layer] = direction
    
    # plot sims
    cos_sims_df = pd.DataFrame(cos_sims)
    plot = (
        ggplot(cos_sims_df, aes(x="layer", y="other_layer", fill="cos_sim"))
        + geom_tile()
        + ggtitle(title)
    )

    # add text for each tile
    for i in range(cos_sims_df.shape[0]):
        plot += geom_text(
            aes(x=cos_sims_df["layer"][i], y=cos_sims_df["other_layer"][i], label=f"{cos_sims_df['cos_sim'][i]:.2f}"),
            size=4,
            color="white"
        )
    plot.save(loc)


if __name__ == "__main__":
    plot_benchmark()