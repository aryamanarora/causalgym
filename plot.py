from matplotlib import axis
from plotnine import ggplot, aes, geom_line, geom_point, ggtitle, geom_tile, geom_text, facet_wrap, theme, element_text, facet_grid, geom_histogram
from plotnine.scales import scale_x_log10, scale_fill_cmap, scale_x_continuous
import json
import pandas as pd
import torch
from data import Dataset
import argparse


def plot_benchmark():
    data = json.load(open("logs/benchmark.json", "r"))
    df = pd.DataFrame(data)
    df["series"] = (df["model"].str.contains("gpt2"))
    df["series"] = df["series"].apply(lambda x: "gpt2" if x else "pythia")
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


def plot_pos_iia(df: pd.DataFrame, title="position iia", loc="figs/das/pos_iia.pdf", sentence=None):
    """Plot position iia for DAS."""

    # get last step
    last_step = df["step"].max()
    df = df[(df["step"] == last_step) | (df["step"] == -1)]
    print(df["method"].unique())
    
    # group df by pos and layer
    df = df[["pos", "layer", "iia", "method"]]
    df = df.groupby(["pos", "layer", "method"]).mean().reset_index()
    df["iia_formatted"] = df["iia"].apply(lambda x: f"{x:.2f}")

    # plot
    plot = (
        ggplot(df, aes(x="pos", y="layer"))
        + geom_tile(aes(fill="iia")) + scale_fill_cmap("Purples", limits=[0,1])
        + geom_text(aes(label="iia_formatted"), color="black", size=10) + ggtitle(title)
        + facet_wrap("~method")
    )

    # modify x axis labels to use sentence
    if sentence is not None:
        plot += scale_x_continuous(breaks=list(range(len(sentence))), labels=sentence)
        plot += theme(axis_text_x=element_text(rotation=45, hjust=1))

    plot.save(loc, width=10, height=10)


def plot_pos_acc(df: pd.DataFrame, title="position acc", loc="figs/das/pos_acc.pdf", sentence=None):
    """Plot position acc for DAS."""

    # get last step
    last_step = df["step"].max()
    df = df[(df["step"] == last_step) | (df["step"] == -1)]
    
    # group df by pos and layer
    df = df[["pos", "layer", "accuracy", "method"]]
    df = df.groupby(["pos", "layer", "method"]).mean().reset_index()
    df["accuracy_formatted"] = df["accuracy"].apply(lambda x: f"{x:.2f}")

    # plot
    plot = (
        ggplot(df, aes(x="pos", y="layer"))
        + geom_tile(aes(fill="accuracy")) + scale_fill_cmap("Purples", limits=[0,1])
        + geom_text(aes(label="accuracy_formatted"), color="black", size=10) + ggtitle(title)
        + facet_wrap("~method")
    )

    # modify x axis labels to use sentence
    if sentence is not None:
        plot += scale_x_continuous(breaks=list(range(len(sentence))), labels=sentence)
        plot += theme(axis_text_x=element_text(rotation=45, hjust=1))

    plot.save(loc, width=10, height=10)

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
    cos_sims_df["cos_sim_formatted"] = cos_sims_df["cos_sim"].apply(lambda x: f"{x:.2f}")
    plot = (
        ggplot(cos_sims_df, aes(x="layer", y="other_layer"))
        + geom_tile(aes(fill="cos_sim")) + scale_fill_cmap("Purples", limits=[0,1])
        + geom_text(aes(label="cos_sim_formatted"), color="black", size=10)
        + ggtitle(title)
    )
    plot.save(loc)


def plot_weights(weights, title="DAS weights", loc="figs/das/weights.png"):
    """Plot DAS weights."""
    
    df = pd.DataFrame(weights, columns=["step", "layer", "pos", "weight", "dim"])

    # df = df[df["layer"] == 0]
    # print(len(df))
    # plot = (
    #     ggplot(df, aes(x="step", y="weight", group="factor(dim)"))
    #     + facet_grid("layer~pos")
    #     + geom_line(alpha=0.1)
    #     + ggtitle(title)
    # )

    df = df[df["step"] == df["step"].max()]
    print(len(df))
    plot = (
        ggplot(df, aes(x="weight"))
        + geom_histogram(bins=50)
        + facet_grid("layer~pos")
        + ggtitle(title)
    )

    plot.save(loc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, default="pos_iia")
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    with open(args.file, 'r') as f:
        log = json.load(f)
        if args.plot == "pos_iia":
            plot_pos_iia(pd.DataFrame(log["data"]), sentence=log["metadata"]["span_names"])
        elif args.plot == "pos_acc":
            plot_pos_acc(pd.DataFrame(log["data"]), sentence=log["metadata"]["span_names"])