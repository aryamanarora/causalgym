from plotnine import ggplot, aes, geom_line, geom_point, ggtitle, geom_tile, geom_text, facet_wrap, theme, element_text, geom_smooth, facet_grid
from plotnine.scales import scale_x_log10, scale_fill_cmap, scale_x_continuous, scale_fill_gradient2
import json
import pandas as pd
import glob
import argparse
from data import Dataset
from eval import augment_data
from itertools import combinations
import torch


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


def plot_per_pos(df: pd.DataFrame, metric="iia", loc="figs/das/pos_iia.pdf", sentence=None):
    """Plot position iia for DAS."""

    # get last step
    title = f"position {metric}"
    last_step = df["step"].max()
    df = df[(df["step"] == last_step) | (df["step"] == -1)]
    print(df["method"].unique())
    
    # group df by pos and layer
    df = df[["pos", "layer", metric, "method"]]
    df = df.groupby(["pos", "layer", "method"]).mean().reset_index()
    df[f"{metric}_formatted"] = df[metric].apply(lambda x: f"{x:.2f}")

    # plot
    plot = (
        ggplot(df, aes(x="pos", y="layer"))
        + geom_tile(aes(fill=metric))
        + geom_text(aes(label=f"{metric}_formatted"), color="black", size=7) + ggtitle(title)
        + facet_wrap("~method")
    )
    if metric != "odds_ratio":
        plot += scale_fill_cmap("Purples", limits=[0,1])
    else:
        plot += scale_fill_gradient2(low="orange", mid="white", high="purple", midpoint=0)

    # modify x axis labels to use sentence
    if sentence is not None:
        plot += scale_x_continuous(breaks=list(range(len(sentence))), labels=sentence)
        plot += theme(axis_text_x=element_text(rotation=45, hjust=1))
    plot.save(loc, width=10, height=10)


def plot_cos_sim_per_method(vecs: list[dict], loc="figs/das/cos_sim_method.pdf"):
    paired_cos_sims = []
    for v1 in vecs:
        for v2 in vecs:
            if v1["pos"] != v2["pos"] or v1["method"] != v2["method"]:
                continue
            paired_cos_sims.append({
                "method": v1["method"],
                "layer1": v1["layer"],
                "layer2": v2["layer"],
                "pos": v1["pos"],
                "cos_sim": abs(torch.nn.functional.cosine_similarity(
                    torch.tensor(v1["vec"]).reshape(-1),
                    torch.tensor(v2["vec"]).reshape(-1),
                    dim=0
                ).item())
            })
    
    df = pd.DataFrame(paired_cos_sims)
    plot = (
        ggplot(df, aes(x="layer1", y="layer2", fill="cos_sim")) + geom_tile()
        + facet_grid("method~pos") + scale_fill_cmap("Purples", limits=[0,1])
    )
    plot.save(loc, width=5, height=10)


def plot_cos_sim_per_pos(vecs: list[dict], loc="figs/das/cos_sim_pos.pdf"):
    paired_cos_sims = []
    for v1 in vecs:
        for v2 in vecs:
            if v1["pos"] != v2["pos"] or v1["layer"] != v2["layer"]:
                continue
            paired_cos_sims.append({
                "method1": v1["method"],
                "method2": v2["method"],
                "layer": v1["layer"],
                "pos": v1["pos"],
                "cos_sim": abs(torch.nn.functional.cosine_similarity(
                    torch.tensor(v1["vec"]).reshape(-1),
                    torch.tensor(v2["vec"]).reshape(-1),
                    dim=0
                ).item())
            })
    
    df = pd.DataFrame(paired_cos_sims)
    plot = (
        ggplot(df, aes(x="method1", y="method2", fill="cos_sim")) + geom_tile()
        + facet_grid("layer~pos") + scale_fill_cmap("Purples", limits=[0,1])
    )
    plot.save(loc, width=5, height=10)


def plot_all(directory: str):
    for f in glob.glob(f"{directory}/*.json"):
        with open(f, 'r') as f:
            log = json.load(f)
            dataset = Dataset.load_from(log["metadata"]["dataset"])
            pair = dataset.sample_pair()
            sentence = [pair.base[i] if pair.base[i] == pair.src[i] else pair.base[i] + " / " + pair.src[i] for i in range(len(pair.base))]
            plot_per_pos(pd.DataFrame(log["data"]),
                         sentence=sentence,
                         loc=f"figs/das/{log['metadata']['model'].split('/')[-1]}__{log['metadata']['dataset'].split('/')[1]}.pdf")


def load_directory(directory: str):
    # collect all data
    all_data = []
    for f in glob.glob(f"{directory}/*.json"):
        with open(f, 'r') as f:
            j = json.load(f)
            data = j['data']
            data = augment_data(data, {"dataset": j["metadata"]["dataset"], "model": j["metadata"]["model"]})
            all_data.extend(data)
    
    # df
    df = pd.DataFrame(all_data)
    last_step = df["step"].max()
    df = df[(df["step"] == last_step) | (df["step"] == -1)]
    df["dataset"] = df["dataset"].apply(lambda x: x.split("/")[1])
    return df


def summarise(directory: str):
    # collect all data
    df = load_directory(directory)

    # get average iia over layers, max'd 
    df = df[["dataset", "model", "method", "layer", "pos", "iia"]]
    df = df.groupby(["dataset", "model", "method", "layer", "pos"]).mean().reset_index()
    df = df.groupby(["dataset", "model", "method", "layer"]).max().reset_index()
    df = df.groupby(["dataset", "model", "method"]).mean().reset_index()

    # make latex table
    for model in df["model"].unique():
        split = df[df["model"] == model][["dataset", "method", "iia"]]
        split["dataset"] = split["dataset"].apply(lambda x: "\\texttt{" + x.replace("_", "\\_") + "}")

        # make table with rows = method, cols = dataset
        split = split.pivot(index="dataset", columns="method", values="iia")
        split = split.reset_index()
        
        # take average over rows and append to bottom
        avg = split.drop(columns=["dataset"]).mean(axis=0)
        avg["dataset"] = "Average"
        avg = avg[["dataset"] + list(avg.drop(columns=["dataset"]).index)]

        # to dict
        avg = avg.to_dict()
        split = pd.concat([split, pd.DataFrame([avg])], ignore_index=True)
        
        # bold the largest per row
        for i, row in split.iterrows():
            # ignore dataset col
            max_val = row[1:].max()
            for col in split.columns:
                # format as percentage
                if col != "dataset":
                    split.loc[i, col] = f"{row[col] * 100:.2f}"
                if row[col] == max_val:
                    split.loc[i, col] = "\\textbf{" + split.loc[i, col] + "}"
        
        # reorder columns by avg, high to low
        order = split.iloc[-1].sort_values(ascending=False).index

        # remove vanilla, place at end
        order = list(order)
        order.remove("vanilla")
        order.remove("dataset")
        order.append("vanilla")
        split = split[["dataset"] + list(order)]

        print(split.to_latex(index=False))


def compare(directory: str):
    # collect all data
    df = load_directory(directory)
    # df = df[df["method"].isin([method1, method2])].reset_index(drop=True)
    df = df[["dataset", "model", "layer", "pos", "method", "iia"]]
    df = df.groupby(["dataset", "model", "layer", "pos", "method"]).mean().reset_index()
    # df = df.pivot(index=["dataset", "model", "layer", "pos"], columns="method", values="iia").reset_index()

    # make columns of method1 and method2 for all possible pairs of methods
    # Create a new DataFrame for paired comparisons
    paired_df = []

    # Iterate over each unique combination of dataset, model, layer, pos
    for _, group in df.groupby(["dataset", "model", "layer", "pos"]):
        # Get all combinations of methods within the group
        model = group["model"].values.tolist()[0]
        for (method1, iia1), (method2, iia2) in combinations(group[["method", "iia"]].values, 2):
            paired_df.append({"model": model, "method1": method1, "method2": method2, "iia1": iia1, "iia2": iia2})
    df = pd.DataFrame(paired_df)

    plot = (
        ggplot(df, aes(x="iia2", y="iia1", fill="model")) + geom_point(alpha=0.2, stroke=0)
        + facet_grid("method1~method2")
    )
    plot.save("figs/das/compare.pdf", width=10, height=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, default="iia")
    parser.add_argument("--file", type=str)
    args = parser.parse_args()

    if args.plot in ["iia", "acc", "odds", "cos_sim_method", "cos_sim_pos"]:
        with open(args.file, 'r') as f:
            data = json.load(f)
            if args.plot == "iia":
                plot_per_pos(pd.DataFrame(data["data"]), sentence=data["metadata"]["span_names"])
            elif args.plot == "acc":
                plot_per_pos(pd.DataFrame(data["data"]), metric="accuracy",
                             sentence=data["metadata"]["span_names"], loc="figs/das/pos_acc.pdf")
            elif args.plot == "odds":
                plot_per_pos(pd.DataFrame(data["data"]), metric="odds_ratio",
                             sentence=data["metadata"]["span_names"], loc="figs/das/pos_odds.pdf")
            elif args.plot == "cos_sim_method":
                plot_cos_sim_per_method(data["vec"], loc="figs/das/cos_sim_method.pdf")
            elif args.plot == "cos_sim_pos":
                plot_cos_sim_per_pos(data["vec"], loc="figs/das/cos_sim_pos.pdf")
    elif args.plot == "benchmark":
        plot_benchmark()
    elif args.plot == "compare":
        compare(args.file)
    elif args.plot == "all":
        plot_all(args.file)
        summarise(args.file)