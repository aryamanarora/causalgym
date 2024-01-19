from matplotlib import axis
from plotnine import ggplot, aes, geom_line, geom_point, ggtitle, geom_tile, geom_text, facet_wrap, theme, element_text, facet_grid, geom_histogram
from plotnine.scales import scale_x_log10, scale_fill_cmap, scale_x_continuous
import json
import pandas as pd
import glob
import argparse
from eval import augment_data


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
        + geom_tile(aes(fill=metric)) + scale_fill_cmap("Purples", limits=[0,1])
        + geom_text(aes(label=f"{metric}_formatted"), color="black", size=10) + ggtitle(title)
        + facet_wrap("~method")
    )

    # modify x axis labels to use sentence
    if sentence is not None:
        plot += scale_x_continuous(breaks=list(range(len(sentence))), labels=sentence)
        plot += theme(axis_text_x=element_text(rotation=45, hjust=1))
    plot.save(loc, width=10, height=10)


def summarise(directory: str):

    # collect all data
    all_data = []
    for f in glob.glob(f"{args.file}/*.json"):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, default="iia")
    parser.add_argument("--file", type=str)
    args = parser.parse_args()

    if args.plot in ["iia", "acc"]:
        with open(args.file, 'r') as f:
            log = json.load(f)
            if args.plot == "iia":
                plot_per_pos(pd.DataFrame(log["data"]), sentence=log["metadata"]["span_names"])
            elif args.plot == "acc":
                plot_per_pos(pd.DataFrame(log["data"]), sentence=log["metadata"]["span_names"])
    elif args.plot == "benchmark":
        plot_benchmark()
    elif args.plot == "all":
        summarise(args.file)