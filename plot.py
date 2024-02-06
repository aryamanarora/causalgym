from plotnine import (
    ggplot, aes, geom_line, geom_point, ggtitle, geom_tile, theme, element_blank,
    geom_text, facet_wrap, theme, element_text, geom_smooth, facet_grid, theme_bw,
    xlab, ylab, theme_set, theme_gray, stat_summary, geom_hline
)
from plotnine.scales import scale_x_log10, scale_fill_cmap, scale_x_continuous, scale_fill_gradient2
import json
import pandas as pd
import glob
import argparse
from data import Dataset
from itertools import combinations
import torch
from utils import parameters
from tqdm import tqdm
import multiprocessing
import math


theme_set(theme_gray(base_family="serif"))


classification = {
    'agr_gender': 'Agreement',
    'agr_sv_num_subj-relc': 'Agreement',
    'agr_sv_num_obj-relc': 'Agreement',
    'agr_sv_num_pp': 'Agreement',
    'agr_refl_num_subj-relc': 'Licensing',
    'agr_refl_num_obj-relc': 'Licensing',
    'agr_refl_num_pp': 'Licensing',
    'npi_any_subj-relc': 'Licensing',
    'npi_any_obj-relc': 'Licensing',
    'npi_ever_subj-relc': 'Licensing',
    'npi_ever_obj-relc': 'Licensing',
    'garden_mvrr': 'Garden path effects',
    'garden_mvrr_mod': 'Garden path effects',
    'garden_npz_obj': 'Garden path effects',
    'garden_npz_obj_mod': 'Garden path effects',
    'garden_npz_v-trans': 'Garden path effects',
    'garden_npz_v-trans_mod': 'Garden path effects',
    'gss_subord': 'Gross syntactic state',
    'gss_subord_subj-relc': 'Gross syntactic state',
    'gss_subord_obj-relc': 'Gross syntactic state',
    'gss_subord_pp': 'Gross syntactic state',
    'cleft': 'Long-distance',
    'cleft_mod': 'Long-distance',
    'filler_gap_embed_3': 'Long-distance',
    'filler_gap_embed_4': 'Long-distance',
    'filler_gap_hierarchy': 'Long-distance',
    'filler_gap_obj': 'Long-distance',
    'filler_gap_pp': 'Long-distance',
    'filler_gap_subj': 'Long-distance',
}
classification_order = ['Agreement', 'Licensing', 'Garden path effects', 'Gross syntactic state', 'Long-distance']


def load_file(file_path):
    with open(file_path, 'r') as f:
        j = json.load(f)
        data = j['data']
        df = pd.DataFrame(data)
        df["dataset"] = j["metadata"]["dataset"].split("/")[1]
        df["model"] = j["metadata"]["model"]
        return df


def load_directory(directory: str, reload: bool=False, filter_step: bool=True):
    if reload or not glob.glob(f"{directory}/combined.csv"):
        print(f"reloading {directory}")
        # load all files (in parallel for speedup)
        file_paths = glob.glob(f"{directory}/*.json")
        dfs = []
        with multiprocessing.Pool() as pool:
            for df in tqdm(pool.imap_unordered(load_file, file_paths), total=len(file_paths)):
                # summary stats
                df["acc"] = df["base_p_src"] < df["base_p_base"]
                df["iia"] = (df["p_src"] > df["p_base"]) * 100
                df["odds"] = df['base_p_base'] - df['base_p_src'] + df['p_src'] - df['p_base']
                df = df[["dataset", "step", "model", "method", "layer", "pos", "odds", "iia", "acc"]]

                # store
                dfs.append(df)
        
        # merge
        df = pd.concat(dfs, ignore_index=True)
        df = df.groupby(["dataset", "step", "model", "method", "layer", "pos"]).mean().reset_index()

        # final formatting
        df["model"] = df["model"].apply(lambda x: x.split("/")[-1])
        model_order = [x.split("/")[-1] for x in list(parameters.keys())[::-1]]
        df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
        df.to_csv(f"{directory}/combined.csv", index=False)
    else:
        print(f"using existing {directory}")
        df = pd.read_csv(f"{directory}/combined.csv")

    if filter_step:
        last_step = df["step"].max()
        df = df[(df["step"] == last_step) | (df["step"] == -1)]
        df.drop(columns=["step"], inplace=True)
    return df


def plot_acc(directory: str, reload: bool=False):
    """Plot raw accuracy for each model at each task."""

    # compute acc
    df = load_directory(directory, reload)
    df = df[df["method"] == "vanilla"]
    df = df[["dataset", "model", "acc"]]
    df = df.groupby(["dataset", "model"]).mean().reset_index()
    df["params"] = df["model"].apply(lambda x: parameters[x])
    df["type"] = df["dataset"].apply(lambda x: classification[x])
    df["type"] = pd.Categorical(df["type"], categories=classification_order, ordered=True)
    
    # plot
    plot = (
        ggplot(df, aes(x="params", y="acc"))
        + geom_line(aes(group="dataset"), alpha=0.2) + theme(
            axis_text_x=element_text(rotation=90, hjust=0.5),
            panel_grid_minor=element_blank())
        + xlab("Parameters") + ylab("Accuracy")
        # + geom_point(color="black", fill="white", size=2)
        + stat_summary()
        + stat_summary(geom="line")
        + scale_x_log10()
        + facet_wrap("type", nrow=1)
        + geom_hline(yintercept=0.5, linetype="dashed")
    )
    plot.save(f"{directory}/figs_acc.pdf", width=8, height=2.5)


def plot_per_pos(directory: str, reload: bool=False, metric: str="iia"):
    """Plot position iia for DAS."""

    # load
    df = load_directory(directory, reload)
    df = df[["dataset", "model", "method", "layer", "pos", metric]]
    print(len(df))

    # plot
    for dataset in df["dataset"].unique():
        plot = (
            ggplot(df[df["dataset"] == dataset], aes(x="pos", y="layer"))
            + geom_tile(aes(fill=metric))
            # + geom_text(aes(label=f"{metric}_formatted"), color="black", size=7) + ggtitle(metric)
            + facet_wrap("~method", scales="free_y")
        )
        if metric != "odds":
            plot += scale_fill_cmap("Purples", limits=[0,100])
        else:
            plot += scale_fill_gradient2(low="orange", mid="white", high="purple", midpoint=0)

        # modify x axis labels to use sentence
        sentence = Dataset.load_from(f"syntaxgym/{dataset}").span_names
        if sentence is not None:
            plot += scale_x_continuous(breaks=list(range(len(sentence))), labels=sentence)
            plot += theme(axis_text_x=element_text(rotation=90, hjust=0.5))
        plot.save(f"{directory}/figs_{dataset.replace('/', '_')}_{metric}.pdf", width=20, height=10)


def summarise(directory: str, reload: bool=False, metric: str="odds"):
    # collect all data
    df = load_directory(directory, reload)

    # get average iia over layers, max'd 
    df = df[["dataset", "model", "method", "layer", "pos", metric]]
    df.drop(columns=["pos"], inplace=True)
    df = df.groupby(["dataset", "model", "method", "layer"]).max().reset_index()
    df.drop(columns=["layer"], inplace=True)
    df = df.groupby(["dataset", "model", "method"]).mean().reset_index()

    # make latex table
    for model in df["model"].unique():
        split = df[df["model"] == model][["dataset", "method", metric]]
        split["dataset"] = split["dataset"].apply(lambda x: "\\texttt{" + x.replace("_", "\\_") + "}")

        # make table with rows = method, cols = dataset
        split = split.pivot(index="dataset", columns="method", values=metric)
        split = split.reset_index()
        
        # take average over rows and append to bottom
        avg = split.drop(columns=["dataset"]).mean(axis=0)
        avg["dataset"] = "Average"
        avg = avg[["dataset"] + list(avg.drop(columns=["dataset"]).index)]

        # to dict
        avg = avg.to_dict()
        split = pd.concat([split, pd.DataFrame([avg])], ignore_index=True)
        
        # reorder columns by avg, high to low
        order = ["das", "mean", "probe", "pca", "kmeans", "lda", "random", "vanilla"]
        split = split[["dataset"] + list(order)]
        
        # bold the largest per row
        for i, row in split.iterrows():
            # ignore dataset col
            max_val = row[1:].max()
            for col in split.columns:
                # format as percentage
                if col != "dataset":
                    split.loc[i, col] = f"{float(row[col]):.2f}"
                if row[col] == max_val:
                    split.loc[i, col] = "\\textbf{" + split.loc[i, col] + "}"

        with open(f"{directory}/{model.replace('/', '_')}__{metric}.txt", "w") as f:
            f.write(split.to_latex(index=False))
            print("wrote", model, metric)


def average_per_method(directory: str, reload: bool=False, metric: str="odds"):
    # collect all data
    df = load_directory(directory, reload, filter_step=False)

    # get average iia over layers, max'd 
    df = df[["dataset", "step", "model", "method", "layer", "pos", metric]]
    df.drop(columns=["pos"], inplace=True)
    df = df.groupby(["dataset", "step", "model", "method", "layer"]).max().reset_index()
    df.drop(columns=["layer"], inplace=True)
    df = df.groupby(["dataset", "step", "model", "method"]).mean().reset_index()
    df.drop(columns=["dataset"], inplace=True)
    df = df.groupby(["model", "method", "step"]).mean().reset_index()

    for model in df["model"].unique():
        split = df[df["model"] == model]
        split.drop(columns=["model"], inplace=True)
        split = split.sort_values(by=metric, ascending=False)
        with open(f"{directory}/{model.replace('/', '_')}__{metric}__avg.txt", "w") as f:
            f.write(split.to_latex(index=False))
            print("wrote", model, metric)


def probe_hyperparam_plot(directory: str, reload: bool=False, metric: str="odds"):
    # collect all data
    df = load_directory(directory, reload)

    # get average iia over layers, max'd 
    df = df[["dataset", "model", "method", "layer", "pos", metric]]
    df.drop(columns=["pos"], inplace=True)
    df = df.groupby(["dataset", "model", "method", "layer"]).max().reset_index()
    df.drop(columns=["layer"], inplace=True)
    df = df.groupby(["dataset", "model", "method"]).mean().reset_index()
    df.drop(columns=["dataset"], inplace=True)
    df = df.groupby(["model", "method"]).mean().reset_index()

    # filter
    df = df[df["method"].str.contains("probe_l2_int")]
    df["$\lambda$"] = df["method"].apply(lambda x: 1 / float(x.split("_")[-1]))
    df["params"] = df["model"].apply(lambda x: parameters[x])

    # plot
    plot = (
        ggplot(df, aes(x="$\lambda$", y=metric, group="model"))
        + geom_line(aes(color="model"))
        + geom_point(aes(color="model"))
        + scale_x_log10()
    )
    plot.save(f"{directory}/figs_probe_hyperparam.pdf", width=5, height=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, default="acc")
    parser.add_argument("--metric", type=str, default="iia")
    parser.add_argument("--file", type=str)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # base accuracy of each model on each task
    if args.plot == "acc":
        plot_acc(args.file, reload=args.reload)
    elif args.plot == "summary":
        summarise(args.file, args.reload, args.metric)
    elif args.plot == "avg":
        average_per_method(args.file, args.reload, args.metric)
    elif args.plot == "pos":
        plot_per_pos(args.file, args.reload, args.metric)
    elif args.plot == "probe_hyperparam":
        probe_hyperparam_plot(args.file, args.reload, args.metric)