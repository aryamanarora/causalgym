from plotnine import (
    ggplot, aes, geom_line, geom_point, ggtitle, geom_tile, theme, element_blank,
    geom_text, facet_wrap, theme, element_text, geom_smooth, facet_grid, theme_bw,
    xlab, ylab, theme_set, theme_gray, stat_summary, geom_hline, theme_bw, element_rect,
    theme_void, geom_boxplot
)
from plotnine.scales import (
    scale_x_log10, scale_fill_cmap, scale_x_continuous, scale_fill_gradient2, xlim,
    scale_y_continuous, scale_y_reverse, scale_color_cmap, scale_color_gradient2
)
import json
import pandas as pd
import glob
import argparse
from data import Dataset
import numpy as np
from utils import parameters
from tqdm import tqdm
import multiprocessing
import math


theme_set(theme_bw(base_family="Nimbus Roman", base_size=12)
    + theme(
        axis_text_x=element_text(rotation=90, hjust=0.5),
        panel_border=element_rect(fill="None", color="#000", size=0.5, zorder=-1000000),
        legend_key=element_rect(color="None"),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        strip_background=element_rect(color="None", fill="None"),))


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
for key in list(classification.keys()):
    classification[key + "_inverted"] = classification[key]
classification_order = ['Agreement', 'Licensing', 'Garden path effects', 'Gross syntactic state', 'Long-distance']
model_order = [x for x in list(parameters.keys())[::-1]]
method_order = ["das", "probe", "probe_0", "probe_1", "mean", "pca", "kmeans", "lda", "random", "vanilla"]


def pick_better_probe(orig_df: pd.DataFrame, metrics: list[str]):
    # get average iia over layers, max'd
    df = orig_df.copy()
    df = df[["dataset", "model", "method", "layer", "pos"] + metrics]
    df.drop(columns=["pos"], inplace=True)
    df = df.groupby(["dataset", "model", "method", "layer"]).max().reset_index()
    df.drop(columns=["dataset"], inplace=True)
    df = df.groupby(["model", "method", "layer"]).mean().reset_index()

    # remove nans
    df = df.dropna()

    # pick overall better from probe_0 and probe_1
    per_method_avg = df.groupby(["model", "method"]).mean().reset_index()
    for model in per_method_avg["model"].unique():
        if model in ["pythia-14m", "pythia-31m", "pythia-70m"]:
            orig_df.loc[orig_df["model"] == model, "method"] = orig_df[orig_df["model"] == model]["method"].apply(lambda x: "probe" if x == "probe_0" else x)
            continue
        probe_0 = per_method_avg[(per_method_avg["model"] == model) & (per_method_avg["method"] == "probe_0")][metrics[0]].values[0]
        probe_1 = per_method_avg[(per_method_avg["model"] == model) & (per_method_avg["method"] == "probe_1")][metrics[0]].values[0]
        if math.isnan(probe_0) or math.isnan(probe_1):
            continue
        elif probe_0 > probe_1:
            orig_df.loc[orig_df["model"] == model, "method"] = orig_df[orig_df["model"] == model]["method"].apply(lambda x: "probe" if x == "probe_0" else x)
        else:
            orig_df.loc[orig_df["model"] == model, "method"] = orig_df[orig_df["model"] == model]["method"].apply(lambda x: "probe" if x == "probe_1" else x)
    return orig_df


def load_file(file_path):
    with open(file_path, 'r') as f:
        j = json.load(f)

        # model name
        model_name = j["metadata"]["model"]
        if file_path.split("_")[-1].startswith("step"):
            model_name += "_" + file_path.split("_")[-1].split(".json")[0]
        model_name = model_name.replace("_step", "\nstep")

        # dataset name
        dataset_name = j["metadata"]["dataset"].split("/")[1]
        if j["metadata"].get("invert_labels", False):
            dataset_name += "_inverted"

        data = j['data']
        df = pd.DataFrame(data)
        df["dataset"] = dataset_name
        df["model"] = model_name
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
                if "accuracy" not in df.columns:
                    df["accuracy"] = np.nan
                df = df[["dataset", "step", "model", "method", "layer",
                         "base_p_base", "base_p_src", "p_src", "p_base",
                         "pos", "odds", "iia", "acc", "accuracy"]]
                df["base_p_base"] = df["base_p_base"].apply(lambda x: math.exp(x))
                df["base_p_src"] = df["base_p_src"].apply(lambda x: math.exp(x))
                df["p_base"] = df["p_base"].apply(lambda x: math.exp(x))
                df["p_src"] = df["p_src"].apply(lambda x: math.exp(x))

                # store
                dfs.append(df)
        
        # merge
        df = pd.concat(dfs, ignore_index=True)
        df = df.groupby(["dataset", "step", "model", "method", "layer", "pos"]).mean().reset_index()

        # final formatting
        df["model"] = df["model"].apply(lambda x: x.split("/")[-1])
        df.to_csv(f"{directory}/combined.csv", index=False)
    else:
        print(f"using existing {directory}")
        df = pd.read_csv(f"{directory}/combined.csv")

    if filter_step:
        last_step = df["step"].max()
        df = df[(df["step"] == last_step) | (df["step"] == -1)]
        df.drop(columns=["step"], inplace=True)
    for model in sorted(list(df["model"].unique()), key=lambda x: int(x.split("step")[-1]) if "step" in x else 143000):
        print(model)
        if model not in model_order:
            model_order.append(model)
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=list(classification.keys()), ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df["trainstep"] = df["model"].apply(lambda x: int(x.split("step")[-1]) if "step" in x else 143000)
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
    df.dropna(inplace=True)
    print(df)
    
    # plot
    plot = (
        ggplot(df, aes(x="params", y="acc"))
        + geom_line(aes(group="dataset"), alpha=0.2) + theme(
            axis_text_x=element_text(rotation=90, hjust=0.5),
            panel_grid_minor=element_blank())
        + xlab("Parameters") + ylab("Accuracy")
        # + geom_point(color="black", fill="white", size=2)
        + stat_summary(group="type")
        + stat_summary(group="type", geom="line")
        + scale_x_log10()
        + facet_wrap("type", nrow=1)
        + geom_hline(yintercept=0.5, linetype="dashed")
    )
    plot.save(f"{directory}/figs_acc.pdf", width=8, height=2.5)


def plot_per_pos(directory: str, reload: bool=False, metric: str="iia", plot_all: bool=False):
    """Plot position iia for DAS."""

    # load
    df = load_directory(directory, reload)
    df = df[["dataset", "model", "method", "layer", "pos", "acc", metric]]

    # pick overall better from probe_0 and probe_1
    df = pick_better_probe(df, [metric])
    if not plot_all:
        if metric in ["iia", "odds"]:
            df = df[df["method"].isin(["das"])]
        else:
            df = df[df["method"].isin(["probe", "lda"])]
    print(len(df))

    # plot
    for dataset in df["dataset"].unique():
        # modify x axis labels to use sentence
        dataset_src = Dataset.load_from(f"syntaxgym/{dataset}")
        pair = dataset_src.sample_pair()
        sentence = [pair.base[i] if pair.base[i] == pair.src[i] else pair.base[i] + ' / ' + pair.src[i] for i in range(len(pair.base))]
        dataset_df = df[df["dataset"] == dataset]

        # check sentence length
        rows = []
        for i in range(len(sentence)):
            if len(dataset_df[dataset_df["pos"] == i]) == 0:
                for model in dataset_df["model"].unique():
                    default_val = 0
                    acc = dataset_df[dataset_df["model"] == model]["acc"].mean()
                    if metric == "iia":
                        default_val = (1 - acc) * 100
                    elif metric == "accuracy":
                        default_val = 0.5
                    for layer in dataset_df[dataset_df["model"] == model]["layer"].unique():
                        for method in dataset_df["method"].unique():
                            row = {"dataset": dataset, "model": model, "layer": layer, "pos": i, "method": method, "acc": acc, metric: default_val}
                            rows.append(row)

        # add rows to df
        dataset_df = pd.concat([dataset_df, pd.DataFrame(rows)])
        dataset_df = dataset_df[dataset_df["method"].isin(["vanilla", "das", "probe", "mean", "pca", "kmeans", "lda", "random"])]
        dataset_df["model"] = pd.Categorical(dataset_df["model"], categories=model_order, ordered=True)
        dataset_df["method"] = pd.Categorical(dataset_df["method"], categories=method_order, ordered=True)

        plot = (
            ggplot(dataset_df, aes(x="layer", y="pos"))
            + geom_tile(aes(fill=metric, color=metric))
            + facet_grid("method~model", scales="free_x")
        )
        if metric == "iia":
            plot += scale_fill_cmap("Purples", limits=[0,100])
            plot += scale_color_cmap("Purples", limits=[0,100])
        elif metric == "accuracy":
            plot += scale_fill_cmap("Purples", limits=[0,1])
            plot += scale_color_cmap("Purples", limits=[0,1])
        else:
            plot += scale_fill_gradient2(low="orange", mid="white", high="purple", midpoint=0)
            plot += scale_color_gradient2(low="orange", mid="white", high="purple", midpoint=0)

        if sentence is not None:
            plot += scale_y_reverse(breaks=list(range(len(sentence))), labels=sentence, expand=[0, 0])
            plot += scale_x_continuous(expand=[0, 0])
            plot += theme(
                axis_text_x=element_text(rotation=90, hjust=0.5),
                axis_text_y=element_text(size=8),
                panel_border=element_rect(fill="None", color="#000", size=1),
                strip_background=element_rect(color="None", fill="None"),
                strip_text_x=element_text(size=9),
                legend_key_height=10,
            )
        height = 1.8
        if metric == "accuracy" and not plot_all:
            height = 2.5
        if plot_all:
            height = 6
        plot.save(f"{directory}/figs_{dataset.replace('/', '_')}_{metric}{'_all' if plot_all else ''}.pdf", width=8, height=height)


def summarise(directory: str, reload: bool=False, metric: str="odds"):
    # collect all data
    df = load_directory(directory, reload)

    # get model/task acc
    task_acc = df[["dataset", "model", "acc"]]
    task_acc = task_acc.groupby(["dataset", "model"]).mean().reset_index()

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
        order = ["das", "probe_0", "probe_1", "mean", "pca", "kmeans", "lda", "random", "vanilla"]
        if model in ["pythia-14m", "pythia-31m", "pythia-70m"]:
            order.remove("probe_1")
        split = split[["dataset"] + list(order)]

        # add an acc column using task_acc
        acc = task_acc[task_acc["model"] == model]
        acc = acc[["dataset", "acc"]]
        acc["dataset"] = acc["dataset"].apply(lambda x: "\\texttt{" + x.replace("_", "\\_") + "}")
        split = split.merge(acc, on="dataset", how="left")
        split = split[["dataset", "acc"] + list(order)]

        # add avg acc to the last row
        avg_acc = acc["acc"].mean()
        split.loc[split.index[-1], "acc"] = round(avg_acc, 2)
        
        # bold the largest per row
        for i, row in split.iterrows():
            # ignore dataset col
            max_val = row[2:].max()
            for col in split.columns:
                # format as percentage
                if col != "dataset":
                    split.loc[i, col] = f"{float(row[col]):.2f}"
                if row[col] == max_val:
                    split.loc[i, col] = "\\textbf{" + split.loc[i, col] + "}"
        
        # prepend "\rowcolor{Gainsboro!60}" if the acc is below 60
        for i, row in split.iterrows():
            if float(row["acc"]) <= 0.6:
                split.loc[i, "dataset"] = "\\rowcolor{Gainsboro!60}" + split.loc[i, "dataset"]

        with open(f"{directory}/{model.replace('/', '_')}__{metric}.txt", "w") as f:
            f.write(split.to_latex(index=False))
            print("wrote", model, metric)


def average_per_method(directory: str, reload: bool=False, metric: str="odds"):
    # collect all data
    df = load_directory(directory, reload, filter_step=True)

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


def plot_per_layer(directory: str, reload: bool=False, metric: str="odds"):
    # collect all data
    df = load_directory(directory, reload)

    # get average iia over layers, max'd
    df = df[["dataset", "model", "method", "layer", "pos", metric]]
    df.drop(columns=["pos"], inplace=True)
    df = df.groupby(["dataset", "model", "method", "layer"]).max().reset_index()
    df.drop(columns=["dataset"], inplace=True)
    df = df.groupby(["model", "method", "layer"]).mean().reset_index()

    # remove nans
    df = df.dropna()
    print(df)

    # pick overall better from probe_0 and probe_1
    per_method_avg = df.groupby(["model", "method"]).mean().reset_index()
    for model in per_method_avg["model"].unique():
        probe_0 = per_method_avg[(per_method_avg["model"] == model) & (per_method_avg["method"] == "probe_0")][metric].values[0]
        probe_1 = per_method_avg[(per_method_avg["model"] == model) & (per_method_avg["method"] == "probe_1")][metric].values[0]
        if probe_0 > probe_1:
            df = df[df["method"] != "probe_1"]
            df["method"] = df["method"].apply(lambda x: "probe" if x == "probe_0" else x)
        else:
            df = df[df["method"] != "probe_0"]
            df["method"] = df["method"].apply(lambda x: "probe" if x == "probe_1" else x)

    plot = (
        ggplot(df, aes(x="layer", y=metric, group="method", color="method"))
        + geom_line() + facet_wrap("~model", scales="free_x", nrow=1)
    )
    plot.save(f"{directory}/figs_{metric}_per_layer.pdf", width=10, height=3)


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


def plot_accuracy_vs_metric(directory: str, reload: bool=False, metric: str="odds"):
    # load
    df = load_directory(directory, reload)
    df = df[["dataset", "model", "method", "layer", "pos", "accuracy", metric]]
    df = df.groupby(["dataset", "model", "method", "layer", "pos"]).mean().reset_index()

    # pick overall better from probe_0 and probe_1
    df = pick_better_probe(df, [metric, "accuracy"])
    df = df[df["method"].isin(["probe", "lda"])]
    df.dropna(inplace=True)

    # round accuracy to 1 decimal
    # df["accuracy"] = df["accuracy"].apply(lambda x: round(x, 1))
    # df["accuracy"] = pd.Categorical(df["accuracy"], categories=df["accuracy"].unique(), ordered=True)

    plot = (
        ggplot(df, aes(x="accuracy", y=metric))
        + geom_point(alpha=0.05, size=0.5, color="None", fill="black")
        # + geom_boxplot()
        + facet_grid("method~model")
    )
    plot.save(f"{directory}/figs_accuracy_vs_{metric}.png", width=8, height=2.5, dpi=300)


def plot_pos_vs_trainstep(directory: str, reload: bool=False, metric: str="odds"):
    # load
    df = load_directory(directory, reload)

    # pick overall better from probe_0 and probe_1
    df = pick_better_probe(df, [metric, "accuracy"])
    df = df[df["method"].isin(["das"])]

    # drop layer
    df = df[["dataset", "model", "trainstep", "method", "layer", "pos", metric]]
    df = df.groupby(["dataset", "model", "trainstep", "method", "layer", "pos"]).mean().reset_index()
    df.dropna(inplace=True, ignore_index=True)

    # pos names
    for dataset in df["dataset"].unique():

        # select pos
        if dataset.startswith("npi_any_subj-relc"):
            df = df[((df["dataset"] == dataset) & (df["pos"].isin([1, 2, 3, 7, 8]))) | (df["dataset"] != dataset)]
        
        # add diff
        if "_inverted" in dataset:
            diff_name = dataset.replace("_inverted", "_diff")
            # find non-inverted and subtract to find diff
            non_inverted = dataset.replace("_inverted", "")
            das_df = df[df["method"] == "das"]
            if len(das_df) == 0:
                continue
            inverted_df = das_df[das_df["dataset"] == dataset].drop(columns=["dataset"])
            non_inverted_df = das_df[das_df["dataset"] == non_inverted].drop(columns=["dataset"])
            merged = inverted_df.merge(non_inverted_df, on=["model", "trainstep", "method", "layer", "pos"], suffixes=("_inverted", "_orig"))
            merged.dropna(inplace=True)
            merged[metric] = merged[f"{metric}_orig"] - merged[f"{metric}_inverted"]
            merged["dataset"] = diff_name
            print(merged)
            df = pd.concat([df, merged[["dataset", "model", "trainstep", "method", "layer", "pos", metric]]], ignore_index=True)
    
    # pos names
    for dataset in df["dataset"].unique():
        if "_diff" in dataset or "_inverted" in dataset:
            continue
        dataset_src = Dataset.load_from(f"syntaxgym/{dataset}")
        df.loc[df["dataset"].str.startswith(dataset), "pos_name"] = df.loc[df["dataset"].str.startswith(dataset), "pos"].apply(lambda x: dataset_src.span_names[x])
    
    # change trainstep 0 to 0.5
    df["trainstep"] = df["trainstep"].apply(lambda x: 0.5 if x == 0 else x)
    df["dataset"] = df["dataset"].apply(lambda x: x.replace("_inverted", "\ninverted"))
    df["dataset"] = df["dataset"].apply(lambda x: x.replace("_diff", "\ndiff"))
    df["dataset"] = df["dataset"].apply(lambda x: x.replace("npi_any_subj-relc\n", ""))
    df["dataset"] = df["dataset"].apply(lambda x: x.replace("npi_any_subj-relc", "orig"))

    # plot over trainsteps
    plot = (
        ggplot(df, aes(x="layer", y=metric, group="pos_name", color="pos_name"))
        + geom_line()
        + facet_grid("dataset~trainstep")
        # + scale_x_log10()
    )
    plot.save(f"{directory}/figs_{metric}_vs_trainstep.pdf", width=10, height=3)


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
    elif args.plot == "pos_all":
        plot_per_pos(args.file, args.reload, args.metric, plot_all=True)
    elif args.plot == "probe_hyperparam":
        probe_hyperparam_plot(args.file, args.reload, args.metric)
    elif args.plot == "layer":
        plot_per_layer(args.file, args.reload, args.metric)
    elif args.plot == "accuracy_vs_metric":
        plot_accuracy_vs_metric(args.file, args.reload, args.metric)
    elif args.plot == "pos_vs_trainstep":
        plot_pos_vs_trainstep(args.file, args.reload, args.metric)