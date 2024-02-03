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


def load_directory(directory: str, reload: bool=False):
    if reload or not glob.glob(f"{directory}/combined.csv"):
        print(f"reloading {directory}")
        # load all files (in parallel for speedup)
        file_paths = glob.glob(f"{directory}/*.json")
        dfs = []
        with multiprocessing.Pool() as pool:
            for df in tqdm(pool.imap_unordered(load_file, file_paths), total=len(file_paths)):
                dfs.append(df)
        
        # discard intermediate steps for DAS
        df = pd.concat(dfs, ignore_index=True)
        last_step = df["step"].max()
        df = df[(df["step"] == last_step) | (df["step"] == -1)]

        # store combined df
        df["acc"] = df["base_p_src"] < df["base_p_base"]
        df["iia"] = df["p_src"] > df["p_base"]
        df["odds"] = df['base_p_base'] - df['base_p_src'] + df['p_src'] - df['p_base']
        df["odds"] = df["odds"].apply(lambda x: math.exp(x))
        df.to_csv(f"{directory}/combined.csv", index=False)

        return df
    else:
        print(f"using existing {directory}")
        return pd.read_csv(f"{directory}/combined.csv")


def plot_acc(directory: str, reload: bool=False):
    """Plot raw accuracy for each model at each task."""

    # compute acc
    df = load_directory(directory, reload)
    df = df[df["method"] == "vanilla"]
    df = df[["dataset", "model", "base_p_base", "base_p_src"]]
    df["acc"] = df["base_p_src"] < df["base_p_base"]
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
    if metric == "iia":
        df["iia"] = df["iia"].apply(lambda x: 100 * x)
    elif metric == "odds":
        df["odds"] = df["odds"].apply(lambda x: math.log(x))
    df = df[df["method"] == "das"]
    df = df[["dataset", "model", "method", "layer", "pos", metric]]
    df = df.groupby(["dataset", "model", "method", "layer", "pos"]).mean().reset_index()

    # order
    df["model"] = df["model"].apply(lambda x: x.split("/")[-1])
    model_order = [x.split("/")[-1] for x in list(parameters.keys())[::-1]]
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    print(len(df))

    # plot
    for dataset in df["dataset"].unique():
        plot = (
            ggplot(df[df["dataset"] == dataset], aes(x="pos", y="layer"))
            + geom_tile(aes(fill=metric))
            # + geom_text(aes(label=f"{metric}_formatted"), color="black", size=7) + ggtitle(metric)
            + facet_wrap("~model", scales="free_y", nrow=1)
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
        plot.save(f"{directory}/figs_{dataset.replace('/', '_')}_{metric}.pdf", width=20, height=5)


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


def summarise(directory: str, reload: bool=False, metric: str="odds"):
    # collect all data
    df = load_directory(directory, reload)
    if metric == "iia":
        df["iia"] = df["iia"].apply(lambda x: 100 * x)
    elif metric == "odds":
        df["odds"] = df["odds"].apply(lambda x: math.log(x))

    # get average iia over layers, max'd 
    df = df[["dataset", "model", "method", "layer", "pos", metric]]
    df = df.groupby(["dataset", "model", "method", "layer", "pos"]).mean().reset_index()
    df = df.groupby(["dataset", "model", "method", "layer"]).max().reset_index()
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
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # base accuracy of each model on each task
    if args.plot == "acc":
        plot_acc(args.file, reload=args.reload)
    elif args.plot == "odds_summary":
        summarise(args.file, args.reload, "odds")
    elif args.plot == "iia_summary":
        summarise(args.file, args.reload, "iia")
    elif args.plot == "odds_pos":
        plot_per_pos(args.file, args.reload, "odds")
    elif args.plot == "iia_pos":
        plot_per_pos(args.file, args.reload, "iia")