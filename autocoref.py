from fastcoref import spacy_component
import spacy
import json
import glob
import os
from tqdm import tqdm
import torch
import pandas as pd
from plotnine import ggplot, aes, facet_wrap, facet_grid, geom_bar, theme, element_text, geom_errorbar, ggtitle
from plotnine.scales import scale_color_manual
import argparse
import scipy.stats as stats

def binomial_confidence_interval(count, total, confidence=0.95):
    """Calculate a binomial confidence using Wilson score."""

    # calculate confidence interval
    p = count / total
    n = total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    lower = (2 * n * p + z**2 - z * ((4 * n * p * (1 - p) + z**2)**(1/2))) / (2 * (n + z**2))
    upper = (2 * n * p + z**2 + z * ((4 * n * p * (1 - p) + z**2)**(1/2))) / (2 * (n + z**2))

    return lower, upper

def autocoref():
    # make logs/overall directory
    if not os.path.exists('logs/overall'):
        os.makedirs('logs/overall')

    # load coref model
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(
    "fastcoref", 
    config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'}
    )

    final = {}

    # for each file
    for file in tqdm(glob.glob("logs/*.json")):
        with open(file, 'r') as f:

            # load data
            data = json.load(f)
            res = {}

            # run coref
            for key in data:
                if key == 'metadata': continue
                res[key] = {}
                res[key]['counts'] = data[key]['counts']
                res[key]['counts_resolved'] = {option: 0 for option in data[key]['counts']}
                res[key]['counts_resolved_pronoun'] = {option: 0 for option in data[key]['counts']}

                for sent in data[key]['sentences']:
                    # resolve coref
                    doc = nlp(
                        sent,
                        component_cfg={"fastcoref": {'resolve_text': True}}
                    )
                    resolved = doc._.resolved_text
                    for option in data[key]['counts']:
                        res[key]['counts_resolved'][option] += (1 if option in '.'.join(resolved.split('.')[1:]) else 0)
                        res[key]['counts_resolved_pronoun'][option] += (1 if resolved.split('. ')[1].startswith(option) else 0)
            
            # save data
            final[file] = res

    # dump final
    with open('logs/overall/overall.json', 'w') as f:
        json.dump(final, f, indent=4)

def plot():
    """Plot the results of autocoref"""

    with open('logs/overall/overall.json', 'r') as f:
        data = json.load(f)

    # make order
    order = [(0, 'human')]
    
    # prepare pandas
    rows = []
    for key in data:
        model = key[len('logs/'):-len('.json')]

        # get param nums
        with open(key, 'r') as f:
            params = json.load(f)["metadata"]["num_parameters"]
            order.append((params, model))

        for sent in data[key]:
            for metric in data[key][sent]:
                for option in data[key][sent][metric]:
                    count = data[key][sent][metric][option]
                    lower, upper = binomial_confidence_interval(count, 100) if metric in ["counts", "counts_resolved_pronoun"] else [None, None]
                    rows.append({
                        "model": model,
                        "is_human": False,
                        "sent": sent,
                        "metric": metric,
                        "option": option,
                        "count": count,
                        "lower": lower,
                        "upper": upper
                    })
    
    # read stimuli data
    with open('stimuli.json', 'r') as f:
        stimuli = json.load(f)

    for stimulus in stimuli:
        for option in stimulus['human']:
            rows.append({
                "model": "human",
                "is_human": True,
                "sent": stimulus['text'],
                "metric": "counts_resolved_pronoun",
                "option": option,
                "count": stimulus['human'][option] * 100,
                "lower": None,
                "upper": None
            })
    
    # df, set model order
    order = sorted(order, key=lambda x: x[0])
    order = [x[1] for x in order]
    df = pd.DataFrame(rows)
    df['model'] = pd.Categorical(df['model'].astype(str))
    df['model'].cat.set_categories(order, inplace=True)

    # plot
    plot = (ggplot(df[df['model'] != 'human'], aes(x="model", y="count", fill="option"))
            + geom_bar(stat="identity") + facet_grid("metric~sent", scales='free_y')
            + theme(figure_size=(15, 6), axis_text_x=element_text(rotation=45, hjust=1)))
    plot.save("logs/overall/plot.pdf")

    # plot probs for counts_resolved_pronoun with error bars
    df_pron = df[df['metric'] == 'counts_resolved_pronoun']
    df_pron["prob"] = df_pron["count"].map(lambda x: x / 100)

    # plot
    plot = (ggplot(df_pron, aes(x="model", y="prob", fill="option", color="is_human"))
            + scale_color_manual(values=["#0000FF00", "black"])
            + geom_bar(stat="identity")
            + geom_errorbar(aes(ymin="lower", ymax="upper"), width=0.2, color="black")
            + facet_grid("option~sent", scales='free_y')
            + theme(figure_size=(15, 6), axis_text_x=element_text(rotation=45, hjust=1))
            + ggtitle("What does 'He' resolve to?"))
    plot.save("logs/overall/plot_pron.pdf")

    # now do just counts
    df_counts = df[df['metric'] == 'counts']
    df_counts["prob"] = df_counts["count"].map(lambda x: x / 100)

    # plot
    plot = (ggplot(df_counts, aes(x="model", y="prob", fill="option", color="is_human"))
            + scale_color_manual(values=["#0000FF00", "black"])
            + geom_bar(stat="identity")
            + geom_errorbar(aes(ymin="lower", ymax="upper"), width=0.2, color="black")
            + facet_grid("option~sent", scales='free_y')
            + theme(figure_size=(15, 6), axis_text_x=element_text(rotation=45, hjust=1))
            + ggtitle("Is a participant mentioned by name?"))
    plot.save("logs/overall/plot_counts.pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autocoref", action="store_true", help="run autocoref")
    parser.add_argument("--plot", action="store_true", help="plot autocoref results")
    args = parser.parse_args()

    if args.autocoref:
        autocoref()
    if args.plot:
        plot()

if __name__ == "__main__":
    main()