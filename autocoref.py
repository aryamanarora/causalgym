from fastcoref import spacy_component
import spacy
import json
import glob
import os
from tqdm import tqdm
import torch
import pandas as pd
from plotnine import ggplot, aes, facet_wrap, facet_grid, geom_bar, theme, element_text, geom_errorbar, ggtitle, geom_hline, geom_point
from plotnine.scales import scale_color_manual, scale_x_log10, ylim
import argparse
import scipy.stats as stats

names = ["Participant 1", "Participant 2"]

def binomial_confidence_interval(count, total, confidence=0.95):
    """Calculate a binomial confidence using Wilson score."""

    # calculate confidence interval
    p = count / total
    n = total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    lower = (2 * n * p + z**2 - z * ((4 * n * p * (1 - p) + z**2)**(1/2))) / (2 * (n + z**2))
    upper = (2 * n * p + z**2 + z * ((4 * n * p * (1 - p) + z**2)**(1/2))) / (2 * (n + z**2))

    return lower, upper

def autocoref(folder="logs/new"):
    """Run autocoref on all logs."""

    # make logs/overall directory
    if not os.path.exists(f'{folder}/overall'):
        os.makedirs(f'{folder}/overall')

    # load coref model
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(
        "fastcoref", 
        config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'}
    )

    final = {}

    # for each file
    for file in glob.glob(f"{folder}/*.json"):
        with open(file, 'r') as f:
            print(file)

            # load data
            data = json.load(f)
            metadata = data['metadata']
            data = data['data']
            res = {}
            sents = [[sent for sent in data[key]['sentences']] for key in data]
            sents = [sent for sublist in sents for sent in sublist]

            # batch process
            docs = nlp.pipe(
                sents,
                component_cfg={"fastcoref": {'resolve_text': True}}
            )

            # run coref
            for key in tqdm(data):
                res[key] = {}
                res[key]['results'] = []
                res[key]['counts'] = data[key]['counts']
                res[key]['counts_resolved'] = {option: 0 for option in data[key]['counts']}
                res[key]['counts_resolved_pronoun'] = {option: 0 for option in data[key]['counts']}
                
                num_sents = len(data[key]['sentences'])

                # get metrics
                for i in range(num_sents):
                    doc = next(docs)
                    resolved = doc._.resolved_text
                    res[key]['results'].append({
                        "text": data[key]['sentences'][i],
                        "resolved": resolved,
                    })
                    for option in data[key]['counts']:
                        res[key]['counts_resolved'][option] += (1 if option in '.'.join(resolved.split('.')[1:]) else 0)
                        res[key]['counts_resolved_pronoun'][option] += (1 if resolved.split('. ')[1].startswith(option) else 0)
            
            # save data
            final[metadata['model']] = res

    # dump final
    with open(f'{folder}/overall/overall.json', 'w') as f:
        json.dump(final, f, indent=4)

def plot_individual(folder="logs/new"):
    """Plot the results of autocoref."""

    with open(f'{folder}/overall/overall.json', 'r') as f:
        data = json.load(f)

    # make order
    order = [(0, 'human')]
    
    # prepare pandas
    rows = []
    for key in data:
        model = key.split('logs/')[-1].split('.json')[0] if key.endswith('.json') else key

        # get param nums
        with open(f'{folder}/{model.replace("/", "-")}.json', 'r') as f:
            params = json.load(f)["metadata"]["num_parameters"]
            order.append((params, model))

        for sent in data[key]:
            for metric in data[key][sent]:
                if metric == 'results': continue
                for option in data[key][sent][metric]:
                    i = 0 if sent.startswith(option) else 1

                    # calculate confidence interval for probs
                    count = data[key][sent][metric][option]
                    lower, upper = binomial_confidence_interval(count, len(data[key][sent]['results'])) if metric in ["counts", "counts_resolved_pronoun"] else [None, None]

                    # add model data
                    rows.append({
                        "model": model,
                        "type": model.split('-')[0],
                        "is_human": False,
                        "sent": sent,
                        "metric": metric,
                        "option": names[i],
                        "count": count,
                        "prob": count / len(data[key][sent]['results']),
                        "lower": lower,
                        "upper": upper,
                        "params": params
                    })
    
    # read stimuli data
    with open('stimuli.json', 'r') as f:
        stimuli = json.load(f)

    # add human data
    for stimulus in stimuli:
        for option in stimulus['human']:
            i = 0 if stimulus['text'].startswith(option) else 1
            rows.append({
                "model": "human",
                "type": "human",
                "is_human": True,
                "sent": stimulus['text'],
                "metric": "counts_resolved_pronoun",
                "option": names[i],
                "count": None,
                "prob": stimulus['human'][option],
                "lower": None,
                "upper": None,
                "params": None
            })
    
    # df, set model order
    order = sorted(order, key=lambda x: x[0])
    order = [x[1] for x in order]
    df = pd.DataFrame(rows)
    df['model'] = pd.Categorical(df['model'].astype(str))
    df['model'].cat.set_categories(order, inplace=True)
    df['sent'] = df['sent'].map(lambda x: '\n'.join([x[i:i+20] for i in range(0, len(x), 20)]))
    df['sent'] = pd.Categorical(df['sent'].astype(str))
    df['sent'].cat.set_categories(['\n'.join([x['text'][i:i+20] for i in range(0, len(x['text']), 20)]) for x in sorted(stimuli, key=lambda x: list(x['human'].values())[0])], inplace=True)

    # plot
    plot = (ggplot(df[df['model'] != 'human'], aes(x="model", y="count", fill="option"))
            + geom_bar(stat="identity") + facet_grid("metric~sent", scales='free_y')
            + theme(figure_size=(15, 6), axis_text_x=element_text(rotation=45, hjust=1)))
    plot.save(f"{folder}/overall/plot.pdf")

    # plot probs for counts_resolved_pronoun with error bars
    df_pron = df[df['metric'] == 'counts_resolved_pronoun']

    # plot
    plot = (ggplot(df_pron[df_pron['model'] != 'human'], aes(x="model", y="prob", fill="option"))
            + scale_color_manual(values=["#0000FF00", "black"])
            + geom_bar(stat="identity")
            + geom_errorbar(aes(ymin="lower", ymax="upper"), width=0.2, color="black")
            + facet_grid("option~sent", scales='free_y')
            + theme(figure_size=(25, 6), axis_text_x=element_text(rotation=45, hjust=1))
            + ggtitle("What does '(S)he' resolve to?")
            + geom_hline(df_pron[df_pron['model'] == 'human'], aes(yintercept="prob"), linetype="dashed", show_legend=True)
            + ylim(0, 1))
    plot.save(f"{folder}/overall/plot_pron.pdf")

    # now do just counts
    df_counts = df[df['metric'] == 'counts']

    # plot
    plot = (ggplot(df_counts[df_counts['model'] != 'human'], aes(x="model", y="prob", fill="option"))
            + scale_color_manual(values=["#0000FF00", "black"])
            + geom_bar(stat="identity")
            + geom_errorbar(aes(ymin="lower", ymax="upper"), width=0.2, color="black")
            + facet_grid("option~sent", scales='free_y')
            + theme(figure_size=(25, 6), axis_text_x=element_text(rotation=45, hjust=1))
            + ggtitle("Is a participant mentioned by name?")
            + ylim(0, 1))
    plot.save(f"{folder}/overall/plot_counts.pdf")

def plot_aggregate():
    """Plot aggregate statistics across various settings."""

    # get all data
    data = {}
    for folder in glob.glob("logs/*"):
        overall = f"{folder}/overall/overall.json"
        if not os.path.exists(overall): continue
        with open(overall, 'r') as f:
            cur = json.load(f)
            data[folder[len("logs/"):]] = cur

    rows = []
    order = set()
    for setting in data:
        for model in data[setting]:

            # get param nums
            try:
                with open(f'logs/plain_sampling/{model.replace("/", "-")}.json', 'r') as f:
                    params = json.load(f)["metadata"]["num_parameters"]
                    order.add((params, model))
            except:
                pass

            # get data
            for sent in data[setting][model]:
                for generations in data[setting][model][sent]['results']:
                    tokens = generations['text'][len(sent):].split(" ")
                    resolution = None
                    if len(tokens) == 1:
                        token = ""
                    else:
                        token = tokens[1].strip()
                        # if token in ['has', 'was', 'had']:
                        #     token = ' '.join(tokens[1:3])
                        token = token.strip(",.:?!'\"").replace("\n", "\\n").replace(" ", "_")
                        for option in data[setting][model][sent]['counts']:
                            if generations['resolved'].split(". ")[1].startswith(option):
                                resolution = option
                                break
                    rows.append({
                        "setting": setting,
                        "model": model.split("/")[-1].split(".json")[0] if "logs/" in model else model.replace("/", "-"),
                        "sent": sent,
                        "token": token,
                        "resolution": resolution
                    })

    df = pd.DataFrame(rows)

    order = sorted(order, key=lambda x: x[0])
    order = [x[1].replace("/", "-") for x in order]
    df['model'] = pd.Categorical(df['model'].astype(str))
    df['model'].cat.set_categories(order, inplace=True)

    for setting in df['setting'].unique():
        for sent in df['sent'].unique():
            filtered = df[df['sent'] == sent]
            filtered = filtered[filtered['setting'] == setting]
            filtered = filtered.dropna()

            # count up by token column and set token order by count
            filtered = filtered.groupby(['token', 'model', 'resolution', 'sent', 'setting']).size().reset_index(name='count')

            # print(len(df))
            plot = (ggplot(filtered, aes(x='token', y='count', fill='resolution'))
                    + geom_bar(stat='identity')
                    + ggtitle(sent)
                    + theme(axis_text_x=element_text(rotation=90, hjust=0.5, size=2), figure_size=(10, 25))
                    + facet_wrap("model", ncol=1)
                    )
            plot.save(f"logs/{setting}/overall/{sent}.pdf")
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autocoref", action="store_true", help="run autocoref")
    parser.add_argument("--plot", action="store_true", help="plot autocoref results")
    parser.add_argument("--plot_agg", action="store_true", help="plot all autocoref results")
    parser.add_argument("--folder", default="logs/new", help="folder to use")
    args = parser.parse_args()

    if args.autocoref:
        autocoref(folder=args.folder)
    if args.plot:
        if args.folder == "all":
            for folder in glob.glob("logs/*"):
                plot_individual(folder=folder)
        else:
            plot_individual(folder=args.folder)
    if args.plot_agg:
        plot_aggregate()

if __name__ == "__main__":
    main()