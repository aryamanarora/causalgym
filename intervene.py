import torch
import csv
from collections import namedtuple
import random
from utils import MODELS, WEIGHTS
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import pandas as pd
from tqdm import tqdm
import argparse
from plotnine import ggplot, geom_point, geom_label, geom_tile, aes, facet_wrap, theme, element_text, \
                     geom_bar, geom_hline, scale_y_log10
from plotnine.scales import scale_x_continuous

# add align-transformers to path
sys.path.append("../align-transformers/")
from models.utils import embed_to_distrib, top_vals, format_token, lsm, sm
from models.configuration_alignable_model import AlignableRepresentationConfig, AlignableConfig
from models.alignable_base import AlignableModel
from models.interventions import VanillaIntervention
from models.gpt_neox.modelings_alignable_gpt_neox import create_gpt_neox
from umap import UMAP

# sentence helpers
Sentence = namedtuple("Sentence", ["sentence", "verb", "name1", "name2", "connective"])
names = {
    "he": ["John", "Bill", "Joseph", "Patrick", "Ken", "Geoff", "Simon", "Richard", "David", "Michael"],
    "she": ["Amanda", "Britney", "Catherine", "Dorothy", "Elizabeth", "Fiona", "Gina", "Helen", "Irene", "Jane"]
}
flattened_names = [(name, gender) for gender in names for name in names[gender]]
connectives = ["because"]

with open('data/ferstl.csv', 'r') as f:
    reader = csv.reader(f)
    verbs = [tuple(x) for x in reader]

def make_sentence(tokenizer: AutoTokenizer, name1=None, verb=None, name2=None, connective=None):
    while name1 is None or name1 == name2:
        test = random.choice(flattened_names)
        if len(tokenizer(test[0])['input_ids']) == 1:
            name1 = test
    while name2 is None or name1 == name2:
        test = random.choice(flattened_names)
        if len(tokenizer(' ' + test[0])['input_ids']) == 1:
            name2 = test
    while verb is None:
        test = random.choice(verbs)
        if len(tokenizer(' ' + test[0])['input_ids']) == 1:
            verb = test
    while connective is None:
        connective = random.choice(connectives)
    sent = f"{name1[0]} {verb[0]} {name2[0]} {connective}"
    return Sentence(
        sentence=sent,
        verb=verb,
        name1=name1,
        name2=name2,
        connective=connective,
    )

def simple_position_config(model_type, intervention_type, layer):
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,             # layer
                intervention_type, # intervention type
                "pos",             # intervention unit
                1                  # max number of unit
            ),
        ],
        alignable_interventions_type=VanillaIntervention,
    )
    return alignable_config

def kldiv(input, target):
    return torch.nn.functional.kl_div(input, target, reduction="sum", log_target=True).cpu().detach().item()

@torch.inference_mode()
def plot_next_token_map(gpt, tokenizer, sentences, model, device='cpu'):
    # run model, get distribs
    print(sentences[0])
    prompts = [s.sentence for s in sentences]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    res = sm(gpt(**inputs).logits)
    mask = inputs['attention_mask']
    top_vals(tokenizer, res[0][mask[0] == 1][-1], 10)
    distribs = []
    for i in range(len(prompts)):
        distrib = res[i][mask[i] == 1]
        distribs.append(distrib[-1].cpu().detach().type(torch.float32).numpy())
    
    # umap distribs
    umap = UMAP(n_components=2, random_state=42)
    umap.fit(distribs)
    umap_distribs = umap.transform(distribs)

    # make plot
    df = pd.DataFrame(umap_distribs, columns=['x', 'y'])
    df['prompt'] = prompts
    df['verb'] = [s.verb[0] for s in sentences]
    df['type'] = [s.verb[1] for s in sentences]
    df['p(he)'] = [d[tokenizer(' he').input_ids[0]].item() for d in distribs]
    df['p(she)'] = [d[tokenizer(' she').input_ids[0]].item() for d in distribs]
    g = (ggplot(df) + geom_label(aes(x='p(he)', y='p(she)', label='verb', color='type'), alpha=0.5)
        + theme(axis_text_x=element_text(rotation=90), figure_size=(10, 10)))
    g.save(f"figs/{model.replace('/', '-')}-next-token-probs.pdf")

@torch.inference_mode()
def experiment(model="EleutherAI/pythia-70m", revision="main", intervene="verb"):
    """Run experiment."""

    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    gpt = AutoModelForCausalLM.from_pretrained(
        model,
        revision=revision,
        torch_dtype=WEIGHTS.get(model, torch.bfloat16) if device == "cuda:0" else torch.float32
    ).to(device)

    # layers
    nodes = []
    for l in range(gpt.config.num_hidden_layers - 1, -1, -1):
        nodes.append(f'f{l}')
        nodes.append(f'a{l}')

    # make stimuli
    base = make_sentence(tokenizer, name1=("Joseph", "he"), name2=("Elizabeth", "she"), connective="because")
    intervention = {
        "tokenizer": tokenizer,
        "name1": None if intervene == "name1" else base.name1,
        "verb": None if intervene == "verb" else base.verb,
        "name2": None if intervene == "name2" else base.name2,
        "connective": None if intervene == "connective" else base.connective,
    }
    others = {}
    for i in range(len(verbs)):
        intervention['verb'] = verbs[i]
        s = make_sentence(**intervention)
        others[s.verb[0]] = s
    others = list(others.values())
    print(len(others))
    plot_next_token_map(gpt, tokenizer, others, model, device)
    return
    sources = [make_sentence(**intervention)]
    print(base.sentence)
    print(sources[0].sentence)

    # tokenizer
    base = tokenizer(base.sentence, return_tensors="pt")
    base = {key: value.to(device) for key, value in base.items()}
    print(base)
    sources = [tokenizer(s.sentence, return_tensors="pt") for s in sources]
    sources = [{key: value.to(device) for key, value in x.items()} for x in sources]
    print(len(base['input_ids'][0]))
    print(len(sources[0]['input_ids'][0]))
    input()

    # get logits
    base_logits = lsm(gpt(**base).logits)
    sources_logits = [lsm(gpt(**x).logits) for x in sources]
    top_vals(tokenizer, base_logits[0, -1], 10)
    print('---')
    top_vals(tokenizer, sources_logits[0][0, -1], 10)

    # intervene on each layer
    data = []
    for layer_i in tqdm(range(gpt.config.num_hidden_layers)):
        alignable_config = simple_position_config(type(gpt), "mlp_output", layer_i)
        alignable = AlignableModel(alignable_config, gpt)
        for pos_i in range(len(base['input_ids'][0])):
            _, counterfactual_outputs = alignable(
                base,
                sources,
                {"sources->base": ([[[pos_i]]], [[[pos_i]]])}
            )
            logits = lsm(counterfactual_outputs.logits)
            data.append({
                'layer': f"f{layer_i}",
                'pos': pos_i,
                'type': "mlp_output",
                'kldiv_base': kldiv(logits[0, -1], base_logits[0, -1]),
                'kldiv_source': kldiv(logits[0, -1], sources_logits[0][0, -1]),
            })
                
        alignable_config = simple_position_config(type(gpt), "attention_input", layer_i)
        alignable = AlignableModel(alignable_config, gpt)
        for pos_i in range(len(base['input_ids'][0])):
            _, counterfactual_outputs = alignable(
                base,
                sources,
                {"sources->base": ([[[pos_i]]], [[[pos_i]]])}
            )
            logits = lsm(counterfactual_outputs.logits)
            data.append({
                'layer': f"a{layer_i}",
                'pos': pos_i,
                'type': "attention_input",
                'kldiv_base': kldiv(logits[0, -1], base_logits[0, -1]),
                'kldiv_source': kldiv(logits[0, -1], sources_logits[0][0, -1]),
            })

    # make df
    df = pd.DataFrame(data)
    df['layer'] = df['layer'].astype('category')
    df['layer'] = pd.Categorical(df['layer'], categories=nodes[::-1], ordered=True)

    # plot
    labels = [format_token(tokenizer, x) for x in base['input_ids'][0]]
    g = (ggplot(df) + geom_tile(aes(x='pos', y='layer', fill='kldiv_base', color='kldiv_base'))
        + theme(axis_text_x=element_text(rotation=90), figure_size=(10, 10))
        + scale_x_continuous(breaks=list(range(len(labels))), labels=labels))
    
    # save fig
    g.save(f"figs/{model}-{intervene}.pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m", help="name of model")
    parser.add_argument("--revision", default="main", help="revision of model")
    parser.add_argument("--intervene", default="verb", help="what part of the sentence to intervene on")
    args = parser.parse_args()
    print(vars(args))
    
    experiment(**vars(args))

if __name__ == "__main__":
    main()