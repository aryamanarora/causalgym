import torch
import csv
import random
from utils import MODELS, WEIGHTS, Sentence, get_options, make_sentence
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import pandas as pd
from tqdm import tqdm
import argparse
from plotnine import ggplot, geom_point, geom_label, geom_tile, aes, facet_wrap, theme, element_text, \
                     geom_bar, geom_hline, scale_y_log10, ggtitle
from plotnine.scales import scale_x_continuous
from typing import List

# add align-transformers to path
sys.path.append("../align-transformers/")
from models.utils import embed_to_distrib, top_vals, format_token, lsm, sm
from models.configuration_alignable_model import AlignableRepresentationConfig, AlignableConfig
from models.alignable_base import AlignableModel
from models.interventions import VanillaIntervention
from models.gpt_neox.modelings_alignable_gpt_neox import create_gpt_neox
from umap import UMAP

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
        alignable_interventions_type=VanillaIntervention
    )
    return alignable_config

def kldiv(input, target):
    return torch.nn.functional.kl_div(input, target, reduction="sum", log_target=True).cpu().detach().item()

@torch.inference_mode()
def plot_next_token_map(
    gpt: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentences: List[Sentence],
    model: str,
    device: str='cpu'
):
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
    return df

@torch.inference_mode()
def experiment(
    model: str="EleutherAI/pythia-70m",
    revision: str="main",
    intervene: str="verb",
    pos: int=0
):
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

    # make stimuli
    options = get_options(tokenizer, token_length=1)
    base_orig = make_sentence(
        options,
        name1=("Joseph", "he"),
        name2=("Elizabeth", "she"),
        verb=("hated", "ExpStim"),
        connective="because"
    )
    intervention = {
        "options": options,
        "name1": None if intervene == "name1" else base_orig.name1,
        "verb": None if intervene == "verb" else base_orig.verb,
        "name2": None if intervene == "name2" else base_orig.name2,
        "connective": None if intervene == "connective" else base_orig.connective,
    }

    # intervention time
    sources_orig = []
    for _ in range(100):
        sources_orig.append(make_sentence(**intervention))
    print(base_orig.sentence)
    for s in sources_orig:
        print(s.sentence)

    # make others w intervention
    others = {}
    for i in range(len(options['verbs'])):
        intervention['verb'] = options['verbs'][i]
        s = make_sentence(**intervention)
        others[s.verb[0]] = s
    others = list(others.values())
    print(len(others))

    # plot
    plot_next_token_map(gpt, tokenizer, others, model, device)

    # tokenizer
    base = tokenizer(base_orig.sentence, return_tensors="pt")
    base = {key: value.to(device) for key, value in base.items()}
    sources = [tokenizer(s.sentence, return_tensors="pt") for s in sources_orig]
    sources = [{key: value.to(device) for key, value in x.items()} for x in sources]

    # layers
    nodes = ["none"]
    for p in range(len(base['input_ids'][0])):
        for l in range(gpt.config.num_hidden_layers - 1, -1, -1):
            nodes.append(f'{l}.{p}')
            # nodes.append(f'f{l}.{pos}')
            # nodes.append(f'a{l}.{pos}')

    # get logits
    data = []
    base_logits = sm(gpt(**base).logits)
    sources_logits = [sm(gpt(**x).logits) for x in sources]
    for i in range(len(sources) + 1):
        logits = sources_logits[i - 1] if i > 0 else base_logits
        data.append({
            'layer': f"none",
            'verb': sources_orig[i - 1].verb[0] if i > 0 else base_orig.verb[0],
            'verb_type': sources_orig[i - 1].verb[1] if i > 0 else base_orig.verb[1],
            'is_base': i == 0,
            'p(he)': logits[0, -1, tokenizer(' he').input_ids[0]].item(),
            'p(she)': logits[0, -1, tokenizer(' she').input_ids[0]].item(),
        })

    # intervene on each layer
    pos_i = pos
    for layer_i in tqdm(range(gpt.config.num_hidden_layers)):
        alignable_config = simple_position_config(type(gpt), "block_output", layer_i)
        alignable = AlignableModel(alignable_config, gpt)
        # for pos_i in range(len(base['input_ids'][0])):
        for i, source in enumerate(sources):
            _, counterfactual_outputs = alignable(
                source,
                [base],
                {"sources->base": ([[[pos_i]]], [[[pos_i]]])}
            )
            logits = sm(counterfactual_outputs.logits)
            data.append({
                'layer': f"{layer_i}.{pos_i}",
                'verb': sources_orig[i].verb[0],
                'verb_type': sources_orig[i].verb[1],
                'is_base': False,
                'p(he)': logits[0, -1, tokenizer(' he').input_ids[0]].item(),
                'p(she)': logits[0, -1, tokenizer(' she').input_ids[0]].item(),
                # 'kldiv_base': kldiv(logits[0, -1].log(), base_logits[0, -1].log()),
            })

    # make df
    df = pd.DataFrame(data)
    df['layer'] = df['layer'].astype('category')
    df['layer'] = pd.Categorical(df['layer'], categories=nodes[::-1], ordered=True)

    # plot
    # labels = [format_token(tokenizer, x) for x in base['input_ids'][0]]
    # g = (ggplot(df) + geom_tile(aes(x='pos', y='layer', fill='kldiv_base', color='kldiv_base'))
    #     + theme(axis_text_x=element_text(rotation=90), figure_size=(10, 10))
    #     + scale_x_continuous(breaks=list(range(len(labels))), labels=labels))
    g = (ggplot(df) + geom_label(aes(x='p(he)', y='p(she)', label='verb', fill='verb_type', boxstyle='is_base'), alpha=0.5)
        + theme(axis_text_x=element_text(rotation=90), figure_size=(20, 20))
        + facet_wrap('layer'))
    g.save(f"figs/{model.replace('/', '-')}-intervene-{intervene}-{pos_i}.pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m", help="name of model")
    parser.add_argument("--revision", default="main", help="revision of model")
    parser.add_argument("--intervene", default="verb", help="what part of the sentence to intervene on")
    parser.add_argument("--pos", default=0, help="intervention position", type=int)
    args = parser.parse_args()
    print(vars(args))
    
    experiment(**vars(args))

if __name__ == "__main__":
    main()