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
        alignable_interventions_type=VanillaIntervention,
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
    intervene: str="verb"
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

    # layers
    nodes = []
    for l in range(gpt.config.num_hidden_layers - 1, -1, -1):
        nodes.append(f'f{l}')
        nodes.append(f'a{l}')

    # make stimuli
    options = get_options(tokenizer, token_length=1)
    base = make_sentence(
        options,
        name1=("Joseph", "he"),
        name2=("Elizabeth", "she"),
        verb=("loved", "ExpStim"),
        connective="because"
    )
    intervention = {
        "options": options,
        "name1": None if intervene == "name1" else base.name1,
        "verb": None if intervene == "verb" else base.verb,
        "name2": None if intervene == "name2" else base.name2,
        "connective": None if intervene == "connective" else base.connective,
    }

    # intervention time
    sources_orig = []
    for _ in range(10):
        sources_orig.append(make_sentence(**intervention))
    print(base.sentence)
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
    base = tokenizer(base.sentence, return_tensors="pt")
    base = {key: value.to(device) for key, value in base.items()}
    sources = [tokenizer(s.sentence, return_tensors="pt") for s in sources_orig]
    sources = [{key: value.to(device) for key, value in x.items()} for x in sources]

    # get logits
    base_logits = lsm(gpt(**base).logits)
    sources_logits = [lsm(gpt(**x).logits) for x in sources]

    # intervene on each layer
    data = []
    for layer_i in tqdm(range(gpt.config.num_hidden_layers)):
        for intervention_type in ["mlp_output", "attention_input"]:
            alignable_config = simple_position_config(type(gpt), intervention_type, layer_i)
            alignable = AlignableModel(alignable_config, gpt)
            for pos_i in range(len(base['input_ids'][0])):
                for i, source in enumerate(sources):
                    _, counterfactual_outputs = alignable(
                        source,
                        [base],
                        {"sources->base": ([[[pos_i]]], [[[pos_i]]])}
                    )
                    logits = lsm(counterfactual_outputs.logits)
                    data.append({
                        'layer': f"f{layer_i}",
                        'pos': pos_i,
                        'type': intervention_type,
                        'verb': sources_orig[i].verb[0],
                        'verb_type': sources_orig[i].verb[1],
                        'p(he)': logits[0, -1, tokenizer(' he').input_ids[0]].exp().item(),
                        'p(she)': logits[0, -1, tokenizer(' she').input_ids[0]].exp().item(),
                        'kldiv_base': kldiv(logits[0, -1], base_logits[0, -1]),
                        'kldiv_source': kldiv(logits[0, -1], sources_logits[0][0, -1]),
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
    g = (ggplot(df) + geom_label(aes(x='p(he)', y='p(she)', label='verb', fill='layer'), alpha=0.3)
        + theme(axis_text_x=element_text(rotation=90), figure_size=(10, 10))
        + facet_wrap('pos'))
    g.save(f"figs/{model.replace('/', '-')}-intervene-{intervene}.pdf")

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