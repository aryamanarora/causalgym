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
                     facet_grid, geom_bar, geom_hline, scale_y_log10, ggtitle
from plotnine.scales import scale_x_continuous, scale_fill_cmap, scale_y_reverse
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
    num_samples: int=20
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

    # tokens to check
    tokens = tokenizer.encode(" he she", return_tensors="pt")[0].to(device)
    data = []

    # make data
    samples = []
    for i in range(num_samples):
        # make base and source
        base_orig = make_sentence(options, connective="because")
        intervention = {
            'name1': base_orig.name1 if intervene != 'name1' else None,
            'verb': base_orig.verb if intervene != 'verb' else None,
            'name2': base_orig.name2 if intervene != 'name2' else None,
            'connective': base_orig.connective if intervene != 'connective' else None,
        }
        intervention['options'] = options
        source_orig = make_sentence(**intervention)
        intervened = f"{getattr(base_orig, intervene)}/{getattr(source_orig, intervene)}"
        base = tokenizer(base_orig.sentence, return_tensors="pt").to(device)
        sources = [tokenizer(source_orig.sentence, return_tensors="pt").to(device)]
        base_logits = sm(gpt(**base).logits)
        print(base_orig.sentence, source_orig.sentence)
        samples.append((base_orig, source_orig, base, sources, base_logits))

    # intervene on each layer
    for layer_i in tqdm(range(gpt.config.num_hidden_layers)):

        # make config
        alignable_config = simple_position_config(type(gpt), "block_output", layer_i)
        alignable = AlignableModel(alignable_config, gpt)

        for i in range(num_samples):
            base_orig, source_orig, base, sources, base_logits = samples[i]

            # intervention time
            for pos_i in range(len(base['input_ids'][0])):
                _, counterfactual_outputs = alignable(
                    base,
                    sources,
                    {"sources->base": ([[[pos_i]]], [[[pos_i]]])}
                )
                logits = sm(counterfactual_outputs.logits)
                data.extend([{
                    'layer': layer_i,
                    'pos': pos_i,
                    'base': base_orig.sentence,
                    'source': source_orig.sentence,
                    'genders': f"{base_orig.name1[1]}/{source_orig.name1[1]}, {base_orig.name2[1]}/{source_orig.name2[1]}",
                    'intervened': intervened,
                    'is_base': False,
                    'token': format_token(tokenizer, token),
                    'prob': logits[0, -1, token].item() - base_logits[0, -1, token].item(),
                } for token in tokens])

    # make df
    df = pd.DataFrame(data)
    df['layer'] = df['layer'].astype('int')
    df['pos'] = df['pos'].astype('int')
    df['prob'] = df['prob'].astype('float')

    # plot
    plot = (ggplot(df, aes(x="layer", y="pos")) + scale_y_reverse() + facet_grid("genders ~ token")
            + geom_tile(aes(fill="prob")) + scale_fill_cmap("PuOr") + theme(figure_size=(10, 20)))
    plot.save(f"figs/{model.replace('/', '-')}-intervene-{intervene}.pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m", help="name of model")
    parser.add_argument("--revision", default="main", help="revision of model")
    parser.add_argument("--intervene", default="verb", help="what part of the sentence to intervene on")
    parser.add_argument("--num_samples", default=20, type=int, help="number of samples to run")
    args = parser.parse_args()
    print(vars(args))
    
    experiment(**vars(args))

if __name__ == "__main__":
    main()