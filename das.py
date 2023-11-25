import torch
import csv
import random
from utils import MODELS, WEIGHTS, Sentence, get_options, make_sentence
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
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
from models.utils import embed_to_distrib, top_vals, format_token, lsm, sm, count_parameters
from models.configuration_alignable_model import AlignableRepresentationConfig, AlignableConfig
from models.alignable_base import AlignableModel
from models.interventions import VanillaIntervention, RotatedSpaceIntervention
from models.gpt_neox.modelings_alignable_gpt_neox import create_gpt_neox
from umap import UMAP

def rotated_space_intervention(num_dims):
    def func(args, proj_dim):
        intervention = RotatedSpaceIntervention(args)
        intervention.set_interchange_dim(num_dims)
        return intervention
    return func

def intervention_config(model_type, intervention_type, layer, num_dims):
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
        alignable_interventions_type=rotated_space_intervention(num_dims)
    )
    return alignable_config

# load model
model = "EleutherAI/pythia-70m"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.pad_token = tokenizer.eos_token
gpt = AutoModelForCausalLM.from_pretrained(
    model,
    revision="main",
    torch_dtype=WEIGHTS.get(model, torch.bfloat16) if device == "cuda:0" else torch.float32
).to(device)

# tokenize
base = tokenizer("<|endoftext|>John is a", return_tensors="pt")
sources = [tokenizer("<|endoftext|>Jane is a", return_tensors="pt")]
tokens = tokenizer.encode(" stupid") # token we want to maximize the probability of

# just check final output probs for base and source
with torch.no_grad():
    base_logits = gpt(**base).logits
    source_logits = gpt(**sources[0]).logits

    # tops
    top_vals(tokenizer, sm(base_logits)[0, -1], 10)
    print('---')
    top_vals(tokenizer, sm(source_logits)[0, -1], 10)
    print('---')

    # top diffs
    top_vals(tokenizer, (source_logits - base_logits)[0, -1], 10)
    print('---')
    top_vals(tokenizer, -(source_logits - base_logits)[0, -1], 10)

# intervene on each layer
# only intervening on layer 0, pos 1, dim 1
data = []
# for layer_i in tqdm(range(gpt.config.num_hidden_layers)):
for layer_i in [0]:

    # for pos_i in range(1, len(base.input_ids[0])):
    for pos_i in [1]:
        
        # how many dims to intervene on
        for num_dims in [1]:

            print(layer_i, pos_i, num_dims)
        
            # set up alignable model
            alignable_config = intervention_config(type(gpt), "block_output", layer_i, num_dims)
            alignable = AlignableModel(alignable_config, gpt)
            alignable.set_device(device)
            alignable.disable_model_gradients()

            # optimizer
            t_total = 300
            warm_up_steps = 0.1 * t_total
            optimizer_params = []
            for k, v in alignable.interventions.items():
                optimizer_params += [{'params': v[0].rotate_layer.parameters()}]
                # optimizer_params += [{'params': v[0].intervention_boundaries, 'lr': 1e-2}]
            print("model trainable parameters: ", count_parameters(alignable.model))
            print("intervention trainable parameters: ", alignable.count_parameters())

            optimizer = torch.optim.Adam(
                optimizer_params,
                lr=1e-3
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warm_up_steps,
                num_training_steps=t_total
            )

            epochs = 3
            gradient_accumulation_steps = 1
            total_step = 0
            target_total_step = t_total * epochs
            temperature_start = 50.0
            temperature_end = 0.1
            temperature_schedule = torch.linspace(
                temperature_start, temperature_end, target_total_step
            ).to(torch.bfloat16).to(device)
            alignable.set_temperature(temperature_schedule[total_step])

            def calculate_loss(logits, labels):
                shift_logits = logits[..., :, :].contiguous()
                shift_labels = labels[..., :, :].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits[0, -1].softmax(-1)
                shift_labels = torch.tensor(tokens[0])
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                
                return loss
            
            iterator = tqdm(range(t_total))
            for step in iterator:
                _, counterfactual_outputs = alignable(
                    base,
                    sources,
                    {"sources->base": ([[[pos_i]]], [[[pos_i]]])}
                )
                
                # loss and backprop
                loss = calculate_loss(
                    counterfactual_outputs.logits, source_logits
                )
                loss_str = round(loss.item(), 2)

                stats = {'loss': loss_str}
                distrib = sm(counterfactual_outputs.logits)[0, -1]
                for tok in tokens:
                    prob = distrib[tok].item()
                    stats[format_token(tokenizer, tok)] = round(prob, 4)
                iterator.set_postfix(stats)

                loss.backward()
                optimizer.step()
                scheduler.step()
                alignable.set_zero_grad()
                alignable.set_temperature(temperature_schedule[total_step])
                total_step += 1
            
            # check top vals after trained intervention
            top_vals(tokenizer, sm(counterfactual_outputs.logits)[0, -1], 10)

                # get stats
                    # data.append({
                    #     "layer": layer_i,
                    #     "token": format_token(tokenizer, tok),
                    #     "prob": prob,
                    #     "pos": pos_i
                    # })

# # make df
# df = pd.DataFrame(data)
# df['layer'] = df['layer'].astype('int')
# df['pos'] = df['pos'].astype('int')
# df['prob'] = df['prob'].astype('float')

# # plot
# plot = (ggplot(df, aes(x="layer", y="pos")) + scale_y_reverse() + facet_grid("~token")
#         + geom_tile(aes(fill="prob")) + scale_fill_cmap("Purples"))
# print(plot)