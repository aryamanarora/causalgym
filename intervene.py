# from transformers import pipeline, set_seed
# import torch
# import sys

# sys.path.append("../align-transformers")
# from models.utils import print_forward_hooks
# from models.configuration_alignable_model import AlignableRepresentationConfig, AlignableConfig
# from models.alignable_base import AlignableModel
# from models.interventions import VanillaIntervention
# from models.gpt2.modelings_alignable_gpt2 import create_gpt2

# set_seed(42)

# # load gpt2
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# config, tokenizer, gpt = create_gpt2(cache_dir="/Users/aryamanarora/.cache/huggingface/hub")
# prompt = "John seized the comic from Bill. He"
# input_ids = tokenizer.encode(prompt, return_tensors="pt")
# output = gpt.generate(input_ids, max_length = 50, num_beams=1)
# output_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(output_text)

# # intervention config
# def simple_position_config(model_type, intervention_type, layer):
#     alignable_config = AlignableConfig(
#         alignable_model_type=model_type,
#         alignable_representations=[
#             AlignableRepresentationConfig(
#                 layer,             # layer
#                 intervention_type, # intervention type
#                 "pos",             # intervention unit
#                 1                  # max number of unit
#             ),
#         ],
#         alignable_interventions_type=VanillaIntervention,
#     )
#     return alignable_config

# # intervention
# alignable_config = simple_position_config(type(gpt), "mlp_output", 0)
# alignable = AlignableModel(alignable_config, gpt)

# # setup
# prompt = "John seized the comic from Bill. He"
# base = tokenizer(prompt, return_tensors="pt")
# prompt2 = "Tom seized the comic from Bill. He"
# source = tokenizer(prompt2, return_tensors="pt")

# # generate
# base_outputs, counterfactual_outputs = alignable.generate(
#     base,
#     [source],
#     {"sources->base": ([[[0]]], [[[0]]])},
#     max_length = 50, num_beams=1
# )
# print(base_outputs)
# print(counterfactual_outputs)
# print(tokenizer.decode(base_outputs[0]))
# print(tokenizer.decode(counterfactual_outputs[0]))