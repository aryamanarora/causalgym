from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
from transformers import pipeline, set_seed
from fastcoref import FCoref
import torch
from collections import defaultdict
from plotnine import ggplot, aes, geom_point, geom_histogram, facet_wrap, facet_grid
from plotnine.scales import xlim
import pandas as pd

START = len('<|endoftext|>')

set_seed(42)

def get_bounds(text, needle):
    start = text.find(needle)
    end = start + len(needle)
    return (start, end)

# load gpt2
name, revision = "gpt2-medium", "main"
generator = pipeline('text-generation', model=name, revision=revision, device='cpu')
print("loaded model")

# load fastcoref
model = FCoref(device='cpu')
print("loaded coref model")

# stimuli
stimuli = [
    {"text": "John seized the comic from Bill. He", "options": ["John", "Bill"], "pronoun": "He"},
    {"text": "John passed the comic to Bill. He", "options": ["John", "Bill"], "pronoun": "He"},
]

data = []

# generate 100 continuations of agent_biased (stop at period)
with torch.inference_mode():
    for stimulus in stimuli:
        # get start and end positions of pronoun in text
        pronoun = get_bounds(stimulus['text'], stimulus['pronoun'])
        options = [get_bounds(stimulus['text'], option) for option in stimulus['options']]

        # make sents
        sents = generator('<|endoftext|>' + stimulus['text'], max_length=30, num_return_sequences=5, eos_token_id=13)
        sents = [sent['generated_text'][START:].rstrip('.') + '.' for sent in sents]

        # do coref
        pred = model.predict(sents)
        for sent in pred:

            # calculate probs
            logits = []
            for option in options:
                try:
                    logits.append(sent.get_logit(span_i=pronoun, span_j=option))
                except:
                    logits.append(-1000000)
            logits = torch.Tensor(logits)
            probs = torch.softmax(logits, dim=0)
            print(sent.text)
            print(sent.get_clusters())

            # data
            for i in range(len(options)):
                data.append({
                    'prompt': stimulus['text'],
                    'sent': sent.text,
                    'he': sent.text[pronoun[0]:pronoun[1]],
                    'mention': sent.text[options[i][0]:options[i][1]],
                    'prob': probs[i].item()
                })
                print(data[-1])
            print()
            # input()
        
# make plot
df = pd.DataFrame(data)
plot = ggplot(df, aes(x='prob')) + geom_histogram(bins=10) + facet_grid('prompt ~ mention') + xlim(0, 1)
print(plot)
        