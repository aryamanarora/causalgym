import glob
import json
from tqdm import tqdm
from collections import defaultdict

data = {}

for file in tqdm(glob.glob("logs/*/overall/overall.json")):
    path = file.split("/")
    with open(file) as f:
        data[path[1]] = json.load(f)

with open("logs/overall.json", "w") as f:
    json.dump(data, f, indent=4)

data = {
    "data": [],
}

models = defaultdict(lambda: len(models))

for file in tqdm(glob.glob("logs/kldiv/new/*.json")):
    path = file.split("/")
    with open(file) as f:
        data["data"].extend(json.load(f))

for i in tqdm(range(len(data["data"]))):
    data["data"][i]["model"] = models[data["data"][i]["model"]]

data["models"] = dict(models)

with open("logs/kldiv/all.json", "w") as f:
    json.dump(data, f, indent=4)