import glob
import json
from tqdm import tqdm

data = {}

for file in tqdm(glob.glob("logs/*/overall/overall.json")):
    path = file.split("/")
    with open(file) as f:
        data[path[1]] = json.load(f)

with open("logs/overall.json", "w") as f:
    json.dump(data, f, indent=4)