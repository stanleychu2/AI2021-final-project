import re 
import sys
import json
from tqdm.auto import tqdm
from collections import defaultdict
from datasets import load_dataset

wiki_idx = json.load(open("wiki.json", "r", encoding='UTF-8'))
inputs = open(sys.argv[1], "r", encoding='UTF-8').readlines()
output = open(sys.argv[2], "w", encoding='UTF-8')
length = []

for line in tqdm(inputs):
    text = json.loads(line)
    if text['label']=="NOT ENOUGH INFO":
        continue
    text['evidence'] = [ i for e in text['evidence'] for i in e]
    evidences = defaultdict(list)
    for e in text['evidence']:
        if e[3] not in evidences[e[2]]:
            evidences[e[2]].append(e[3])
    knowledge = " ".join([ wiki_idx[k][e] for (k, v) in evidences.items() for e in sorted(v) if k in wiki_idx])
    
    sample = {"id": text['id'], "sentence1": text['claim'], "sentence2": knowledge, "label": (1 if text['label']=="SUPPORTS" else 0)}
    length.append(len((text['claim']+knowledge).split()))
    json.dump(sample, output, ensure_ascii=False)
    output.write("\n")


