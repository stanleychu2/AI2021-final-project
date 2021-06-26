import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

import json
import pandas as pd
from tqdm.auto import tqdm
from argparse import ArgumentParser

def parse_args():

    parser = ArgumentParser()

    parser.add_argument(
        "--test",
        type=str,
        help="Directory to the test data.",
        default="preprocess/test.json",
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        help="Directory to load the model file.",
        default="ckpt",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    test = open(args.test, "r", encoding='UTF-8').readlines()
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)

    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt_dir).to(device)
    model.eval()

    examples = []
    truths = []
    submit = {"id": [], "label": []}
    for line in tqdm(test):
        tempt = json.loads(line)
        truths.append(tempt['label'])
        inputs = tokenizer(tempt["sentence1"], tempt['sentence2'], padding='max_length', max_length=128,
                           truncation=True, return_tensors='pt').to(device)
        submit['id'].append(tempt['id'])
        with torch.no_grad():
            result = model(**inputs).logits.squeeze(0)
            submit['label'].append((1 if torch.softmax(result, dim=-1)[1].item() > .5 else 0))

    precision, recall, f1, _ = precision_recall_fscore_support(truths, submit['label'], zero_division=0, average="binary")
    print(f"accuracy: {sum([ i==k for (i,k) in zip(submit['label'], truths)])/len(truths)}, f1: {f1}, recall: {recall}, precision: {precision}")
    df = pd.DataFrame(submit)
    df.to_csv("submission.csv", index=False)

