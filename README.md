# Artificial Intelligence 2021 Spring - Fact Verification

### Data
- data folder: original data from fever.ai
- preprocess folder: add knowledge from wiki.json
``` console
gdown --id "1OQ6oK08jqjFHnlrM24m3gXYB6VdWQD6C"
unzip wiki.zip
rm wiki.zip
python3 preprocess.py data/train.jsonl preprocess/train.json
```

### Train
- MODEL=xlnet-base-cased
- MODEL=distilgpt2
- MODEL=nielsr/coref-roberta-base
- weights save at `ckpt`
```console
bash run_fever.sh
```

### Predict
```console
python3 predict.py --test preprocess/valid.json --ckpt checkpoint
```
