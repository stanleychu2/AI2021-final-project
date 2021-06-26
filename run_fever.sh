#MODEL='xlnet-base-cased'
#MODEL='distilgpt2'
MODEL='nielsr/coref-roberta-base'
EPOCH="5"
RUN_NAME='roberta'

python3 classification.py \
    --output_dir "./ckpt/${RUN_NAME}" \
    --overwrite_output_dir \
    --model_name_or_path "${MODEL}" \
    --metric_for_best_model "f1" \
    --train_file "preprocess/train.json" \
    --validation_file "preprocess/valid.json" \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --learning_rate 5e-5 \
    --num_train_epochs "${EPOCH}" \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --fp16 True \
    --seed 29 \
    --logging_steps 100 \
    --load_best_model_at_end \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --dataloader_num_workers 8


