python3 run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --overwrite_cache \
    --do_lower_case \
    --version_2_with_negative \
    --save_steps 100000 \
    --train_file $SQUAD_DIR/train-v2.0.json \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --per_gpu_train_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./tmp/bert_fine_tuned/