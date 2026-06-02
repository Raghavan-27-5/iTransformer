#!/bin/bash
# ============================================================
# Experiment: iTransformer + NHiTS Head  |  VayuMithra  |  H=96
# Dataset   : 10Y hourly, 32 stations × 4 vars = 128 variates
# ============================================================

export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/vayumithra/ \
  --data_path vayumithra_10y.csv \
  --model_id vayumithra_96_nhits \
  --model iTransformerNHiTS \
  --data vayumithra \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 128 \
  --dec_in 128 \
  --c_out 128 \
  --d_model 512 \
  --d_ff 512 \
  --n_heads 8 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 \
  --use_norm 1 \
  --nhits_n_stacks 3 \
  --nhits_dropout 0.1 \
  --des 'nhits_head' \
  --itr 1 \
  2>&1 | tee logs/vayumithra_96_nhits.log