#!/bin/bash
# ============================================================
# Experiment: iTransformer + Linear Head  |  Weather  |  H=96
# Purpose   : Baseline sanity check — must reproduce your
#             earlier numbers (MSE ~0.17354, MAE ~0.21x)
# ============================================================

export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_baseline \
  --model iTransformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 512 \
  --d_ff 512 \
  --n_heads 8 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 \
  --use_norm 1 \
  --des 'baseline' \
  --itr 1 \
  2>&1 | tee logs/weather_96_baseline.log