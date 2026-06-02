#!/bin/bash
# ============================================================
# Experiment: iTransformer + NHiTS Head  |  ECL  |  H=96
# ============================================================

export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_nhits \
  --model iTransformerNHiTS \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
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
  2>&1 | tee logs/ECL_96_nhits.log