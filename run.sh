#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python hygenq.py \
  --model mar_large --diffloss_d 8 --diffloss_w 1280 \
  --num_iter 64 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 \
  --resume pretrained_models/mar/mar_large \
  --class_num 1000 \
  --w_bits 8 --a_bits 8 \
  --exclude_layers "diffloss.net.input_proj,diffloss.net.final_layer.linear" \
  --input_quant --weight_quant --calib5
