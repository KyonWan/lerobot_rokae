#!/bin/bash

lerobot-train \
  --dataset.repo_id=place_metal_parts_filtered_v3.0 \
  --dataset.root=/root/autodl-tmp/data/place_metal_parts_filtered_v3.0 \
  --dataset.video_backend=torchcodec \
  --policy.type=diffusion \
  --policy.push_to_hub=false \
  --output_dir=/root/autodl-tmp/train_output/diffusion_rokae \
  --batch_size=32 \
  --steps=20000 \
  --save_freq=2000 \
  --num_workers=8 \
  --tolerance_s=0.2 \
  --wandb.enable=true \
  --wandb.project=lerobot-diffusion-policy \
  --job_name="place_metal_diffusion_batch32"