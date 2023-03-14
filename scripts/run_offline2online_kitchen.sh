#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export WANDB_DISABLED=True

env=kitchen-mixed-v0
env=kitchen-partial-v0
env=kitchen-complete-v0


mixing_ratio=0.25

for seed in 102
do
python train_finetune.py \
--env_name=$env \
--config=configs/kitchen_finetune_config.py \
--eval_episodes=20 \
--eval_interval=20000 \
--replay_buffer_size=2000000 \
--save_dir=/raid/mitsuhiko/iql/${env}-${mixing_ratio} \
--num_pretraining_steps=500000 \
--max_steps=2500000 \
--logging.online \
--seed $seed \
--logging.project=IQL-offline2online-kitchen-0123 \
--mixing_ratio=$mixing_ratio \

done
