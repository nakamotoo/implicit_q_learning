#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=3
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_DISABLED=True

env=kitchen-mixed-v0
# env=kitchen-partial-v0
# env=kitchen-complete-v0


mixing_ratio=0.5

for seed in 1 2 3 4 5
do
python train_finetune.py \
--env_name=$env \
--config=configs/kitchen_finetune_config.py \
--eval_episodes=20 \
--eval_interval=2000 \
--replay_buffer_size=2000000 \
--save_dir=/raid/mitsuhiko/iql/${env}-${mixing_ratio} \
--num_pretraining_steps=4000 \
--max_steps=2500000 \
--logging.online \
--seed $seed \
--mixing_ratio=$mixing_ratio \
--logging.project=IQL-offline2online-kitchen-0123 \

done
