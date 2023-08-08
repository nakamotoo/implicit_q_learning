#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=7

# export WANDB_DISABLED=True

# halfcheetah-random-v2, halfcheetah-medium-v2, halfcheetah-medium-replay-v2, halfcheetah-medium-expert-v2, halfcheetah-expert-v2
# hopper-random-v2, hopper-medium-v2, hopper-medium-replay-v2, hopper-medium-expert-v2, hopper-expert-v2
# walker2d-random-v2, walker2d-medium-v2, walker2d-medium-replay-v2, walker2d-medium-expert-v2, walker2d-expert-v2

env=hopper-random-v2

mixing_ratio=0.5
# online_expa_max=100000
# 10 50 100
online_temperature=100

# 7 8 9
for seed in 0 1 2
do
python train_finetune.py \
--env_name=$env \
--config=configs/mujoco_config.py \
--eval_episodes=10 \
--eval_interval=50000 \
--replay_buffer_size=2000000 \
--save_dir=/raid/mitsuhiko/iql/${env}-${mixing_ratio} \
--num_pretraining_steps=500000 \
--max_steps=1000000 \
--logging.online \
--seed $seed \
--mixing_ratio=$mixing_ratio \
--logging.project=0806-locomotion-IQL \
--online_temperature=$online_temperature \
--log_interval=5000
done