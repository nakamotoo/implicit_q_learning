#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export WANDB_DISABLED=True

# halfcheetah-medium-v2, halfcheetah-medium-replay-v2, halfcheetah-medium-expert-v2, halfcheetah-random-v2, halfcheetah-expoert-v2
env=halfcheetah-medium-v2

mixing_ratio=0.5
# online_expa_max=100000
# online_temperature=10

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
--log_interval=5000
done
# --online_temperature=$online_temperature \
