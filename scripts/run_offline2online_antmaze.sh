#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=7
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export WANDB_DISABLED=True

# env=antmaze-medium-play-v2
# env=antmaze-medium-diverse-v2
# env=antmaze-large-play-v2
env=antmaze-large-diverse-v2

mixing_ratio=0.5
# online_expa_max=100000
online_temperature=10

# 7 8 9
for seed in 6
do
python train_finetune.py \
--env_name=$env \
--config=configs/antmaze_finetune_config.py \
--eval_episodes=50 \
--eval_interval=50000 \
--replay_buffer_size=2000000 \
--save_dir=/raid/mitsuhiko/iql/${env}-${mixing_ratio} \
--num_pretraining_steps=1000000 \
--max_steps=1000000 \
--logging.online \
--seed $seed \
--mixing_ratio=$mixing_ratio \
--logging.project=IQL-offline2online-antmaze-final \
--online_temperature=$online_temperature \
--log_interval=5000
done
