#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=6
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_DISABLED=True

# env=antmaze-medium-play-v2
# env=antmaze-medium-diverse-v2
# env=antmaze-large-play-v2
env=antmaze-large-diverse-v2

mixing_ratio=0.5
online_expa_max=100000

for seed in 42 43 44
do
python train_finetune.py \
--env_name=$env \
--config=configs/antmaze_finetune_config.py \
--eval_episodes=1 \
--eval_interval=20000 \
--replay_buffer_size=2000000 \
--save_dir=/raid/mitsuhiko/iql/${env}-${mixing_ratio} \
--num_pretraining_steps=1 \
--max_steps=1000000 \
--logging.online \
--seed $seed \
--mixing_ratio=$mixing_ratio \
--logging.project=IQL-offline2online-antmaze-AdvantageUpperLimit-sweep \
--online_expa_max=$online_expa_max

done
