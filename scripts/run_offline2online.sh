#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=3
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# env=antmaze-medium-play-v2
# env=antmaze-medium-diverse-v2
# env=antmaze-large-play-v2
env=antmaze-large-diverse-v2

python train_finetune.py \
--env_name=$env \
--config=configs/antmaze_finetune_config.py \
--eval_episodes=50 \
--eval_interval=50000 \
--replay_buffer_size=2000000 \
--save_dir=/raid/mitsuhiko/iql/${env}
