#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=4
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export WANDB_DISABLED=True

# env=antmaze-medium-play-v2
# env=antmaze-medium-diverse-v2
# env=antmaze-large-play-v2
# env=antmaze-large-diverse-v2

# env=pen-binary-v0
# env=door-binary-v0
env=relocate-binary-v0

mixing_ratio=0.5

for seed in 5
do
python train_finetune.py \
--env_name=$env \
--config=configs/adroit_finetune_config.py \
--eval_episodes=50 \
--eval_interval=2000 \
--replay_buffer_size=2000000 \
--save_dir=/raid/mitsuhiko/iql/${env}-${mixing_ratio} \
--num_pretraining_steps=20000 \
--max_steps=3000000 \
--logging.online \
--seed $seed \
--mixing_ratio=$mixing_ratio \
--logging.project=IQL-offline2online-adroit-truncation_0118 \

done
