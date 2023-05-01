#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=7
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export D4RL_SUPPRESS_IMPORT_ERROR=1
# export WANDB_DISABLED=True

# env=kitchen-mixed-v0
# env=kitchen-partial-v0
env=kitchen-complete-v0


mixing_ratio=0.25

#  3 4 5
for seed in 5
do
python train_finetune.py \
--env_name=$env \
--config=configs/kitchen_finetune_config.py \
--eval_episodes=10 \
--eval_interval=50000 \
--replay_buffer_size=2000000 \
--save_dir=/nfs/kun2/users/mitsuhiko/logs/iql/${env}-${mixing_ratio} \
--num_pretraining_steps=500000 \
--max_steps=2000000 \
--logging.online \
--seed $seed \
--logging.project=IQL-offline2online-kitchen-for-Neurips \
--mixing_ratio=$mixing_ratio \

done
