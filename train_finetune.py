import os
from typing import Tuple

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from dataset_utils import (Batch, D4RLDataset, ReplayBuffer, AdroitBinaryDataset, AdroitBinaryTruncDataset,
                           split_into_trajectories, MixingReplayBuffer)
from evaluation import evaluate
from learner import Learner
from utils import (WandBLogger, get_user_flags)
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 100,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('num_pretraining_steps', int(1e6),
                     'Number of pretraining steps.')
flags.DEFINE_integer('replay_buffer_size', 2000000,
                     'Replay buffer size (=max_steps if unspecified).')
flags.DEFINE_integer('init_dataset_size', None,
                     'Offline data size (uses all data if unspecified).')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_float('mixing_ratio', -1.0, 'the ratio of offline data in the batch')
flags.DEFINE_float('online_temperature', -1.0, 'the IQL temparature for online phase')
flags.DEFINE_float('online_expa_max', -1.0, 'the IQL temparature for online phase')
flags.DEFINE_boolean('truncate_demos', True, 'Truncate demos for Adroit or not')




config_flags.DEFINE_config_dict('logging', WandBLogger.get_default_config())

config_flags.DEFINE_config_file(
    'config',
    'configs/antmaze_finetune_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    print("ENV", env_name)
    if env_name in ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0"]:
        if FLAGS.truncate_demos:
            dataset = AdroitBinaryTruncDataset(env)
        else:
            dataset = AdroitBinaryDataset(env)
    else:
        dataset = D4RLDataset(env)

    if 'antmaze' in FLAGS.env_name:
        # dataset.rewards -= 1.0
        pass  # normalized in the batch instead
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)

    return env, dataset


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb',
                                                str(FLAGS.seed)),
                                   write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    variant = get_user_flags(FLAGS)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    if "binary" in FLAGS.env_name:
        import mj_envs

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    action_dim = env.action_space.shape[0]

    if FLAGS.mixing_ratio >= 0:
        offline_buffer = ReplayBuffer(env.observation_space, action_dim,FLAGS.replay_buffer_size)
        offline_buffer.initialize_with_dataset(dataset, FLAGS.init_dataset_size)
        online_buffer = ReplayBuffer(env.observation_space, action_dim,FLAGS.replay_buffer_size or FLAGS.max_steps)
        replay_buffer = MixingReplayBuffer([offline_buffer, online_buffer], mixing_ratio=1.0)
    else:
        replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                    FLAGS.replay_buffer_size or FLAGS.max_steps)
        replay_buffer.initialize_with_dataset(dataset, FLAGS.init_dataset_size)


    kwargs = dict(FLAGS.config)
    
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis], **kwargs)

    eval_returns = []
    observation, done = env.reset(), False
    env_steps = 0
    grad_steps = 0

    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(range(1,FLAGS.max_steps + FLAGS.num_pretraining_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):

        if i == FLAGS.num_pretraining_steps + 1:
            if FLAGS.mixing_ratio >= 0:
                replay_buffer.set_mixing_ratio(FLAGS.mixing_ratio)
            if FLAGS.online_temperature >= 0:
                print("changing temperature to:", FLAGS.online_temperature)
                agent.temperature = FLAGS.online_temperature

            if FLAGS.online_expa_max >= 0:
                print("changing advantage upper limit to:", FLAGS.online_expa_max)
                agent.expa_max=FLAGS.online_expa_max

        if i >= FLAGS.num_pretraining_steps + 1:
            action = agent.sample_actions(observation, )
            action = np.clip(action, -1, 1)
            next_observation, reward, done, info = env.step(action)
            env_steps += 1

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(observation, action, reward, mask,
                                 float(done), next_observation)
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    summary_writer.add_scalar(f'training/{k}', v,
                                              info['total']['timesteps'])
        else:
            info = {}
            info['total'] = {'timesteps': i}

        batch = replay_buffer.sample(FLAGS.batch_size)
        if 'antmaze' in FLAGS.env_name:
            batch = Batch(observations=batch.observations,
                          actions=batch.actions,
                          rewards=batch.rewards - 1,
                          masks=batch.masks,
                          next_observations=batch.next_observations)

        # initial evaluation
        if i == 1:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)
            wandb_logger.log(eval_stats, step=0)
        
        update_info = agent.update(batch)
        grad_steps += 1

        if i % FLAGS.log_interval == 0:
            wandb_logger.log({'env_steps': env_steps}, step=i)
            wandb_logger.log({'grad_steps': grad_steps}, step=i)
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                    wandb_logger.log({f'training/{k}': v}, step=i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
                    wandb_logger.log({f'training/{k}_mean': v.mean()}, step=i)
                    wandb_logger.log({f'training/{k}_max': v.max()}, step=i)
                    wandb_logger.log({f'training/{k}_min': v.min()}, step=i)
                    wandb_logger.log({f'training/{k}_std': v.std()}, step=i)

            additional_info = agent.log_diff_q(batch)
            for k, v in additional_info.items():
                wandb_logger.log({f'training/{k}': v}, step=i)

            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            wandb_logger.log(eval_stats, step=i)

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
