from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    returns = []
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        rewards = []
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, rew, done, info = env.step(action)
            rewards.append(rew)

        for k in stats.keys():
            stats[k].append(info['episode'][k])
        returns.append(np.sum(rewards))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    
    stats["average_return"] = np.mean(returns)
    stats["average_normalizd_return"] = np.mean([env.get_normalized_score(ret) for ret in returns])

    return stats
