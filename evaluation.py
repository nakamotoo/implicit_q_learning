from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    returns = []
    if env.spec.name in ['pen-binary', 'door-binary', 'relocate-binary']:
        goal_achieved = []
    if 'kitchen' in env.spec.name:
        num_stages_solved = []

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        rewards = []
        infos = []
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation, rew, done, info = env.step(action)
            rewards.append(rew)
            infos.append(info)

        for k in stats.keys():
            stats[k].append(info['episode'][k])
        returns.append(np.sum(rewards))
        if env.spec.name in ['pen-binary', 'door-binary', 'relocate-binary']:
            goal_achieved.append(int(np.any([i['goal_achieved'] for i in infos])))

        if 'kitchen' in env.spec.name:
            num_stages_solved.append(rewards[-1])

    for k, v in stats.items():
        stats[k] = np.mean(v)
    stats["average_return"] = np.mean(returns)
    if env.spec.name in ['pen-binary', 'door-binary', 'relocate-binary']:
        stats["goal_achieved_rate"] = np.mean(goal_achieved)
    else:
        stats["average_normalized_return"] = np.mean([env.get_normalized_score(ret) for ret in returns])

    if 'kitchen' in env.spec.name:
        stats["num_stages_solved"] = np.mean(num_stages_solved)

    return stats
