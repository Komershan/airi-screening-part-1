import numpy as np
import pyrallis
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

import gymnasium as gym

@dataclass
class TrainConfig:

    environment: str = 'bandits-odd-v0'
    episode_size: int = 100
    num_episodes: int = 10
    num_train_steps: int = 2000

    def __post_init__(self):
        env = gym.make(self.environment)
        self.n_arms = env.get_action_space_size()
        pass


class UCB:
    def __init__(self, config: TrainConfig):
        self.action_number = 0
        self.n_arms = config.n_arms
        self.usage_count = np.zeros(config.n_arms)
        self.arm_reward = np.zeros(config.n_arms)
    
    def get_action(self, observation):
        self.action_number += 1
        if self.action_number <= self.n_arms:
            return self.action_number - 1
        else:
            armes_ratings = self.arm_reward / self.usage_count + np.sqrt(2 * math.log(self.action_number) / self.usage_count)
            return int(np.argmax(armes_ratings))

    def update_policy(self, action: int, reward: int):
        self.usage_count[action] += 1
        self.arm_reward[action] += reward

    def train(self, config: TrainConfig):
        model, histories = train(config)
        self.__dict__ = model.__dict__

        return histories


def train(config: TrainConfig):
    environment = gym.make(config.environment)
    environment.reset()

    model = UCB(config)

    curr_history = []
    observation = []

    for n_steps in range(config.num_train_steps):
        curr_history.append(environment.observation_to_int())
        action = model.get_action(observation)
        curr_history.append(action)
        observation, reward, is_terminated, truncated, additional_info = environment.step(action)
        curr_history.append(reward)
        model.update_policy(action, reward)


    for episode_num in range(config.num_episodes):
        for n_steps in range(config.episode_size):
            curr_history.append(environment.observation_to_int())
            action = model.get_action(observation)
            curr_history.append(action)
            observation, reward, is_terminated, truncated, additional_info = environment.step(action)
            curr_history.append(reward)
            model.update_policy(action, reward)

    return model, [curr_history]


if __name__ == "__main__":
    config = pyrallis.parse(config_class=TrainConfig)
    train(config)