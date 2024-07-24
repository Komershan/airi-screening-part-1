import numpy as np
import pyrallis
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

@dataclass
class TrainConfig:

    def __post_init__(self):
        pass


class UCB(nn.Module):
    def __init__(
            self,
            n_arms: int = 10
    ):
        self.action_number = 0
        self.n_arms = n_arms
        self.usage_count = np.zeros(n_arms)
        self.arm_reward = np.zeros(n_arms)
    
    def get_action(self, observation):
        self.action_number += 1
        if self.action_number <= self.n_arms:
            return self.action_number - 1
        else:
            armes_ratings = self.arm_reward / self.action_number + np.sqrt(2 * math.log(self.action_number) / self.usage_count)
            return int(np.argmax(armes_ratings))

    def update_policy(self, action: int, reward: int):
        self.usage_count[action] += 1
        self.arm_reward[action] += reward

    def train(self, environment):
        pass


@pyrallis.wrap()
def train(config: TrainConfig):
    return UCB(config)

if __name__ == "__main__":
    train()