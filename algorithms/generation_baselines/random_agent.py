import numpy as np
from numpy.random import Generator, PCG64
import pyrallis
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import gymnasium as gym

@dataclass
class TrainConfig:

    def __post_init__(self):
        pass


class Random(nn.Module):
    def __init__(
            self,
            seed: int = 69,
            action_space = gym.spaces.Discrete,
    ):
        self.seed = seed
        self.action_space = action_space
    
    def get_action(self, observation):
        return self.action_space.sample()

    def update_policy(self, action: int, reward: int):
        pass

    def train(self, environment):
        pass


@pyrallis.wrap()
def train(config: TrainConfig):
    return Random(config)

if __name__ == "__main__":
    train()