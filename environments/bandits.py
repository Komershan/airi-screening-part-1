import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np

class BanditsEnv(gym.Env):
    def __init__(
            self,
            n_arms: int,
            state_size: int,
            max_steps: int,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_arms = n_arms
        self.max_steps = max_steps
        self.random_generator = np.random.default_rng()
        self.probs = self.random_generator.uniform(low=0, high=1, size=n_arms)
        self.action_number = 0
        self.observation = [0]
        self.action_space = Discrete(n_arms)
        self.observation_space = Discrete(1)

    def get_action_space_size(self):
        return self.n_arms
    
    def get_observation_space_size(self):
        return 1

    def step(self, action: int):
        self.action_number += 1
        reward = int(self.random_generator.uniform(low=0, high=1) <= self.probs[action])
        return self.observation, reward, (self.action_number >= self.max_steps), None, {}
    
    def reset(self):
        self.random_generator = np.random.default_rng()
        self.probs = self.random_generator.uniform(low=0, high=1, size=self.n_arms)

    def seed(self, seed=None) -> None:
        np.random.seed(seed)

    def observation_to_int(self):
        return self.observation[0]
        