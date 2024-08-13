import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
from numpy.random import Generator, PCG64

class BanditsEnv(gym.Env):
    def __init__(
            self,
            n_arms: int,
            max_steps: int,
            generation_seed: int,
            distribution_type: str = "uniform",
            **kwargs
    ):
        self.mapping_dict = {
            'n_arms': 'int',
            'max_steps': 'int',
            'probs': 'array',
            'seed': 'int'
        }

        super().__init__(**kwargs)
        self.n_arms = n_arms
        self.max_steps = max_steps
        self.distribution_type = distribution_type
        self.rng = Generator(PCG64())
        self.probs = np.zeros(n_arms)
        self.action_number = 0
        self.observation = [0]
        self.action_space = Discrete(n_arms)
        self.observation_space = Discrete(1)
        self.seed = 69
        self._set_probs()

    def step(self, action: int):
        self.action_number += 1
        reward = int(self.rng.uniform(low=0, high=1) <= self.probs[action])
        return self.observation, reward, (self.action_number >= self.max_steps), None, {}
    
    def reset(self):
        super().reset(seed=self.seed)
        self._set_probs()
        return self.get_observation()

    def get_action_space_size(self):
        return 10

    def set_seed(self, seed=None) -> None:
        self.seed = seed
        self.rng = Generator(PCG64(seed=self.seed))

    def observation_to_int(self):
        return self.observation[0]
    
    def get_observation(self):
        return self.observation
    
    def get_params_dict(self):
        return {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
    
    def get_mapping_dict(self):
        return self.mapping_dict
    
    def load_from_dict(self, dict):
        self.__dict__.update(dict)
        self.rng = Generator(PCG64(seed=self.seed))

    def _set_probs(self):
        # Generate probs
        assert self.distribution_type in ["odd", "even", "uniform"]
        if self.distribution_type == "uniform":
            self.probs = np.random.random(self.n_arms)
        elif self.distribution_type in ["odd", "even"]:
            for i in range(0, 10, 2):
                low_prob = self.rng.uniform(low=0, high=0.05)
                if self.distribution_type == "odd":
                    self.probs[i] = 1 - low_prob
                    self.probs[i + 1] = low_prob
                else:
                    self.probs[i + 1] = 1 - low_prob
                    self.probs[i] = low_prob
        
    def __eq__(self, other):
        if type(other) is type(self):
            is_equal = True
            is_equal = is_equal and (self.n_arms == other.n_arms)
            is_equal = is_equal and not np.allclose(self.probs, other.probs)

            return is_equal

        return False