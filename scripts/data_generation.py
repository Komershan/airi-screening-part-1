import gymnasium as gym

import environments.bandits

from utils.dataset_utils import HistoriesDataset

from algorithms.algorithm_register import GENERATOR_CLASS

from dataclasses import dataclass
import pyrallis
from typing import Dict, List, Optional
import numpy as np


@dataclass
class GenerationConfig:
    environments_list = ["bandits-10-v0"]
    generator_name: str = "ucb"
    generator_specs = {
        "n_arms": 10
    }
    file_name: str = "dataset.hdf5"
    resize_history: Optional[int] = 2

def generate_histories(config: GenerationConfig):

    histories = None

    for environment_name in config.environments_list:
        # Init generator
        generator = GENERATOR_CLASS[config.generator_name](**config.generator_specs)
        
        # Init environment
        environment = gym.make(environment_name)

        if histories is None:
            histories = HistoriesDataset(vocab_size=environment.get_action_space_size(), resize_history=config.resize_history)
        
        # Train algorithm
        generator.train(environment)
        environment.reset()

        observation = []
        curr_history = []

        is_terminated = False
        while not is_terminated:
            curr_history.append(environment.observation_to_int())
            action = generator.get_action(observation)
            curr_history.append(action)
            observation, reward, is_terminated, truncated, additional_info = environment.step(action)
            curr_history.append(reward)
            generator.update_policy(action, reward)

        histories.append_data(np.array(curr_history, dtype=np.int32))
        print(len(curr_history))


    # Save histories as hdf5 file
    histories.write(config.file_name)


@pyrallis.wrap()
def main(config: GenerationConfig):
    print(f"Environment name: {config.environments_list}")
    print(f"Generator name: {config.generator_name}")
    print(f"Generator specs: {config.generator_specs}")

    # Generate histories
    generate_histories(config)


if __name__ == "__main__":
    main()