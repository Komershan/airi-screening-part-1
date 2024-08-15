'''
This script we use in order to generate environments and histories datasets for AD and ED training

I propose unified saving structure for environments and datasets. 
If you want to check it, you can do it in ./utils/dataset_utils.py
'''
import gymnasium as gym
import environments.bandits
from utils.dataset_utils import HistoriesDataset, envs_to_h5, h5_to_envs
from algorithms.algorithm_register import GENERATOR_CLASS, config_from_classname
from dataclasses import dataclass
import pyrallis
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import copy
from utils.utils import set_seed

@dataclass
class GenerationConfig:
    # Here we specify generator name for generating history
    generator_name: str = "A2C"
    # You can also provide generator_specs if you want to. 
    # Default parameters are stored in following TrainConfigs in definition files
    generator_specs = {}
    # Here we specify number of train and test tasks
    n_test_tasks: int = 10
    n_train_tasks: int = 10

    # Here we provide episode sizes and episode numbers for generating expert interactions for ED.
    episode_size: int = 100
    num_episodes: int = 10

    # Here we specify train and test environments names.
    train_env: str = "Dark-Room-v0"
    test_env: str = "Dark-Room-v0"

    # Here we provide saving paths for dataset, train tasks and test tasks
    dataset_name: str = "dark_room_dataset.hdf5"
    train_tasks_filename: str = "./data/darkroom/dark_room_train.hdf5"
    test_tasks_filename: str = "./data/darkroom/dark_room_test.hdf5"

    # You can specify load_train = True if you want to generate dataset using same train tasks
    # or generate different test tasks based on train_tasks
    load_train: bool = False
    # You must set this to false it if you want to generate expert trajectories instead of learning histories
    learning_histories: bool = True
    # You can set it to false if you don't want to generate dataset
    generate_dataset: bool = True

    # This is a dataset parameter. You can check it here: ./utils/dataset_utils.py
    resize_history: Optional[int] = 40
    filtering_window: Optional[int] = None

    seed: int = 11


def generate_histories(
    config: GenerationConfig, generator_config, train_tasks: list[gym.Env]
):
    # Init generator
    model = GENERATOR_CLASS[config.generator_name](generator_config)

    histories = []

    # Init dataset for storing histories
    dataset = HistoriesDataset(
        vocab_size=max(train_tasks[0].get_action_space_size(), train_tasks[0].get_observation_space_size()),
        resize_history=config.resize_history,
    )

    for environment in train_tasks:
        # Train algorithm and get history
        curr_history = []
        generator = copy.deepcopy(model)
        curr_history = generator.train_from_model(
            config=generator_config, environment=copy.deepcopy(environment)
        )

        environment.reset()

        if not config.learning_histories:
            # If we need to have expert trajectories, we change them to expert trajectories
            curr_history = []
            observation = []

            for episode_num in range(config.num_episodes):
                for n_steps in range(config.episode_size):
                    curr_history.append(environment.observation_to_int())
                    action = generator.get_action(observation)
                    curr_history.append(action)
                    observation, reward, is_terminated, truncated, additional_info = (
                        environment.step(action)
                    )
                    curr_history.append(reward)
                    generator.update_policy(action, reward)

            curr_history = [curr_history]
            
        dataset = HistoriesDataset(
            vocab_size=max(train_tasks[0].get_action_space_size(), train_tasks[0].get_observation_space_size()),
            resize_history=config.resize_history,
        )

        # Here we append all histories to dataset

        for history_part in curr_history:
            histories.append(history_part)

        for history in histories:
            dataset.append_data(np.array(history))

        # Save histories as hdf5 file (partly)
        dataset.write(config.dataset_name)

    dataset = HistoriesDataset(
        vocab_size=train_tasks[0].get_action_space_size(),
        resize_history=config.resize_history,
    )

    # If we want to, we can apply filtering window similar to Gato
    if not (config.filtering_window is None):
        max_index = 0
        max_value = 0
        curr_sum = 0
        partial_sums = []
        for i in range(len(histories)):
            curr_total_reward = sum(histories[i][1::3])
            curr_sum += curr_total_reward
            partial_sums.append(curr_total_reward)
            if i < config.filtering_window:
                max_value = curr_sum
            else:
                curr_sum -= partial_sums[i - config.filtering_window]
                if curr_sum > max_value:
                    max_value = curr_sum
                    max_index = i - config.filtering_window + 1

        train_tasks = train_tasks[max_index : max_index + config.filtering_window]
        histories = histories[max_index : max_index + config.filtering_window]

    for history in histories:
        dataset.append_data(np.array(history))

    # Save histories as hdf5 file
    dataset.write(config.dataset_name)

    return train_tasks


def generate_tasks(config: GenerationConfig):
    print("Generate tasks...")

    all_tasks = []

    num_tasks = config.n_train_tasks + config.n_test_tasks

    if config.load_train:
        all_tasks = h5_to_envs(config.train_tasks_filename, config.train_env)
        num_tasks = config.n_test_tasks

    for i in tqdm(range(num_tasks)):
        exit_flag = False
        while not exit_flag:
            environment_name = (
                config.train_env if i < config.n_train_tasks else config.test_env
            )
            if config.load_train:
                environment_name = config.test_env

            # Here we generate task applying random seed
            # If we compare it and it doesn't match with all tasks, then we continue to generate
            # Else we regenerate task

            curr_task = gym.make(environment_name, seed=np.random.randint(10**9 + 7))
            curr_task.reset()
            exit_flag = True
            for compare_task in all_tasks:
                exit_flag = exit_flag and not (curr_task == compare_task)
            if exit_flag:
                all_tasks.append(curr_task)

    np.random.shuffle(np.array(all_tasks))

    return all_tasks[: config.n_train_tasks], all_tasks[config.n_train_tasks :]


@pyrallis.wrap()
def main(generation_config: GenerationConfig):
    generator_config_class = config_from_classname(generation_config.generator_name)
    generator_config = generator_config_class()
    generator_config.__dict__.update(generation_config.generator_specs)

    set_seed(generation_config.seed)

    print(f"Train tasks environment name: {generation_config.train_env}")
    print(f"Test tasks environment name: {generation_config.test_env}")
    print(f"Generator name: {generation_config.generator_name}")
    print(f"Generator specs: {generator_config.__dict__}")

    # Generate tasks as environments

    train_tasks, test_tasks = generate_tasks(generation_config)

    if generation_config.generate_dataset:
        # Generate histories
        train_tasks = generate_histories(
            generation_config, generator_config, train_tasks
        )

    # Save train and test tasks for reproducibility purposes
    envs_to_h5(train_tasks, generation_config.train_tasks_filename)
    envs_to_h5(test_tasks, generation_config.test_tasks_filename)


if __name__ == "__main__":
    main()
