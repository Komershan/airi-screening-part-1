from algorithms.algorithm_register import GENERATOR_CLASS
import wandb
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import os
import torch
import gymnasium as gym
import uuid
import pyrallis
import numpy as np
import copy
import matplotlib.pyplot as plt
from algorithms.algorithm_register import config_from_classname

from tqdm import tqdm
from collections import defaultdict

import environments
from utils.dataset_utils import h5_to_envs
from algorithms.generation_baselines.random_agent import Random

from utils.utils import set_seed, wandb_init

@dataclass
class BenchmarkConfig:
    project: str = "Distillation-reproduction-test-v55"
    group: str = "Test-Group"
    name: str = "Algorithm-Distillation"
    environment = "bandits-odd-v0"
    baseline: str = "UCB"
    baseline_checkpoint: Optional[str] = None
    generator_specs = {
    }
    checkpoints_dir: str = "./checkpoints/bandits"
    test_tasks_filename: str = "./data/bandits/test_tasks_odd.hdf5"
    episode_size: int = 100
    num_episodes: int = 10
    discount: float = 0.99

    eval_seed: int = 39
    num_eval_seeds: int = 1
    device: str = "cuda"

def run_env(model, config, agent_name: str, sampled_task):
    config.name = agent_name
    wandb_init(asdict(config))

    source_model = copy.deepcopy(model)

    environment = copy.deepcopy(sampled_task)
    environment.reset()
    across_episodic_reward, across_discounted_rewards = 0, 0
    episode_rewards = []
    for episode in range(config.num_episodes):
        for step in range(config.episode_size):
            action = model.get_action(environment.observation_to_int())
            observation, reward, is_terminated, truncated, additional_info = environment.step(action)
            model.update_policy(action, reward)
            episode_rewards.append(reward)
            if is_terminated:
                break
        across_episodic_reward += sum(episode_rewards)
        wandb.log({f"discounted_reward": across_discounted_rewards, f"cumulative_reward": across_episodic_reward})
        
        discounted_returns = [0]
        for index in range(len(episode_rewards)):
            discounted_returns.append(episode_rewards[-(index + 1)] + config.discount * discounted_returns[-1])
        discounted_returns = discounted_returns[1:]
        discounted_returns.reverse()
        across_discounted_rewards += discounted_returns[0]

    model = source_model
    wandb.finish()
    return across_episodic_reward

@pyrallis.wrap()
def benchmark(config: BenchmarkConfig):

    set_seed(config.eval_seed)

    models = []

    test_tasks = h5_to_envs(env_name=config.environment, fname=config.test_tasks_filename)
    for checkpoint in os.listdir(config.checkpoints_dir):
        loaded_checkpoint = torch.load(os.path.join(config.checkpoints_dir, checkpoint))
        train_config_dict = loaded_checkpoint["train_config"]

        del train_config_dict['eval_seed']
        del train_config_dict['test_seed']
        
        train_config = GENERATOR_CLASS[loaded_checkpoint["class_name"]].load_config(train_config_dict)
        model = GENERATOR_CLASS[loaded_checkpoint["class_name"]](train_config)
        model.load_state_dict(loaded_checkpoint["state_dict"])
        models.append(model)

    # Generate seeds

    eval_seeds = set()

    while len(eval_seeds) < config.num_eval_seeds:
        eval_seeds.add(np.random.randint(10 ** 9 + 7))

    print(eval_seeds)


    for eval_seed in eval_seeds:

        set_seed(eval_seed, device="cuda")

        sampled_task = test_tasks[np.random.randint(len(test_tasks))]
            
        total_reward_random, total_reward_baseline = 0, 0
            
        if not (config.baseline is None):

            print(sampled_task.action_space)

            random_agent = Random(seed=eval_seed, action_space=sampled_task.action_space)
            total_reward_random = run_env(random_agent, config, agent_name="Random", sampled_task=sampled_task)
                
            if config.baseline_checkpoint is None:
                baseline_config_class = config_from_classname(config.baseline)
                baseline_config = baseline_config_class()
                baseline_config.__dict__.update(config.generator_specs)

                baseline = GENERATOR_CLASS[config.baseline](baseline_config)
                baseline.train(baseline_config)
            else:
                loaded_checkpoint = torch.load(config.baseline_checkpoint)
                train_config_dict = loaded_checkpoint["train_config"]
            
                baseline = GENERATOR_CLASS[loaded_checkpoint["class_name"]](train_config)
                train_config = GENERATOR_CLASS[loaded_checkpoint["class_name"]].load_config(train_config_dict)
                baseline.load_state_dict(loaded_checkpoint["state_dict"])
                baseline.eval()
        
            total_reward_baseline = run_env(baseline, config, agent_name=f"Baseline ({config.baseline})", sampled_task=sampled_task)

            normalized_rewards = {}

        for checkpoint, checkpoint_model in tqdm(zip(os.listdir(config.checkpoints_dir), models)):
            if checkpoint.endswith(".pt"):
                checkpoint_model.eval()
                total_reward_model = run_env(checkpoint_model, config, agent_name=f"{checkpoint}", sampled_task=sampled_task)
                
                if not (config.baseline is None):
                    normalized_rewards[checkpoint] = (total_reward_model - total_reward_random) / (total_reward_baseline - total_reward_random)
                
        wandb.init(project=config.project, name=f"normalized_barplot")

        '''
        plt.rcParams['figure.figsize'] = [10, 10]
        
        fig, ax = plt.subplots()
        
        labels = normalized_rewards.keys()
        counts = normalized_rewards.values()
        bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
        
        rects = ax.bar(labels, counts, color=bar_colors, label=labels)
        ax.bar_label(rects, padding=3)
        
        ax.set_ylabel('fruit supply')
        ax.set_title('Fruit supply by kind and color')
        ax.legend(title='Fruit color')
        plt.savefig('foo.png')
        '''

        data = [[label, val] for (label, val) in normalized_rewards.items()]
        print(data)
        table = wandb.Table(data=data, columns=["label", "value"])
        wandb.log(
            {
                "my_bar_chart_id": wandb.plot.bar(
                    table, "label", "value", title="Custom Bar Chart"
                )
            }
        )
        wandb.finish()


if __name__ == "__main__":
    benchmark()