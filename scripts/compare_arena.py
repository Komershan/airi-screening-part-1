'''
This compare arena we use for benchmark
Here we log cumulative rewards from benchmark and normalize them relative to baseline
Using wandb logging
'''

from algorithms.algorithm_register import GENERATOR_CLASS
import wandb
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import os
import torch
import pyrallis
import numpy as np
import copy
from algorithms.algorithm_register import config_from_classname

from tqdm import tqdm

import environments
from utils.dataset_utils import h5_to_envs
from algorithms.generation_baselines.random_agent import Random

from utils.utils import set_seed, wandb_init

@dataclass
class BenchmarkConfig:
    # Project, group and name are used for wandb initialization
    project: str = "Compare-AD-even-v1"
    group: str = "Test-Group"
    name: str = "Algorithm-Distillation"
    # Here we provide test environment name
    environment = "bandits-odd-v0"
    # Here we provide baseline name and baseline_checkpoint
    # If baseline is not specified, then we don't calculate normalized reward related to baseline
    baseline: Optional[str] = None
    baseline_checkpoint: Optional[str] = None
    # Here we provide baseline specs if they differs from related TrainConfig 
    baseline_specs = {
    }
    # In checkpoints_dir located some checkpoints that we need to evaluate
    checkpoints_dir: str = "./checkpoints/"
    # And also we need to provide saved test_tasks
    test_tasks_filename: str = "./data/bandits/test_tasks_uniform.hdf5"
    # Here we provide episode_size and episode_count for models testing
    episode_size: int = 100
    num_episodes: int = 10
    # And also we provide discount for discounted_reward calculating
    discount: float = 0.99

    # We need to set eval_seed in order to generate num_eval_seeds for generation.
    eval_seed: int = 32
    num_eval_seeds: int = 10
    device: str = "cuda"

# I use run_env function for evaluating execution on checkpoint model and sampled task
def run_env(model, config, agent_name: str, sampled_task, finish=True):

    # We init wandb and model
    config.name = agent_name
    wandb_init(asdict(config))
    source_model = copy.deepcopy(model)

    # Then we copy environment and start to eval model
    environment = copy.deepcopy(sampled_task)
    environment.reset()
    across_episodic_reward, across_discounted_rewards = 0, 0
    for episode in range(config.num_episodes):
        episode_rewards = []
        # Here we run model on episode
        for step in range(config.episode_size):
            action = model.get_action(environment.observation_to_int())
            observation, reward, is_terminated, truncated, additional_info = environment.step(action)
            model.update_policy(action, reward)
            episode_rewards.append(int(reward))
        if not config.environment.startswith('bandits'):
            environment.reset()

        # Here we calculate all reward types for logging
        across_episodic_reward += sum(episode_rewards)        
        discounted_returns = [0]
        for index in range(len(episode_rewards)):
            discounted_returns.append(episode_rewards[-(index + 1)] + config.discount * discounted_returns[-1])
        discounted_returns = discounted_returns[1:]
        discounted_returns.reverse()
        across_discounted_rewards += discounted_returns[0]

        wandb.log({
            f"discounted_reward": across_discounted_rewards,
            f"cumulative_reward": across_episodic_reward,
            f"episode_reward": sum(episode_rewards),
            f"episode_discount_reward": discounted_returns[0]
        })

    model = source_model
    if finish:
        wandb.finish()
    return across_episodic_reward

# This def runs overall benchmark pipeline
@pyrallis.wrap()
def benchmark(config: BenchmarkConfig):

    # Firstly we set seed
    set_seed(config.eval_seed)

    models = []
    checkpoints_names = []

    # Then we load tasks and checkpoint
    test_tasks = h5_to_envs(env_name=config.environment, fname=config.test_tasks_filename)
    for checkpoint in os.listdir(config.checkpoints_dir):
        if checkpoint.endswith('.pt'):
            checkpoints_names.append(checkpoint)
            loaded_checkpoint = torch.load(os.path.join(config.checkpoints_dir, checkpoint), map_location=torch.device('cpu'))
            train_config_dict = loaded_checkpoint["train_config"]

            # GENERATOR_CLASS map needs for loading TrainConfig exactly for checkpoint parameters
            train_config = GENERATOR_CLASS[loaded_checkpoint["class_name"]].load_config(train_config_dict)
            model = GENERATOR_CLASS[loaded_checkpoint["class_name"]](train_config)
            model.load_state_dict(loaded_checkpoint["state_dict"])
            models.append(model)

    # Generate seeds
    eval_seeds = set()

    while len(eval_seeds) < config.num_eval_seeds:
        eval_seeds.add(np.random.randint(10 ** 9 + 7))


    # Here we start to eval checkpointed models for all eval_seeds
    for eval_seed in eval_seeds:

        set_seed(eval_seed, device="cuda")
        sampled_task = test_tasks[np.random.randint(len(test_tasks))]
        total_reward_random, total_reward_baseline = 0, 0

        # if we have baseline, then we need to score random agent and baseline
        if not (config.baseline is None):

            random_agent = Random(seed=eval_seed, action_space=sampled_task.action_space)
            total_reward_random = run_env(random_agent, config, agent_name="Random", sampled_task=sampled_task)

            # This part is same as checkpoint loading. 
            # Except that if we don't have checkpoint for baseline, we train baseline from sampled task
            if config.baseline_checkpoint is None:
                baseline_config_class = config_from_classname(config.baseline)
                baseline_config = baseline_config_class()
                baseline_config.__dict__.update(config.baseline_specs)

                baseline = GENERATOR_CLASS[config.baseline](baseline_config)
                baseline.train(baseline_config, copy.deepcopy(sampled_task))
            else:
                loaded_checkpoint = torch.load(config.baseline_checkpoint)
                train_config_dict = loaded_checkpoint["train_config"]

                train_config = GENERATOR_CLASS[loaded_checkpoint["class_name"]].load_config(train_config_dict)
                baseline = GENERATOR_CLASS[loaded_checkpoint["class_name"]](train_config)
                baseline.load_state_dict(loaded_checkpoint["state_dict"])
                baseline.eval()
        
            total_reward_baseline = run_env(baseline, config, agent_name=f"Baseline ({config.baseline})", sampled_task=sampled_task)

        # Here we evaluate checkpoint models and normalize rewards if we know it
        for checkpoint, checkpoint_model in tqdm(zip(checkpoints_names, models)):
            if checkpoint.endswith(".pt"):
                checkpoint_model.eval()
                total_reward_model = run_env(checkpoint_model, config, agent_name=f"{checkpoint}", sampled_task=sampled_task, finish=False)
                
                if not (config.baseline is None):
                    wandb.log({"normalized_rewards":(total_reward_model - total_reward_random) / (total_reward_baseline - total_reward_random)})
                wandb.finish()


if __name__ == "__main__":
    benchmark()