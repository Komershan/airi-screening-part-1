from algorithms.algorithm_register import GENERATOR_CLASS
import wandb
from dataclasses import asdict, dataclass
import os
import torch
import gymnasium as gym
import uuid
import pyrallis

import environments


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

@dataclass
class BenchmarkConfig:
    project: str = "Distillation-reproduction"
    group: str = "Test-Group"
    name: str = "Algorithm-Distillation"
    environment = "bandits-10-v0"
    checkpoints_dir: str = "./checkpoints"
    max_steps: int = 100

@pyrallis.wrap()
def benchmark(config: BenchmarkConfig):
    wandb_init(asdict(config))

    models = []

    for checkpoint in os.listdir(config.checkpoints_dir):
        loaded_checkpoint = torch.load(os.path.join(config.checkpoints_dir, checkpoint))

        train_config_dict = loaded_checkpoint["train_config"]

        train_config = GENERATOR_CLASS[loaded_checkpoint["class_name"]].load_config(train_config_dict)
        model = GENERATOR_CLASS[loaded_checkpoint["class_name"]](train_config)
        model.load_state_dict(loaded_checkpoint["state_dict"])
        models.append(model)

    for model in models:
        environment = gym.make(config.environment)
        environment.reset()
        cumulative_reward = 0
        model.eval()
        
        for step in range(config.max_steps):
            print(step)
            action = model.get_action(environment.observation_to_int())
            observation, reward, is_terminated, truncated, additional_info = environment.step(action)
            model.update_policy(action, reward)
            cumulative_reward += reward
            wandb.log({f"{model.config.name}_reward": cumulative_reward})

if __name__ == "__main__":
    benchmark()