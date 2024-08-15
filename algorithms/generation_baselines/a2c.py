import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import gymnasium as gym
from dataclasses import asdict, dataclass, make_dataclass
import uuid
import os
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method
import pyrallis
import copy
import torch
import wandb
import math
from tqdm import tqdm
from utils.utils import set_seed
import environments

try:
    set_start_method("spawn")
except RuntimeError:
    pass


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
class TrainConfig:
    project: str = "Algorithm-Distillation-Training-v6"
    group: str = "Baselines"
    name: str = "A2C"
    checkpoints_path: str = "./checkpoints/darkroom/a2c"
    environment_name: str = "Dark-Room-v0"

    seed: int = 42
    deterministic_torch: bool = True
    normalize_rewards: bool = True
    save_checkpoints: bool = True
    device: str = "cpu"

    num_inputs: int = 1
    num_actions: int = 5

    num_episodes_for_actor: int = 10000
    gradient_clipping_norm: float = 5.0
    num_actors: int = 1
    discount: float = 0.99
    mlp_layers_count: int = 3
    hidden_dimension: int = 32
    beta_1: float = 0.9
    beta_2: float = 0.999
    lr: float = 3e-4
    epsilon: float = 1e-2
    log_step: int = 1
    seed: int = 69
    episode_length: int = 20
    single_stream: bool = False
    save_format: str = "uint8"

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

        env = gym.make(self.environment_name)
        self.action_space_size = env.get_action_space_size()


class A2C(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()

        self.config = config

        # Initialize both Actor and Critic models
        network_layers = [
            nn.Linear(config.num_inputs, config.hidden_dimension, device=config.device),
            nn.ReLU(),
        ]

        for layer_count in range(config.mlp_layers_count - 1):
            network_layers.append(
                nn.Linear(
                    config.hidden_dimension,
                    config.hidden_dimension,
                    device=config.device,
                )
            )
            network_layers.append(nn.ReLU())

        # For actor append linear layer with num_actions and softmax

        actor_layers = network_layers
        actor_layers.append(
            nn.Linear(config.hidden_dimension, config.num_actions, device=config.device)
        )
        actor_layers.append(nn.Softmax(dim=1))

        self.actor = nn.Sequential(*actor_layers)

        critic_layers = network_layers[:-2]
        critic_layers.append(
            nn.Linear(config.hidden_dimension, 1, device=config.device)
        )

        self.critic = nn.Sequential(*critic_layers)

        self.actor.to(self.config.device)
        self.critic.to(self.config.device)

        for model_type in [self.actor, self.critic]:
            for layer in model_type:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform(layer.weight)

    def forward(self, x):
        x = x.to(self.config.device)
        actor_output = self.actor(x)
        critic_output = self.critic(x)
        return actor_output, critic_output

    def train_from_model(self, config: TrainConfig, environment):
        model, histories = train(config, environment)
        self.__dict__ = model.__dict__

        return histories

    def get_action(self, observation):
        state = np.array([observation])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.config.device)
        actor_output = self.actor(torch.Tensor([observation]))
        action = torch.argmax(actor_output).item()
        return action

    @staticmethod
    def load_config(config_dict: dict) -> TrainConfig:
        return TrainConfig(**config_dict)

    def update_policy(self, action: int, reward: int):
        pass


def calc_loss_on_episode(mapped_arguments):
    config, environment, model, optimizer = mapped_arguments
    state = environment.reset()[0]
    done = False

    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_log_action_probabilities = []
    critic_outputs = []

    curr_episode_length = 0

    while (not done) and curr_episode_length < config.episode_length:
        curr_episode_length += 1
        action, action_log_prob, critic_output = pick_action_and_get_critic_values(
            model.actor, model.critic, state, config
        )
        next_state, reward, done, truncated, info = environment.step(action)

        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)
        episode_log_action_probabilities.append(action_log_prob)
        critic_outputs.append(critic_output)
        state = next_state

    actor_loss, critic_loss, reward_return = calculate_total_loss(
        episode_rewards, episode_log_action_probabilities, critic_outputs, config
    )
    total_loss = actor_loss + critic_loss

    optimizer.zero_grad()
    total_loss.backward()
    gradients = [param.grad.clone() for param in model.parameters()]
    gradients = [gradient.detach().cpu().numpy() for gradient in gradients]

    history = np.array(
        [
            [episode_states[i], episode_actions[i], episode_rewards[i]]
            for i in range(curr_episode_length)
        ],
        dtype=getattr(np, config.save_format),
    ).flatten()

    return history, reward_return, gradients, total_loss.item()


def pick_action_and_get_critic_values(policy, value, state, config):
    """Picks an action using the policy"""
    state = np.array([state])
    state = torch.from_numpy(state).float().unsqueeze(0).to(config.device)
    actor_output = policy(state)
    critic_output = value(state)

    action_distribution = Categorical(actor_output)
    action = action_distribution.sample().cpu().numpy()

    if random.random() <= config.epsilon:
        action = random.randint(0, config.action_space_size - 1)
    else:
        action = int(action[0])

    actions_log_prob = action_distribution.log_prob(
        torch.Tensor([action]).to(config.device)
    ).to(config.device)

    return action, actions_log_prob, critic_output


def calculate_total_loss(
    episode_rewards, episode_log_action_probabilities, critic_outputs, config
):
    """Calculates the actor loss + critic loss"""

    # Here we calculate discounted rewards

    discounted_returns = [0]
    for index in range(len(episode_rewards)):
        discounted_returns.append(
            episode_rewards[-(index + 1)] + config.discount * discounted_returns[-1]
        )

    discounted_returns = discounted_returns[1:]
    discounted_returns.reverse()
    discounted_returns = np.array(discounted_returns)

    result_discounted_returns = discounted_returns[0]

    if config.normalize_rewards:
        discounted_returns = (discounted_returns - np.mean(discounted_returns)) / (
            np.std(discounted_returns) + 1e-5
        )

    critic_values = torch.cat(critic_outputs)
    advantages = torch.Tensor(discounted_returns).to(config.device) - critic_values
    advantages = advantages.detach().to(config.device)
    critic_loss = (
        torch.Tensor(discounted_returns).to(config.device) - critic_values
    ) ** 2
    critic_loss = critic_loss.mean().to(config.device)

    action_log_probabilities_for_all_episodes = torch.cat(
        episode_log_action_probabilities
    )

    actor_loss = -1.0 * action_log_probabilities_for_all_episodes * advantages
    actor_loss = actor_loss.mean()

    return actor_loss, critic_loss, result_discounted_returns


def train(config: TrainConfig, environment=None):
    set_seed(config.seed)
    if environment is None:
        environment = gym.make(config.environment_name)

    model = A2C(config)
    model.to(config.device)

    wandb_init(asdict(config))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, betas=[config.beta_1, config.beta_2]
    )

    model.share_memory()

    task_environments = []

    stream_histories = [np.array([], dtype="uint8") for i in range(config.num_actors)]

    for i in range(config.num_actors):
        task_environments.append(copy.deepcopy(environment))
        task_environments[-1].set_seed(i + config.seed)

    with mp.Pool(processes=config.num_actors) as pool:
        returns = []
        losses = []
        for episode_counter in tqdm(range(config.num_episodes_for_actor)):
            # config.epsilon = 1 / (math.sqrt(episode_counter + 1))
            args = [
                (
                    config,
                    task_environments[i],
                    copy.deepcopy(model),
                    copy.deepcopy(optimizer),
                )
                for i in range(config.num_actors)
            ]
            parallel_results = pool.map(calc_loss_on_episode, args)
            gradients = []
            index = 0
            for history, discounted_return, gradient, loss in parallel_results:
                print(history)
                stream_histories[index] = np.concatenate(
                    (stream_histories[index], history)
                )
                gradients.append(gradient)
                losses.append(loss)
                returns.append(discounted_return)
                index += 1

            model_parameters = list(model.parameters())

            for index in range(len(model_parameters)):
                param_gradients = []
                for gradient in gradients:
                    param_gradients.append(gradient[index])
                param_gradients = np.asarray(param_gradients, dtype=np.float32)
                param_gradients = np.sum(param_gradients, axis=0)
                model_parameters[index]._grad = torch.from_numpy(param_gradients)
                optimizer.step()

            if (episode_counter + 1) % config.log_step == 0:
                wandb.log({"discounted_returns": np.mean(np.array(returns))})
                wandb.log({"total_loss": np.mean(np.array(losses))})

    for index in range(len(stream_histories)):
        print(stream_histories[index].shape)

    if config.single_stream:
        stream_histories = stream_histories[0:1]

    model_state_dict = model.state_dict()

    save_dict = {
        "train_config": asdict(config),
        "class_name": "A2C",
        "state_dict": model_state_dict,
    }

    if config.save_checkpoints:
        torch.save(save_dict, config.checkpoints_path)

    wandb.finish()

    return model, stream_histories


if __name__ == "__main__":
    config = pyrallis.parse(config_class=TrainConfig)
    train(config)
