from utils.dataset_utils import HistoriesDataset
from algorithms.minGPT.model import GPT
from algorithms.minGPT.trainer import Trainer
from dataclasses import asdict, dataclass, make_dataclass
import os
import uuid

import pyrallis

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TrainConfig:
    project: str = "Algorithm-Distillation"
    group: str = "In-Context-Algorithms"
    name: str = "Algorithm-Distillation-GPT"
    dataset_path: str = "./dataset.hdf5"
    checkpoints_path: str = "./checkpoints"

    seed: int = 42
    eval_seed: int = 0  # Eval environment seed
    test_seed: int = 69
    deterministic_torch: bool = True
    device: str = "cuda"

    online_iterations: int = int(1e2)  # Number of online updates
    batch_size: int = 16
    eval_frequency: int = 1000
    n_test_episodes: int = 10
    normalize_reward: bool = False

    gpt_type: str = "gpt-nano"
    vocab_size: int = 10
    block_size: int = 30

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class AD(GPT):
    def __init__(self, config: TrainConfig):

        self.config = config

        gpt_config = GPT.get_default_config()
        gpt_config.model_type = config.gpt_type
        gpt_config.vocab_size = config.vocab_size
        gpt_config.block_size = config.block_size

        super().__init__(gpt_config)

        self.history = [] # Save history to generate next actions

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None

        if self.training:
            labels = idx.clone().detach()
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            mask = torch.tensor([int(i % 3 == 0) for i in range(t - 1)], dtype=torch.float64).to(device)
            loss_function = nn.NLLLoss2d(reduce=False)
            loss_expanded = loss_function(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_expanded *= mask
            loss = torch.sum(loss_expanded)

        return logits, loss
    
    def get_action(self, observation: int):
        self.history.append(observation)
        logits, loss = self.forward(torch.tensor([self.history[-self.config.block_size:]]))
        logits = logits[:, -1, :]
        print(logits, logits.size(), torch.argmax(logits))
        action = torch.argmax(logits).item()
        self.history.append(action)
        return action
    
    def update_policy(self, action: int, reward: int):
        self.history.append(reward)

    @staticmethod
    def load_config(config_dict: dict) -> TrainConfig:
        return TrainConfig(**config_dict)


@pyrallis.wrap()
def train(config: TrainConfig):
    model = AD(config)
    dataset = HistoriesDataset(config.dataset_path)

    config.vocab_size = dataset.vocab_size

    # create a Trainer object
    from algorithms.minGPT.trainer import Trainer
    
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    train_config.max_iters = 2000
    train_config.num_workers = 0
    train_config.batch_size=1
    trainer = Trainer(train_config, model, dataset)
    
    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")


    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    model_state_dict = model.state_dict()

    save_dict = {
        'train_config': asdict(config),
        'class_name': 'AD',
        'state_dict': model_state_dict
    }

    torch.save(save_dict, config.checkpoints_path)

    return model

    

if __name__ == "__main__":
    train()