from utils.dataset_utils import HistoriesDataset
import math
from dataclasses import asdict, dataclass, make_dataclass
from typing import Optional
import os
import uuid

import pyrallis

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.utils import set_seed


from algorithms.utils import Block
import wandb
from utils.utils import wandb_init

@dataclass
class TrainConfig:
    project: str = "Algorithm-Distillation"
    group: str = "In-Context-Algorithms"
    name: str = "Expert-Distillation-GPT"
    dataset_path: str = "./data/bandits/ED_dataset.hdf5"
    checkpoints_path: str = "./checkpoints"

    seed: int = 42
    deterministic_torch: bool = True
    device: str = "cuda"

    n_layer: int = 8
    n_head: int = 16
    n_embd: int = 512

    embd_pdrop: int = 0.1
    resid_pdrop: int = 0.1
    attn_pdrop: int = 0.1

    mask_prob: float = 0.0

    online_iterations: int = int(1e2)  # Number of online updates
    batch_size: int = 8
    eval_frequency: int = 1000
    n_test_episodes: int = 10
    normalize_reward: bool = False

    learning_rate: float = 1e-3
    max_iters: int = 5000
    num_workers: int = 0

    vocab_size: Optional[int] = None

    block_size: int = 600

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
        dataset = HistoriesDataset(self.dataset_path)
        if self.vocab_size is None:
            self.vocab_size = dataset.vocab_size
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


class AD(nn.Module):
    def __init__(self, config: TrainConfig):

        self.config = config

        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

        self.history = [] # Save history to generate next actions

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

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
            logits = nn.functional.log_softmax(logits, dim=1)
            labels = idx.clone().detach()
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            mask = torch.tensor([int((i  %(t - 1)) % 3 == 0) for i in range((t - 1) * b)], dtype=torch.float64).to(device)
            random_probs = np.random.random((b, t - 1))
            np_mask = np.fromfunction(lambda i, j: (j % 3 == 0), shape=(b, t - 1))
            mask = torch.tensor(np_mask * random_probs > self.config.mask_prob, dtype=torch.float64).to(device)
            loss_function = nn.NLLLoss(reduce=False, reduction=None)
            loss_expanded = loss_function(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_expanded *= (mask.view(1, -1)[0])
            loss = torch.sum(loss_expanded)
        else:
            logits = nn.functional.softmax(logits, dim=1)

        return logits, loss
    
    def get_action(self, observation: int):
        self.history.append(observation)
        logits, loss = self.forward(torch.tensor([self.history[-self.config.block_size:]]))
        logits = logits[:, -1, :]
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
    set_seed(config.seed)
    wandb_init(asdict(config))
    dataset = HistoriesDataset(config.dataset_path)
    model = AD(config)

    # create a Trainer object
    from algorithms.in_context_learning.trainer import Trainer
    
    train_config = Trainer.get_default_config()
    train_config.learning_rate = config.learning_rate # the model we're using is so small that we can go a bit faster
    train_config.max_iters = config.max_iters
    train_config.num_workers = config.num_workers
    train_config.batch_size = config.batch_size
    trainer = Trainer(train_config, model, dataset)
    
    def batch_end_callback(trainer):
        if trainer.iter_num % 100 == 0:
            wandb.log({f"train_loss": trainer.loss.item()})


    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    model_state_dict = model.state_dict()


    save_dict = {
        'train_config': asdict(config),
        'class_name': 'AD',
        'state_dict': model_state_dict
    }

    torch.save(save_dict, f"{config.checkpoints_path}.pt")

    return model

    

if __name__ == "__main__":
    train()