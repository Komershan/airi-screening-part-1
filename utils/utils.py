import torch
import random
import numpy as np
import uuid
import wandb

def set_seed(seed: int, device="cpu"):

    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()