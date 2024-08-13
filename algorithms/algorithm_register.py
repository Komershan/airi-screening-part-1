from algorithms.generation_baselines.ucb import UCB
from algorithms.in_context_learning.AD import AD
from algorithms.generation_baselines.random_agent import Random
from algorithms.generation_baselines.a2c import A2C
import importlib
import os

GENERATOR_CLASS = {
    "UCB": UCB,
    "AD": AD,
    "random": Random,
    "A2C": A2C
}

TRAIN_CONFIGS_CLASSES = {
    "UCB": "algorithms.generation_baselines.ucb",
    "AD": "algorithms.in_context_learning.AD",
    "random": "algorithms.generation_baselines.random_agent",
    "A2C": "algorithms.generation_baselines.a2c"
}

def config_from_classname(model_class: str):
    mod = importlib.import_module(TRAIN_CONFIGS_CLASSES[model_class])
    return mod.TrainConfig