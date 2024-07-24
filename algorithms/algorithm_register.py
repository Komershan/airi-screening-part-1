from algorithms.generation_baselines.ucb import UCB
from algorithms.in_context_learning.AD import AD
import os

GENERATOR_CLASS = {
    "ucb": UCB,
    "AD": AD
}

class AlgorithmRegister:
    def __init__(self, checkpoints: str):
        os.listdir()