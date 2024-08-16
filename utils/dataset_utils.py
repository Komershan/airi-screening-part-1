"""
Here I provide data pipelines (datasets, saving formats) for saving/loading environments and datasets
"""
import h5py
import numpy as np
import json
from typing import Dict, List, Optional
from pydoc import locate
import gymnasium as gym

import environments

import torch

"""
Here I provide universal protocol in order to simplify data saving/loading pipelines
So, how it works? I save mapping dict for environment/dataset, which then used in npify/denpify functions
Where mapping dict consists of (field, dtype) pairs and configures specifically for each environment/dataset in class definition
"""

"""
This function I use when I want to write hdf5 file from mapping function
"""
def write_h5(object_params, fname: str, mapping: dict[str, str], compression="gzip"):
    with h5py.File(fname, "w") as f:
        # I list all mapping fields and save them depends on their type
        for mapping_key, mapping_value in mapping.items():
            assert mapping_key in object_params
            if mapping_value == "array":
                assert isinstance(object_params[mapping_key], np.ndarray)
                f.create_dataset(
                    mapping_key,
                    data=object_params[mapping_key],
                    compression=compression,
                )
            elif mapping_value == "bool":
                f.create_dataset(
                    mapping_key,
                    data=np.array([object_params[mapping_key]], dtype=np.bool_),
                    compression=compression,
                )
            else:
                assert hasattr(np, f"{mapping_value}64")
                f.create_dataset(
                    mapping_key,
                    data=np.array(
                        [object_params[mapping_key]],
                        dtype=getattr(np, f"{mapping_value}64"),
                    ),
                    compression=compression,
                )

'''
This function I use when I want to load saved hdf5 file

I list all keys from mapping and then load parameters from mapping to dict
Which then I can use to initialize classes fields
'''
def load_h5(mapping, fname: str) -> dict:
    data_dict = {}
    with h5py.File(fname, "r") as f:
        for mapping_key, mapping_value in mapping.items():
            if not (mapping_key in f):
                continue
            if mapping_value == "array":
                data_dict[mapping_key] = f[mapping_key][()]
            else:
                data_dict[mapping_key] = locate(mapping_value)(f[mapping_key][0])

    return data_dict

'''
This function I use in order to save list of environments to hdf5 file
'''
def envs_to_h5(envs, fname):
    # In order to save all environments in one file, I need to create one mapping for all environments
    # So, I list environment index, environment field and then create mapping depends on it
    cumulative_mapping, cumulative_params = {}, {}
    for index, env in enumerate(envs):
        mapping_dict, params_dict = env.get_mapping_dict(), env.get_params_dict()
        for key, value in mapping_dict.items():
            cumulative_mapping[f"{key}___{index}"] = value
        for key, value in params_dict.items():
            cumulative_params[f"{key}___{index}"] = value
    write_h5(cumulative_params, fname, cumulative_mapping)

'''
This function I use in order to load list of environments from hdf5 file
'''
def h5_to_envs(fname, env_name):
    index = 0
    mapping = gym.make(env_name).get_mapping_dict()
    exit_flag = False

    loaded_envs = []

    # Here we increase environment index while we can find environment fields
    while not exit_flag:
        locale_mapping = {f"{key}___{index}": value for key, value in mapping.items()}
        data_dict = load_h5(locale_mapping, fname)
        if len(data_dict) == 0:
            exit_flag = True
            break
        env_dict = {key.split("___")[0]: value for key, value in data_dict.items()}
        loaded_env = gym.make(env_name)
        loaded_env.load_from_dict(env_dict)
        loaded_envs.append(loaded_env)
        index += 1

    return loaded_envs

# This class is used for storing histories
# Which we collect to train Algorithm Distillation
class HistoriesDataset:
    def __init__(
            self,
            data_path=None,
            resize_history=None,
            vocab_size=None,
            compress_context_size = 0,
            compress_context_count = 0,
            episode_size = 0
            ):
    # compress_context_size, compress_context_count are used in experiments with compression
        self.mapping_dict = {
            "histories": "array",
            "resize_history": "int",
            "vocab_size": "int",
            "compress_context_size": "int",
            "compress_context_count": "int",
            "episode_size": "int"
        }

        self.compress_context_size = compress_context_size
        self.compress_context_count = compress_context_count
        self.episode_size = episode_size

        self.resize_history = resize_history
        self.vocab_size = vocab_size
        self._random_slices = []
        if data_path != None:
            self.load(data_path)
        else:
            self.histories = None
            self._num_samples = 0

    def __len__(self) -> int:
        return self.histories.shape[0] // 3 - self.resize_history + 1

    def append_data(self, history: np.array):
        if self.histories is None:
            self.histories = np.array(history)
        else:
            self.histories = np.concatenate((self.histories, history), axis=0)
        self._num_samples += 1

    def _reset_data(self) -> dict:
        self.histories = None
        self._num_samples = 0
        return self.histories

    def write(self, fname, compression="gzip"):
        object_params = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("__") and not callable(key)
        }
        write_h5(
            object_params, fname, mapping=self.mapping_dict, compression=compression
        )

    def load(self, fname):
        arguments_dict = load_h5(self.mapping_dict, fname)
        self.__dict__.update(arguments_dict)
        self._num_samples = self.histories.shape[0]

    def __getitem__(self, index):
        assert 0 <= index and index < self.__len__()

        # Here we differentiate getitem for compress experiment

        if not(self.compress_context_count == 0):

            result = []

            context_index = index // self.episode_size

            # In this loop we add compressed histories among past episodes
            for i in range(max(context_index - self.compress_context_count, 0), context_index):
                result += self.histories[i * self.compress_context_size * 3: (i + 1) * self.compress_context_size * 3].tolist()

            # Then we add history from curr episode
            result += self.histories[context_index * self.episode_size * 3: index + 1].tolist()

            return torch.tensor(
                result,
                dtype=torch.long,
            )

        else: 
            if self.resize_history is None:
                return torch.tensor(self.histories.tolist(), dtype=torch.long)
            else:
                return torch.tensor(
                    self.histories[index * 3 : (index + self.resize_history) * 3].tolist(),
                    dtype=torch.long,
                )
