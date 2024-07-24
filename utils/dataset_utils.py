import h5py
import numpy as np
import json
from typing import Dict, List, Optional

import torch


def npify(data_dict: Dict[str, List]) -> dict:
    np_data = {}
    for k in data_dict:
        if k == "terminals":
            dtype = np.bool_
        else:
            dtype = np.int32
        data = np.array(data_dict[k], dtype=dtype)
        np_data[k] = data
    return np_data


def denpify(np_data: Dict[str, np.ndarray]) -> dict:
    data_dict = {}
    for k, v in np_data.items():
        data_dict[k] = v.tolist()
    return data_dict


def write_h5(data_dict: dict, fname: str, vocab_size: int, resize_history=None, compression="gzip"):
    with h5py.File(fname, "w") as f:
        for key, value in data_dict.items():
            f.create_dataset(f"{key}", data=value, compression=compression)

        if resize_history != None:
            f.create_dataset(f"resize_history", data=[resize_history], compression=compression)  

        print(vocab_size)

        f.create_dataset(f"vocab_size", data=[vocab_size], compression=compression)  
        


def load_h5(fname: str) -> dict:
    data_dict = {}
    resize_history = None
    vocab_size = 0
    with h5py.File(fname, "r") as f:
        for key in f.keys():
            if key.isnumeric():
                data_dict[int(key)] = f[key][()]
            else:
                if key == "resize_history":
                    resize_history = int(f[key][0])
                else:
                    vocab_size = int(f[key][0])

    return data_dict, vocab_size, resize_history


class HistoriesDataset:
    def __init__(self, data_path = None, resize_history = None, vocab_size = None):
        self.resize_history = resize_history
        self.vocab_size = vocab_size
        if data_path != None:
            self.load(data_path)
        else:
            self.histories_dict = dict()
            self._num_samples = 0

    def __len__(self) -> int:
        return self._num_samples

    def append_data(self, history: List[int]):
        self.histories_dict[self._num_samples] = np.array(history)
        self._num_samples += 1

    def _reset_data(self) -> dict:
        self.histories_dict = dict()
        self._num_samples = 0
        return self.histories_dict

    def write(self, fname, max_size: Optional[int] = None, compression="gzip"):
        write_h5(self.histories_dict, fname, vocab_size=self.vocab_size, resize_history=self.resize_history, compression=compression)

    def load(self, fname):
        to_denpify, self.vocab_size, self.resize_history = load_h5(fname)
        self.histories_dict = denpify(to_denpify)
        self._num_samples = len(self.histories_dict)

    def __getitem__(self, index):

        if self.resize_history is None:
            return torch.tensor(self.histories_dict[index], dtype=torch.long)
        else:
            assert self.resize_history * 3 <= len(self.histories_dict[index])
            random_slice = np.random.randint(low=0, high=len(self.histories_dict[index]) // 3 - self.resize_history, dtype='int64')
            return torch.tensor(self.histories_dict[index][random_slice * 3:(random_slice + self.resize_history) * 3], dtype=torch.long)
