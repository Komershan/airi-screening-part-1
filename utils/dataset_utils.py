import h5py
import numpy as np
import json
from typing import Dict, List, Optional


def npify(data_dict: Dict[str, List], max_size: Optional[int] = None) -> dict:
    np_data = {}
    for k in data_dict:
        if k == "terminals":
            dtype = np.bool_
        else:
            dtype = np.int32
        data = np.array(data_dict[k], dtype=dtype)
        if max_size is not None:
            data = data[:max_size]
        np_data[k] = data
    return np_data


def denpify(np_data: Dict[str, np.ndarray]) -> dict:
    data_dict = {}
    for k, v in np_data.items():
        data_dict[k] = v.tolist()
    return data_dict


def write_h5(data_dict: dict, fname: str, max_size: Optional[int] = None, compression="gzip"):
    with h5py.File(fname, "w") as f:
        for key, value in data_dict.items():
            if max_size != None and len(value) > 3 * max_size:
                value = value[:3 * max_size]
            f.create_dataset(f"{key}", data=value, compression=compression)


def load_h5(fname: str) -> dict:
    data_dict = {}
    with h5py.File(fname, "r") as f:
        for key in f.keys():
            data_dict[int(key)] = f[key][()]
    return data_dict


class Dataset:
    def __init__(self):
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
        write_h5(self.histories_dict, fname, max_size=max_size, compression=compression)

    def load(self, fname):
        self.data = denpify(load_h5(fname))

    def __getitem__(self, index) -> np.array:
        return self.data[index]