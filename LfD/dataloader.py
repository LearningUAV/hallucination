import os
import torch
import pickle
import numpy as np

from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore")


class Demo_2D_Dataset(Dataset):
    def __init__(self, params, train=True, transform=None):
        """
        Assumed data orginazation
        hallucination/
            LfH_demo/
                # different demos
                2021-01-01-00-00-00_model_666/
                    LfH_demo.p
                ...
        """
        super(Demo_2D_Dataset, self).__init__()

        self.params = params
        self.train = train
        self.transform = transform
        self.train_prop = params.model_params.train_prop

        repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        demo_dir = os.path.join(repo_path, "LfH_demo", params.demo_dir)

        with open(os.path.join(demo_dir, "LfH_demo.p"), "rb") as f:
            d = pickle.load(f)

        self.laser = d["laser"]
        self.goal = d["goal"]
        self.cmd = d["cmd"]

        assert len(self.laser) == len(self.goal) == len(self.cmd)

        train_len = int(np.round(len(self.laser) * self.train_prop))
        if self.train:
            self.laser = self.laser[:train_len]
            self.goal = self.goal[:train_len]
            self.cmd = self.cmd[:train_len]
        else:
            self.laser = self.laser[train_len:]
            self.goal = self.goal[train_len:]
            self.cmd = self.cmd[train_len:]

    def __len__(self):
        return len(self.laser)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = {"laser": self.laser[idx].copy(),
                "goal": self.goal[idx].copy(),
                "cmd": self.cmd[idx].copy()}

        if self.transform:
            data = self.transform(data)

        return data


class Flip(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self,):
        pass

    def __call__(self, data):
        if np.random.rand() > 0.5:
            data["laser"] = np.flip(data["laser"])
            data["goal"][1] *= -1
            data["cmd"][1] *= -1
        return data


class Noise(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, noise_scale=0.05):
        self.noise_scale = noise_scale
        pass

    def __call__(self, data):
        laser_shape = data["laser"].shape
        data["laser"] = np.random.normal(0, self.noise_scale, size=laser_shape)
        return data


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self,):
        pass

    def __call__(self, data):
        new_data = {}
        for key, val in data.items():
            val = val.astype(np.float32)
            new_data[key] = torch.from_numpy(val)
        return new_data
