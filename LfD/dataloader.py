import os
import torch
import pickle
import numpy as np

from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore")


class Demo_2D_Dataset(Dataset):
    def __init__(self, params, transform=None):
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
        self.transform = transform

        repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        demo_dir = os.path.join(repo_path, "LfH_demo", params.demo_dir)

        with open(os.path.join(demo_dir, "LfH_demo.p"), "rb") as f:
            d = pickle.load(f)

        self.laser = d["laser"]
        self.goal = d["goal"]
        self.cmd = d["cmd"]

        assert len(self.laser) == len(self.goal) == len(self.cmd)

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
