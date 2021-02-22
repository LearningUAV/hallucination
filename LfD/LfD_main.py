import os
import sys
import time
import json
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from dataloader import Demo_2D_Dataset, Flip, ToTensor


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class TrainingParams:
    def __init__(self, training_params_fname="params.json", train=True):
        config = json.load(open(training_params_fname))
        for k, v in config.items():
            self.__dict__[k] = v
        self.__dict__ = self._clean_dict(self.__dict__)

        if self.training_params.load_model is not None:
            self.training_params.load_model = os.path.join("..", "interesting_models", self.training_params.load_model)

        if train:
            repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            self.rslts_dir = os.path.join(repo_path, "LfD_2D_rslts", time.strftime("%Y-%m-%d-%H-%M-%S"))
            os.makedirs(self.rslts_dir)
            shutil.copyfile(training_params_fname, os.path.join(self.rslts_dir, "params.json"))

    def _clean_dict(self, _dict):
        for k, v in _dict.items():
            if v == "":  # encode empty string as None
                v = None
            if isinstance(v, dict):
                v = AttrDict(self._clean_dict(v))
            _dict[k] = v
        return _dict


class LfD_2D_model(nn.Module):
    def __init__(self, params):
        super(LfD_2D_model, self).__init__()
        self.params = params
        self.model_params = model_params = params.model_params

        self.fcs = nn.ModuleList()
        prev_layer_size = model_params.obs_size
        for layer_size in model_params.layer_sizes:
            self.fcs.append(nn.Linear(prev_layer_size, layer_size))
            prev_layer_size = layer_size
        self.action_fc = nn.Linear(prev_layer_size, 2)

    def forward(self, laser, goal):
        h = torch.cat([laser, goal], dim=-1)
        for fc in self.fcs:
            h = F.leaky_relu(fc(h))
        action = self.action_fc(h)
        return action


def train(params):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    params.device = device
    training_params = params.training_params

    writer = SummaryWriter(os.path.join(params.rslts_dir, "tensorboard"))
    dataset = Demo_2D_Dataset(params, transform=transforms.Compose([Flip(), ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=training_params.batch_size, shuffle=True, num_workers=4)

    model = LfD_2D_model(params).to(device)
    if training_params.load_model is not None and os.path.exists(training_params.load_model):
        model.load_state_dict(torch.load(training_params.load_model, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=training_params.lr)

    # model saving
    model_dir = os.path.join(params.rslts_dir, "trained_models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch in range(training_params.epochs):
        losses = []
        model.train(mode=True)
        for i_batch, sample_batched in enumerate(dataloader):
            for key, val in sample_batched.items():
                sample_batched[key] = val.to(device)
            # get the inputs; data is a list of [inputs, labels]
            laser = sample_batched["laser"]
            goal = sample_batched["goal"]
            cmd = sample_batched["cmd"]

            optimizer.zero_grad()
            cmd_pred = model(laser, goal)
            loss = torch.mean(torch.sum((cmd - cmd_pred) ** 2, dim=1))
            loss.backward()
            print(loss.item())
            optimizer.step()

            losses.append(loss.item())

            print("{}/{}, {}/{}".format(epoch + 1, training_params.epochs,
                                        i_batch + 1, training_params.batch_per_epoch))
            if i_batch + 1 == training_params.batch_per_epoch:
                break

        if writer is not None:
            writer.add_scalar("loss", np.mean(losses), epoch)

        if (epoch + 1) % training_params.saving_freq == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, "model_{}".format(epoch + 1)),
                       pickle_protocol=2, _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    params = TrainingParams()
    train(params)
