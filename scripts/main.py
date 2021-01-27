import os
import sys
import time
import json
import shutil
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import Hallucination
from dataloader import HallucinationDataset, ToTensor
from utils import plot_ode_opt, plot_opt, plot_obs_dist


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class TrainingParams:
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        training_params_fname = os.path.join(dir_path, "params.json")
        config = json.load(open(training_params_fname))
        for k, v in config.items():
            self.__dict__[k] = v
        self.__dict__ = self._clean_dict(self.__dict__)

        if self.training_params.load_model is not None:
            self.training_params.load_model = os.path.join("..", "interesting_models", self.training_params.load_model)

        if self.train:
            self.rslts_dir = os.path.join("..", "rslts", "{}".format(time.strftime("%Y-%m-%d-%H-%M-%S")))
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


def train(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params.device = device
    training_params = params.training_params
    model_params = params.model_params

    dataset = HallucinationDataset(params, transform=transforms.Compose([ToTensor(device)]))
    dataloader = DataLoader(dataset, batch_size=training_params.batch_size, shuffle=True, num_workers=4)

    model = Hallucination(params).to(device)
    if training_params.load_model is not None and os.path.exists(training_params.load_model):
        model.load_state_dict(torch.load(training_params.load_model, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=training_params.lr)

    # py2 can only use torch 1.0 which doesn"t support tensorboard...
    writer = None
    if sys.version_info[0] == 3:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(params.rslts_dir, "tensorboard"))

    # model saving
    model_dir = os.path.join(params.rslts_dir, "trained_models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    lambda_loc_kl_final = model_params.lambda_loc_kl
    lambda_size_kl_final = model_params.lambda_size_kl
    lambda_mutual_repulsion_final = model_params.lambda_mutual_repulsion
    lambda_reference_repulsion_final = model_params.lambda_reference_repulsion
    lambda_annealing_steps = model_params.lambda_annealing_steps
    for epoch in range(training_params.epochs):
        loss_details = []
        model.train(training=True)
        annealing_coef = (epoch + 1.) / lambda_annealing_steps
        model_params.lambda_loc_kl = lambda_loc_kl_final * annealing_coef
        model_params.lambda_size_kl = lambda_size_kl_final * annealing_coef
        model_params.lambda_mutual_repulsion = lambda_mutual_repulsion_final * annealing_coef
        model_params.lambda_reference_repulsion = lambda_reference_repulsion_final * annealing_coef
        for i_batch, sample_batched in enumerate(dataloader):
            for key, val in sample_batched.items():
                sample_batched[key] = val.to(device)
            # get the inputs; data is a list of [inputs, labels]
            full_traj = sample_batched["full_traj"]
            reference_pts = sample_batched["reference_pts"]
            traj = sample_batched["traj"]

            optimizer.zero_grad()
            recon_traj, recon_control_points, loc_mu, loc_log_var, loc, size_mu, size_log_var, size = \
                model(full_traj, reference_pts, decode=True)
            loss, loss_detail = model.loss(full_traj, traj, recon_traj, reference_pts,
                                           loc_mu, loc_log_var, loc, size_mu, size_log_var, size)
            loss.backward()
            print(loss_detail)
            optimizer.step()

            loss_details.append(loss_detail)

            num_batch = i_batch + epoch * len(dataloader)
            if num_batch % 10 == 0:
                plot_opt(writer, reference_pts, recon_control_points, loc, size, num_batch)
                plot_obs_dist(writer, params, full_traj, loc_mu, loc_log_var, size_mu, size_log_var, num_batch)
            print("{}/{}, {}/{}".format(epoch + 1, training_params.epochs, i_batch + 1, len(dataloader)))

        if writer is not None:
            # list of dict to dict of list
            loss_details = {k: [dic[k] for dic in loss_details] for k in loss_details[0]}
            for k, v in loss_details.items():
                writer.add_scalar("train/{}".format(k), np.mean(v), epoch)

        if (epoch + 1) % training_params.saving_freq == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, "model_{}".format(epoch + 1)),
                       pickle_protocol=2, _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    params = TrainingParams()
    train(params)
