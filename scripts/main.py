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

    lambda_kl_final = model_params.lambda_kl
    lambda_repulsion_final = model_params.lambda_repulsion
    lambda_annealing_steps = model_params.lambda_annealing_steps
    for epoch in range(training_params.epochs):
        losses = []
        recon_losses, repulsive_losses, kl_losses, reference_repulsion_losses = [], [], [], []
        model.train(training=True)
        params.model_params.lambda_kl = lambda_kl_final * (epoch + 1) / lambda_annealing_steps
        params.model_params.lambda_repulsion = lambda_repulsion_final * (epoch + 1) / lambda_annealing_steps
        for i_batch, sample_batched in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            full_traj = sample_batched["full_traj"]
            reference_pts = sample_batched["reference_pts"]
            traj = sample_batched["traj"]

            optimizer.zero_grad()
            recon_traj, recon_control_points, loc_mu, loc_log_var, loc, size_mu, size_log_var, size = \
                model(full_traj, reference_pts, decode=True)
            loss, (recon_loss, repulsive_loss, kl_loss, reference_repulsion_loss) = \
                model.loss(traj, recon_traj, reference_pts,
                           loc_mu, loc_log_var, loc, size_mu, size_log_var, size)
            loss.backward()
            print(recon_loss.item(), repulsive_loss.item(), kl_loss.item(), reference_repulsion_loss.item())
            optimizer.step()
            losses.append(loss.item())

            recon_losses.append(recon_loss.item())
            repulsive_losses.append(repulsive_loss.item())
            kl_losses.append(kl_loss.item())
            reference_repulsion_losses.append(reference_repulsion_loss.item())

            if (i_batch + epoch * len(dataloader)) % 10 == 0:
                plot_opt(writer, reference_pts, recon_control_points,
                         loc, size, i_batch + epoch * len(dataloader))
                plot_obs_dist(writer, reference_pts, recon_control_points,
                              loc_mu, loc_log_var, loc, size_mu, size_log_var, size, i_batch + epoch * len(dataloader))
            print("{}/{}, {}/{}".format(epoch + 1, training_params.epochs, i_batch + 1, len(dataloader)))

        if writer is not None:
            writer.add_scalar("train/loss", np.mean(losses), epoch)
            writer.add_scalar("train/recon_loss", np.mean(recon_losses), epoch)
            writer.add_scalar("train/repulsive_loss", np.mean(repulsive_losses), epoch)
            writer.add_scalar("train/kl_loss", np.mean(kl_losses), epoch)
            writer.add_scalar("train/opt_remain_loss", np.mean(reference_repulsion_losses), epoch)

            model.train(training=False)
            plot_opt(writer, reference_pts, recon_control_points, loc, size, epoch)

        if (epoch + 1) % training_params.saving_freq == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, "model_{0}".format(epoch + 1)),
                       pickle_protocol=2, _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    params = TrainingParams()
    train(params)
