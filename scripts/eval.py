import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import Hallucination
from dataloader import HallucinationDataset, ToTensor
from utils import to_numpy
from main import TrainingParams


def plot_eval_rslts(loc, size, traj, cmd, ori, i_batch, n_samples):
    plt.figure()
    obses = [Ellipse(xy=loc_, width=size_[0], height=size_[1]) for loc_, size_ in zip(loc, size)]
    for obs in obses:
        plt.gca().add_artist(obs)
        obs.set_alpha(0.5)
        obs.set_facecolor(np.random.rand(3))

    r = R.from_quat(ori)
    pos = traj[0]
    plt.plot(pos[:, 0], pos[:, 1])
    cmd_vel, cmd_ang_vel = cmd[0], cmd[1]
    cmd_vel = r.apply([cmd_vel, 0, 0])[:2]
    cmd_ang_vel = r.apply([0, 0, cmd_ang_vel])[-1]
    cmd_ang_vel *= 0.3
    plt.arrow(pos[0, 0], pos[0, 1], cmd_vel[0], cmd_vel[1], width=0.02, label="cmd_vel")
    plt.arrow(pos[0, 0] + cmd[0] / 2.0, pos[0, 1] - cmd_ang_vel / 2.0, 0, cmd_ang_vel, width=0.02, label="cmd_ang_vel")
    plt.legend()
    plt.gca().axis('equal')
    plt.savefig("eval/{}_{}".format(i_batch, n_samples))
    plt.close()


def train(params):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    params.device = device
    Dy = params.Dy
    training_params = params.training_params

    dataset = HallucinationDataset(params, eval=True, transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    model = Hallucination(params, None).to(device)
    if training_params.load_model is not None and os.path.exists(training_params.load_model):
        model.load_state_dict(torch.load(training_params.load_model, map_location=device))
        print("load model finished")

    rslts = {"obs_loc": [],
             "obs_size": [],
             "traj": [],
             "goal": [],
             "cmd": [],
             "ori": []}

    for i_batch, sample_batched in enumerate(dataloader):
        for key, val in sample_batched.items():
            sample_batched[key] = val.to(device)
        # get the inputs; data is a list of [inputs, labels]
        full_traj_tensor = sample_batched["full_traj"]
        reference_pts_tensor = sample_batched["reference_pts"]
        traj_tensor = sample_batched["traj"]
        cmd_tensor = sample_batched["cmd"]
        ori_tensor = sample_batched["ori"]

        reference_pts = to_numpy(reference_pts_tensor)[0]
        full_traj = to_numpy(full_traj_tensor)[0]
        traj = to_numpy(traj_tensor)[0]
        cmd = to_numpy(cmd_tensor)[0]
        ori = to_numpy(ori_tensor)[0]

        n_samples = 0
        while n_samples < params.sample_per_traj:
            if params.model_params.auto_regressive:
                loc_mu, loc_log_var, loc, size_mu, size_log_var, size = model(full_traj_tensor, reference_pts_tensor)
            else:
                if n_samples == 0:
                    loc_mu, loc_log_var, loc, size_mu, size_log_var, size = model(full_traj_tensor, reference_pts_tensor)
                loc = model.encoder.reparameterize(loc_mu, loc_log_var)
                size = torch.log(1 + torch.exp(model.encoder.reparameterize(size_mu, size_log_var)))

            loc = to_numpy(loc)[0]
            size = to_numpy(size)[0]

            full_traj = np.transpose(full_traj, axes=(1, 0))

            diff = reference_pts[3:-3, None] - loc[None]
            diff_norm = np.linalg.norm(diff, axis=-1)
            diff_dir = diff / diff_norm[..., None]
            radius = 1 / np.sqrt((diff_dir ** 2 / size[None] ** 2).sum(axis=-1))
            reference_collision = diff_norm - radius <= params.optimization_params.clearance * 0.0

            if reference_collision.any():
                continue

            rslts["obs_loc"].append(loc)
            rslts["obs_size"].append(size)
            rslts["traj"].append(traj)
            rslts["goal"].append(traj[0, -1])
            rslts["cmd"].append(cmd)
            rslts["ori"].append(ori)

            plot_eval_rslts(loc, size, traj, cmd, ori, i_batch, n_samples)

            n_samples += 1

        print("{}/{}".format(i_batch + 1, len(dataloader)))


if __name__ == "__main__":
    load_dir = "../rslts/2021-02-14-10-53-15"
    model_fname = "model_4700"
    sample_per_traj = 1

    params_fname = os.path.join(load_dir, "params.json")
    params = TrainingParams(params_fname, train=False)

    model_fname = os.path.join(load_dir, "trained_models", model_fname)
    params.training_params.load_model = model_fname
    params.sample_per_traj = sample_per_traj

    train(params)
