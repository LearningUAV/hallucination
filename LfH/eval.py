import os
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import Hallucination
from dataloader import HallucinationDataset, ToTensor
from utils import to_numpy
from main import TrainingParams


def plot_eval_rslts(loc, size, traj, cmd, goal, fname):
    plt.figure()
    obses = [Ellipse(xy=loc_, width=2 * size_[0], height=2 * size_[1]) for loc_, size_ in zip(loc, size)]
    for obs in obses:
        plt.gca().add_artist(obs)
        obs.set_alpha(0.5)
        obs.set_facecolor(np.random.rand(3))

    pos = traj[0]
    cmd_vel, cmd_ang_vel = cmd[0], cmd[1]
    cmd_vel = cmd_vel * 0.4
    cmd_ang_vel = cmd_ang_vel * 0.3
    goal = goal * 0.2
    plt.plot(pos[:, 0], pos[:, 1], label="traj")
    plt.arrow(pos[0, 0], pos[0, 1], cmd_vel, 0, width=0.02, label="cmd_vel")
    plt.arrow(pos[0, 0] + cmd[0] / 2.0, pos[0, 1] - cmd_ang_vel / 2.0, 0, cmd_ang_vel, width=0.02, label="cmd_ang_vel")
    plt.arrow(pos[0, 0], pos[0, 1], goal[0], goal[1], color="g", width=0.02, label="goal")
    plt.legend()
    plt.gca().axis('equal')

    x_max, x_min = pos[:, 0].max(), pos[:, 0].min()
    y_max, y_min = pos[:, 1].max(), pos[:, 1].min()
    x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
    x_range, y_range = x_max - x_min, y_max - y_min
    x_min, x_max = x_mid - 1.5 * x_range, x_mid + 1.5 * x_range
    y_min, y_max = y_mid - 1.5 * y_range, y_mid + 1.5 * y_range
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.savefig(fname)
    plt.close()


def eval(params):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    params.device = device
    training_params = params.training_params

    dataset = HallucinationDataset(params, eval=True, transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    model = Hallucination(params, None).to(device)
    assert os.path.exists(training_params.load_model)
    model.load_state_dict(torch.load(training_params.load_model, map_location=device))
    print("load model finished")

    rslts = {"obs_loc": [],
             "obs_size": [],
             "traj": [],
             "goal": [],
             "cmd": []}

    eval_dir = params.eval_dir
    eval_plots_dir = os.path.join(eval_dir, "plots")
    os.makedirs(eval_plots_dir, exist_ok=True)
    shutil.copyfile(params.params_fname, os.path.join(eval_dir, "params.json"))
    shutil.copyfile(training_params.load_model, os.path.join(eval_dir, "model"))

    for i_batch, sample_batched in enumerate(dataloader):
        for key, val in sample_batched.items():
            sample_batched[key] = val.to(device)
        # get the inputs; data is a list of [inputs, labels]
        full_traj_tensor = sample_batched["full_traj"][:1]
        reference_pts_tensor = sample_batched["reference_pts"][:1]
        traj_tensor = sample_batched["traj"][:1]
        cmd_tensor = sample_batched["cmd"][:1]

        reference_pts = to_numpy(reference_pts_tensor)[0]
        traj = to_numpy(traj_tensor)[0]
        cmd = to_numpy(cmd_tensor)[0]

        n_samples = 0
        print_n_samples = -1
        n_invalid_sample = 0
        while n_samples < params.sample_per_traj:
            if print_n_samples != n_samples:
                print("{}/{} {}/{}".format(i_batch + 1, len(dataloader), n_samples + 1, params.sample_per_traj))
                print_n_samples = n_samples

            if params.model_params.auto_regressive:
                loc_mu, loc_log_var, loc, size_mu, size_log_var, size = model(full_traj_tensor, reference_pts_tensor)
            else:
                if n_samples == 0:
                    loc_mu, loc_log_var, loc, size_mu, size_log_var, size = model(full_traj_tensor, reference_pts_tensor)
                loc = model.encoder.reparameterize(loc_mu, loc_log_var)
                size = torch.log(1 + torch.exp(model.encoder.reparameterize(size_mu, size_log_var)))

            loc = to_numpy(loc)[0]
            size = to_numpy(size)[0]

            # check reference collision
            diff = reference_pts[3:-3, None] - loc[None]
            diff_norm = np.linalg.norm(diff, axis=-1)
            diff_dir = diff / diff_norm[..., None]
            radius = 1 / np.sqrt((diff_dir ** 2 / size[None] ** 2).sum(axis=-1))
            reference_collision = diff_norm - radius <= params.optimization_params.clearance * 0.0

            if reference_collision.any():
                n_invalid_sample += 1
                if n_invalid_sample % 10 == 0:
                    print("\tstuck for {} invalid samples".format(n_invalid_sample))
                if n_invalid_sample == 50:
                    break
                continue

            goal = np.array(traj[0, -1])
            goal /= np.linalg.norm(goal)

            rslts["obs_loc"].append(loc)
            rslts["obs_size"].append(size)
            rslts["traj"].append(traj)
            rslts["goal"].append(goal)
            rslts["cmd"].append(cmd)

            if i_batch % params.plot_freq == 0 and n_samples == 0:
                plot_eval_rslts(loc, size, traj, cmd, goal,
                                fname=os.path.join(eval_plots_dir, "{}_{}".format(i_batch, n_samples)))

            n_samples += 1
            n_invalid_sample = 0

        if i_batch % 100 == 0:
            rslts_ = {key: np.array(val) for key, val in rslts.items()}
            with open(os.path.join(eval_dir, "LfH_eval.p"), "wb") as f:
                pickle.dump(rslts_, f)

    rslts = {key: np.array(val) for key, val in rslts.items()}
    with open(os.path.join(eval_dir, "LfH_eval.p"), "wb") as f:
        pickle.dump(rslts, f)


if __name__ == "__main__":
    load_dir = "2021-02-20-23-15-05"
    model_fname = "model_1440"
    sample_per_traj = 1
    plot_freq = 10

    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    eval_dir = os.path.join(repo_path, "LfH_eval", "{}_{}".format(load_dir, model_fname))
    load_dir = os.path.join(repo_path, "rslts", load_dir)
    model_fname = os.path.join(load_dir, "trained_models", model_fname)
    params_fname = os.path.join(load_dir, "params.json")

    params = TrainingParams(params_fname, train=False)

    params.eval_dir = eval_dir
    params.params_fname = params_fname
    params.training_params.load_model = model_fname
    params.sample_per_traj = sample_per_traj
    params.plot_freq = plot_freq

    eval(params)
