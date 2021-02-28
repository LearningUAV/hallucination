import os
import pickle
import shutil
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import Hallucination
from dataloader import HallucinationDataset, ToTensor
from utils import to_numpy, set_axes_equal, draw_ellipsoid, rotanimate
from LfH_main import TrainingParams


def plot_eval_rslts(loc, size, traj, recon_traj, cmd, goal, fname):
    Dy = loc.shape[-1]
    fig = plt.figure()
    if Dy == 2:
        obses = [Ellipse(xy=loc_, width=2 * size_[0], height=2 * size_[1]) for loc_, size_ in zip(loc, size)]
        for obs in obses:
            plt.gca().add_artist(obs)
            obs.set_alpha(0.5)
            obs.set_facecolor(np.random.rand(3))

        pos = traj[0]
        recon_pos = recon_traj[0]
        cmd_vel, cmd_ang_vel = cmd[0], cmd[1]
        cmd_vel = cmd_vel * 0.4
        cmd_ang_vel = cmd_ang_vel * 0.3
        goal = goal * 0.2
        plt.plot(pos[:, 0], pos[:, 1], label="traj")
        plt.plot(recon_pos[:, 0], recon_pos[:, 1], label="recon_traj")
        plt.arrow(0, 0, cmd_vel, 0, width=0.02, label="cmd_vel")
        plt.arrow(0 + cmd[0] / 2.0, 0 - cmd_ang_vel / 2.0, 0, cmd_ang_vel, width=0.02, label="cmd_ang_vel")
        plt.arrow(0, 0, goal[0], goal[1], color="g", width=0.02, label="goal")
        plt.legend()
        plt.gca().axis('equal')
        plt.xlim([-1, 5])
        plt.ylim([-3, 3])
        plt.savefig(fname)
        plt.close()
    else:
        ax = fig.add_subplot(111, projection="3d")
        for loc_, size_ in zip(loc, size):
            draw_ellipsoid(loc_, size_, ax, np.random.rand(3))

        pos = traj[0]
        recon_pos = recon_traj[0]
        goal = goal * 0.2
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label="traj")
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color="red")
        ax.plot(recon_pos[:, 0], recon_pos[:, 1], recon_pos[:, 2], label="recon_traj")
        ax.plot(*list(zip(pos[0], pos[0] + goal)), color="g", label="goal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        set_axes_equal(ax)
        # ax.axis('off')  # remove axes for visual appeal

        angles = np.linspace(0, 360, 20, endpoint=False)  # Take 20 angles between 0 and 360

        # create an animated gif (20ms between frames)
        rotanimate(ax, angles, fname + '.gif', delay=500, elevation=60, width=8, height=6)
        plt.close()


def eval(params):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    params.device = device
    sample_per_traj = params.sample_per_traj
    batch_size = 32 // sample_per_traj
    training_params = params.training_params

    dataset = HallucinationDataset(params, eval=True, transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = Hallucination(params, None).to(device)
    assert os.path.exists(training_params.load_model)
    model.load_state_dict(torch.load(training_params.load_model, map_location=device))
    print("model loaded")

    rslts = {"obs_loc": [],
             "obs_size": [],
             "traj": [],
             "goal": []}
    if params.Dy == 2:
        rslts["cmd"] = []

    eval_dir = params.eval_dir
    eval_plots_dir = os.path.join(eval_dir, "plots")
    os.makedirs(eval_plots_dir, exist_ok=True)
    shutil.copyfile(params.params_fname, os.path.join(eval_dir, "params.json"))
    shutil.copyfile(training_params.load_model, os.path.join(eval_dir, "model"))

    for i_batch, sample_batched in enumerate(dataloader):
        for key, val in sample_batched.items():
            sample_batched[key] = val.to(device)

        full_traj_tensor = sample_batched["full_traj"]
        reference_pts_tensor = sample_batched["reference_pts"]
        traj_tensor = sample_batched["traj"]

        # (batch_size, ...) to (batch_size * sample_per_traj, ...)
        # for example, when sample_per_traj = 3, then (A, B, C) changes to (A, A, A, B, B, B, C, C, C)
        full_traj_tensor = torch.stack([full_traj_tensor] * sample_per_traj, dim=1)
        reference_pts_tensor = torch.stack([reference_pts_tensor] * sample_per_traj, dim=1)
        full_traj_tensor = full_traj_tensor.view(tuple((-1, *full_traj_tensor.size()[2:])))
        reference_pts_tensor = reference_pts_tensor.view(tuple((-1, *reference_pts_tensor.size()[2:])))

        reference_pts = to_numpy(reference_pts_tensor)
        trajs = to_numpy(traj_tensor)
        trajs = np.stack([trajs] * sample_per_traj, axis=1)
        trajs = trajs.reshape(tuple((-1, *trajs.shape[2:])))

        if params.Dy == 2:
            cmd_tensor = sample_batched["cmd"]
            cmds = to_numpy(cmd_tensor)
            cmds = np.stack([cmds] * sample_per_traj, axis=1)
            cmds = cmds.reshape(tuple((-1, *cmds.shape[2:])))
        else:
            cmds = [None] * (batch_size * sample_per_traj)
            goal_tensor = sample_batched["goal"]
            goals = to_numpy(goal_tensor)
            goals = np.stack([goals] * sample_per_traj, axis=1)
            goals = goals.reshape(tuple((-1, *goals.shape[2:])))

        n_samples = np.zeros(batch_size).astype(np.int)
        print_n_samples = -1
        n_invalid_samples = np.zeros(batch_size).astype(np.int)
        while (n_samples < sample_per_traj).any():
            if print_n_samples != n_samples.sum():
                print("{}/{} {}/{}".format(i_batch + 1, len(dataloader), n_samples.sum(), batch_size * sample_per_traj))
                print_n_samples = n_samples.sum()

            # check if stuck for a long time
            need_break = True
            for n_valid, n_invalid in zip(n_samples, n_invalid_samples):
                if n_valid < sample_per_traj and n_invalid < sample_per_traj * 3:
                    need_break = False
                    break
            if need_break:
                print("give up after {} valid and {} invalid".format(n_samples, n_invalid_samples))
                break

            _, _, loc_tensors, _, _, size_tensors = model(full_traj_tensor, reference_pts_tensor)

            locs = to_numpy(loc_tensors)
            sizes = to_numpy(size_tensors)

            def reference_collision_checker(reference_pts_, loc_, size_):
                # check reference collision
                diff = reference_pts_[3:-3, None] - loc_[None]
                diff_norm = np.linalg.norm(diff, axis=-1)
                diff_dir = diff / diff_norm[..., None]
                radius = 1 / np.sqrt((diff_dir ** 2 / size_[None] ** 2).sum(axis=-1))
                reference_collision = (diff_norm - radius <= params.optimization_params.clearance * 0.0).any()
                return reference_collision

            collision_list = Parallel(n_jobs=os.cpu_count())(
                delayed(reference_collision_checker)(reference_pts_, loc_, size_)
                for reference_pts_, loc_, size_ in zip(reference_pts, locs, sizes)
            )

            for j, (col, loc, size, traj) in enumerate(zip(collision_list, locs, sizes, trajs)):
                idx_in_batch = j // sample_per_traj
                if col:
                    n_invalid_samples[idx_in_batch] += 1
                    continue

                if n_samples[idx_in_batch] < sample_per_traj:
                    n_samples[idx_in_batch] += 1

                    rslts["obs_loc"].append(loc)
                    rslts["obs_size"].append(size)
                    rslts["traj"].append(traj)
                    if params.Dy == 2:
                        goal = traj[0, -1].copy()
                        goal /= np.linalg.norm(goal)
                        rslts["goal"].append(goal)
                        rslts["cmd"].append(cmds[j])
                    else:
                        goal = goals[j]
                        rslts["goal"].append(goal)

                    sample_idx = i_batch * batch_size + idx_in_batch
                    if sample_idx % params.plot_freq == 0 and n_samples[idx_in_batch] == 1:
                        fname = os.path.join(eval_plots_dir, str(sample_idx))
                        recon_traj, _ = model.decode(reference_pts_tensor[j:j + 1],
                                                     loc_tensors[j:j + 1], size_tensors[j:j + 1])
                        recon_traj = to_numpy(recon_traj)[0]
                        plot_eval_rslts(loc, size, traj, recon_traj, cmds[j], goal, fname=fname)

        if i_batch % 100 == 0:
            rslts_ = {key: np.array(val) for key, val in rslts.items()}
            with open(os.path.join(eval_dir, "LfH_eval.p"), "wb") as f:
                pickle.dump(rslts_, f)

    rslts = {key: np.array(val) for key, val in rslts.items()}
    with open(os.path.join(eval_dir, "LfH_eval.p"), "wb") as f:
        pickle.dump(rslts, f)


if __name__ == "__main__":
    load_dir = "2021-02-26-02-22-21"
    model_fname = "model_1000"
    sample_per_traj = 8
    plot_freq = 50
    data_fnames = None  # ["2m.p"]

    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    eval_dir = os.path.join(repo_path, "LfH_eval", "{}_{}".format(load_dir, model_fname))
    load_dir = os.path.join(repo_path, "rslts", "LfH_rslts", load_dir)
    model_fname = os.path.join(load_dir, "trained_models", model_fname)
    params_fname = os.path.join(load_dir, "params.json")

    params = TrainingParams(params_fname, train=False)

    params.eval_dir = eval_dir
    params.params_fname = params_fname
    params.training_params.load_model = model_fname
    params.sample_per_traj = sample_per_traj
    params.plot_freq = plot_freq
    if data_fnames is not None:
        params.data_fnames = data_fnames

    eval(params)
