import os
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from joblib import Parallel, delayed

import torch

import sys
sys.path.append(os.path.abspath("../LfH"))
import utils


class Params:
    LfH_dir = "2021-02-27-02-06-00_model_3600"
    n_render_per_sample = 1
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # for additional obstable generation
    n_pts_to_consider = 5
    n_additional_obs = 5
    loc_radius = 10.0
    loc_span = 70
    size_min, size_max = 0.1, 0.6
    vel_time = 1.0
    vel_span = 60

    # for observation rendering
    batch_size = 128
    image_scale = 2.0
    image_h = 640
    image_v = 480
    max_depth = 1000
    fh = 387.229248046875       # from ego-planner
    fv = 387.229248046875
    ch = 321.04638671875
    cv = 243.44969177246094
    plot_freq = 200
    ego_depth = True

    # post process params
    batch_size = batch_size // n_render_per_sample
    loc_span = loc_span * np.pi / 180
    vel_span = vel_span * np.pi / 180

    image_h = int(image_h / image_scale)
    image_v = int(image_v / image_scale)
    fh /= image_scale
    fv /= image_scale
    ch /= image_scale
    cv /= image_scale
    if ego_depth:
        max_depth = max(max_depth, 500.0)


def angle_diff(angle1, angle2):
    diff = np.abs(angle1 - angle2)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return diff


def find_raycast(pos, dir, loc, size, params, proj_to_x=False):
    """

    :param pos: (Bs1, ..., BsN, 3)
    :param dir: (Bs1, ..., BsN, 3)
    :param loc: (Bs1, ..., BsN, 3)
    :param size: (Bs1, ..., BsN, 3)
    :return: (Bs1, ..., BsN)
    """
    A = (dir ** 2 / size ** 2).sum(axis=-1)
    B = (2 * dir * (pos - loc) / size ** 2).sum(axis=-1)
    C = ((pos - loc) ** 2 / size ** 2).sum(axis=-1) - 1
    delta = B ** 2 - 4 * A * C
    ray = np.full(delta.shape, params.max_depth)
    delta_mask = delta >= 0
    delta = np.maximum(0, delta)
    l1 = (-B - np.sqrt(delta)) / (2 * A)
    l2 = (-B + np.sqrt(delta)) / (2 * A)
    # notice l1 always smaller than l2
    zero_ray_mask = (l1 < 0) & (l2 > 0)
    gt_zero_ray_mask = l1 >= 0
    ray[delta_mask & zero_ray_mask] = 0
    if proj_to_x:
        l1 = l1 * dir[..., 0]
    ray[delta_mask & gt_zero_ray_mask] = l1[delta_mask & gt_zero_ray_mask]
    return ray


def find_raycast_torch(pos, dir, loc, size, params, proj_to_x=False):
    """

    :param pos: (Bs1, ..., BsN, 3)
    :param dir: (Bs1, ..., BsN, 3)
    :param loc: (Bs1, ..., BsN, 3)
    :param size: (Bs1, ..., BsN, 3)
    :return: (Bs1, ..., BsN)
    """
    pos = torch.from_numpy(pos).to(params.device)
    dir = torch.from_numpy(dir).to(params.device)
    loc = torch.from_numpy(loc).to(params.device)
    size = torch.from_numpy(size).to(params.device)

    with torch.no_grad():
        A = (dir ** 2 / size ** 2).sum(dim=-1)
        B = (2 * dir * (pos - loc) / size ** 2).sum(dim=-1)
        C = ((pos - loc) ** 2 / size ** 2).sum(dim=-1) - 1
        delta = B ** 2 - 4 * A * C                              # (Bs1, ..., BsN)
        ray = torch.full(delta.shape, params.max_depth, dtype=torch.float32).to(params.device)
        delta_mask = delta >= 0
        delta = torch.clamp(delta, min=0)
        l1 = (-B - torch.sqrt(delta)) / (2 * A)
        l2 = (-B + torch.sqrt(delta)) / (2 * A)
        # notice l1 always smaller than l2
        zero_ray_mask = (l1 < 0) & (l2 > 0)
        gt_zero_ray_mask = l1 >= 0
        ray[delta_mask & zero_ray_mask] = 0
        if proj_to_x:
            l1 = l1 * dir[..., 0]
        ray[delta_mask & gt_zero_ray_mask] = l1[delta_mask & gt_zero_ray_mask]
    return ray.cpu().numpy()


def is_colliding_w_traj(poses, vels, loc, size, params):
    """
    :param poses: (N, 3) trajectory positions
    :param vels: (N, 3) trajectory velocities
    :param loc: (M, 3) obs location
    :param size: (M, 3) obs size
    :param vel_time, vel_span: colliding if obs within radius = vel * vel_time and angle in vel_span
    :return: (M,) boolean
    """

    vel_time = params.vel_time
    vel_span = params.vel_span
    vel_norm = np.linalg.norm(vels, axis=-1)
    clearance = vel_norm * vel_time

    M = loc.shape[0]
    N = poses.shape[0]

    obs_diff = loc[:, None] - poses[None]                                               # (M, N, 3)
    obs_dir = obs_diff / np.linalg.norm(obs_diff, axis=-1, keepdims=True)               # (M, N, 3)

    vel_dir = vels / vel_norm[:, None]                                                  # (N, 3)
    vel_dir = np.array([vel_dir] * M)                                                   # (M, N, 3)
    cos_ang_diff = (obs_dir * vel_dir).sum(axis=-1, keepdims=True)                      # (M, N, 1)
    normal = np.cross(vel_dir, obs_dir)                                                 # (M, N, 3)
    r = R.from_rotvec(normal.reshape((-1, 3)) * vel_span / 2)
    rotated_dir = r.apply(vel_dir.reshape((-1, 3))).reshape((M, N, 3))                  # (M, N, 3)
    ray_dir = np.where(cos_ang_diff >= np.cos(vel_span / 2), obs_dir, rotated_dir)      # (M, N, 3)

    raycast = find_raycast(poses[None], ray_dir, loc[:, None], size[:, None], params)   # (M, N)
    is_col = (raycast <= clearance).any(axis=-1)                                        # (M)
    return is_col


def generate_additional_obs(traj, params):
    pos, vel = traj
    pt_indexes = np.round(np.linspace(0, len(pos) - 1, params.n_pts_to_consider)).astype(np.int)
    pos, vel = pos[pt_indexes], vel[pt_indexes]

    locs, sizes = [], []
    batch_size = 2 * params.n_additional_obs

    n_valid_obs = 0
    while True:
        rad = np.random.uniform(0, params.loc_radius, batch_size).astype(np.float32)
        ang1 = np.random.uniform(-params.loc_span / 2, params.loc_span / 2, batch_size).astype(np.float32)
        ang2 = np.random.uniform(0, np.pi, batch_size).astype(np.float32)
        loc = np.stack([np.cos(ang1), np.sin(ang1) * np.cos(ang2), np.sin(ang1) * np.cos(ang2)], axis=-1) * rad[:, None]
        size = np.random.uniform(params.size_min, params.size_max, (batch_size, 3)).astype(np.float32)

        is_col =  is_colliding_w_traj(pos, vel, loc, size, params)
        for is_col_, loc_, size_ in zip(is_col, loc, size):
            if is_col_:
                continue
            n_valid_obs += 1
            locs.append(loc_)
            sizes.append(size_)
            if n_valid_obs == params.n_additional_obs:
                break
        if n_valid_obs == params.n_additional_obs:
            break

    return np.array(locs), np.array(sizes)


def render_depth(obs_loc, obs_size, add_obs_loc, add_obs_size, params):
    """

    :param obs_loc: (Bs, N, 3)
    :param obs_size: (Bs, N, 3)
    :param add_obs_loc: (Bs, M, 3)
    :param add_obs_size: (Bs, M, 3)
    :param params:
    :return:
    """
    image_h = params.image_h
    image_v = params.image_v
    image_scale = params.image_scale
    fh = params.fh
    fv = params.fv
    ch = params.ch
    cv = params.cv

    if params.n_additional_obs > 0:
        locs = np.concatenate([obs_loc, add_obs_loc], axis=1)
        sizes = np.concatenate([obs_size, add_obs_size], axis=1)
    else:
        locs, sizes = obs_loc, obs_size

    batch_size, n_obs, _ = locs.shape

    pos = np.zeros((image_v, image_h, 3)).astype(np.float32)
    h_ratio = (np.flip(np.arange(image_h)) - 0.5 / image_scale - ch) / fh
    v_ratio = (np.flip(np.arange(image_v)) - 0.25 / image_scale - cv) / fv
    h_ratio, v_ratio = h_ratio.astype(np.float32), v_ratio.astype(np.float32)
    h_grid, v_grid = np.meshgrid(h_ratio, v_ratio)
    dir = np.stack([np.ones((image_v, image_h), dtype=np.float32), h_grid, v_grid], axis=-1)
    dir /= np.linalg.norm(dir, axis=-1, keepdims=True)

    pos = pos[None, :, :, None]             # (batch_size, image_v, image_h, 1, 3)
    dir = dir[None, :, :, None]             # (batch_size, image_v, image_h, 1, 3)
    locs = locs[:, None, None]              # (1, image_v, image_h, n_obs, 3)
    sizes = sizes[:, None, None]            # (1, image_v, image_h, n_obs, 3)
    depth = find_raycast_torch(pos, dir, locs, sizes, params, proj_to_x=params.ego_depth)
    depth = depth.min(axis=-1)
    if params.ego_depth:
        depth = np.where(depth < 500.0, depth, 0)
    return depth


def plot_render_rslts(obs_loc, obs_size, traj, goal, lin_vel, add_obs_loc, add_obs_size, depth, params, fig_name):
    image_dir = os.path.join(params.demo_dir, "plots")
    os.makedirs(image_dir, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.imshow(depth)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, fig_name + "_depth"))
    plt.close()

    fig = plt.figure(figsize=(5, 5))

    pos = traj[0]
    goal = goal * 1.0
    lin_vel = lin_vel * 1.0
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label="traj")
    ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color="red")
    ax.plot(*list(zip(pos[0], pos[0] + goal)), color="orange", label="goal")
    ax.plot(*list(zip(pos[0], pos[0] + lin_vel)), color="red", label="goal")

    for loc_, size_ in zip(obs_loc, obs_size):
        utils.draw_ellipsoid(loc_, size_, ax, color="red")
    if params.n_additional_obs:
        for loc_, size_ in zip(add_obs_loc, add_obs_size):
            utils.draw_ellipsoid(loc_, size_, ax, color="violet")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    utils.set_axes_equal(ax)
    angles = np.linspace(0, 360, 20, endpoint=False)  # Take 20 angles between 0 and 360

    plt.savefig(os.path.join(image_dir, fig_name + "_traj"))
    utils.rotanimate(ax, angles, os.path.join(image_dir, fig_name + ".gif"), delay=50, elevation=45, width=8, height=6)
    plt.close()


def repeat(input, repeat_time):
    """
    :param array/tensor: (A, B, C)
    :return: (A, A, A, B, B, B, C, C, C) if repeat_time == 3
    """
    array = np.stack([input] * repeat_time, axis=1)
    array = array.reshape(tuple((-1, *array.shape[2:])))
    return array


if __name__ == "__main__":
    params = Params()

    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    params.demo_dir = demo_dir = os.path.join(repo_path, "LfH_demo", params.LfH_dir)
    LfH_dir = os.path.join(repo_path, "LfH_eval", params.LfH_dir)

    os.makedirs(demo_dir, exist_ok=True)
    shutil.copyfile(os.path.join(LfH_dir, "params.json"), os.path.join(demo_dir, "params.json"))
    shutil.copyfile(os.path.join(LfH_dir, "model"), os.path.join(demo_dir, "model"))

    with open(os.path.join(LfH_dir, "LfH_eval.p"), "rb") as f:
        data = pickle.load(f)

    obs_locs = data["obs_loc"]
    obs_sizes = data["obs_size"]
    trajs = data["traj"]
    goals = data["goal"]
    lin_vels = data["lin_vel"]
    ang_vels = data["ang_vel"]

    demo = {"depth": [],
            "goal": [],
            "traj": [],
            "lin_vel": [],
            "ang_vel": []}

    print("Total samples: {}".format(len(goals)))
    plot_freq = params.plot_freq
    batch_size = params.batch_size
    n_render_per_sample = params.n_render_per_sample
    n_render_in_batch = batch_size * n_render_per_sample
    for i in range(0, len(goals), batch_size):
        print("{}/{}".format(i + 1, len(goals)))
        obs_loc_batch = repeat(obs_locs[i:i + batch_size], n_render_per_sample)
        obs_size_batch = repeat(obs_sizes[i:i + batch_size], n_render_per_sample)
        traj_batch = repeat(trajs[i:i + batch_size], n_render_per_sample)
        goal_batch = repeat(goals[i:i + batch_size], n_render_per_sample)
        lin_vel_batch = repeat(lin_vels[i:i + batch_size], n_render_per_sample)
        ang_vel_batch = repeat(ang_vels[i:i + batch_size], n_render_per_sample)

        if params.n_additional_obs > 0:
            add_obs_list = Parallel(n_jobs=os.cpu_count())(delayed(generate_additional_obs)(traj, params)
                                                           for traj in traj_batch)
            add_obs_loc_batch = np.array([ele[0] for ele in add_obs_list])
            add_obs_size_batch = np.array([ele[1] for ele in add_obs_list])
        else:
            add_obs_loc_batch = add_obs_size_batch = None

        depth_batch = render_depth(obs_loc_batch, obs_size_batch, add_obs_loc_batch, add_obs_size_batch, params)

        demo["depth"].extend(depth_batch)
        demo["goal"].extend(goal_batch)
        demo["traj"].extend(traj_batch)
        demo["lin_vel"].extend(lin_vel_batch)
        demo["ang_vel"].extend(ang_vel_batch)
        if i % plot_freq == 0:
            obs_loc = obs_loc_batch[0]
            obs_size = obs_size_batch[0]
            traj = traj_batch[0]
            goal = goal_batch[0]
            lin_vel = lin_vel_batch[0]
            add_obs_loc = add_obs_loc_batch[0] if add_obs_loc_batch is not None else None
            add_obs_size = add_obs_size_batch[0] if add_obs_loc_batch is not None else None
            depth = depth_batch[0]
            plot_render_rslts(obs_loc, obs_size, traj, goal, lin_vel, add_obs_loc, add_obs_size, depth, params,
                              fig_name=str(i))

        if i % 200 == 0:
            demo_ = {k: np.array(v).astype(np.float32) for k, v in demo.items()}
            with open(os.path.join(demo_dir, "LfH_demo.p"), "wb") as f:
                pickle.dump(demo_, f)

    demo = {k: np.array(v).astype(np.float32) for k, v in demo.items()}
    with open(os.path.join(demo_dir, "LfH_demo.p"), "wb") as f:
        pickle.dump(demo, f)
