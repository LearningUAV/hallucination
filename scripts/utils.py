import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def plot_ode_opt(writer, model, reference_pts, recon_control_points, loc, size, epoch):
    batch_size = reference_pts.size(0)
    if batch_size <= 3:
        idx = 0
    else:
        idx = np.random.randint(batch_size - 3)

    reference_pts = reference_pts[idx:idx + 3]
    recon_control_points = recon_control_points[:, idx:idx + 3]
    loc = loc[idx:idx + 3]
    size = size[idx:idx + 3]

    reference_pts, loc, size, recon_control_points, losses = model.test(reference_pts, recon_control_points, loc, size)

    loss_fig, loss_axes = plt.subplots(1, 3, figsize=(18, 4))
    opt_fig, opt_axes = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):

        opt_params = model.params.optimization_params
        loss_axes[i].plot(np.linspace(0, opt_params.ode_t_end, opt_params.ode_num_timestamps), losses[i])
        loss_axes[i].set_xlabel("ODE t")
        loss_axes[i].set_ylabel("loss")

        opt_axes[i].plot(reference_pts[i, :, 0], reference_pts[i, :, 1], label="reference")
        opt_axes[i].scatter(reference_pts[i, :, 0], reference_pts[i, :, 1])

        obses = [Ellipse(xy=loc_, width=size_[0], height=size_[1]) for loc_, size_ in zip(loc[i], size[i])]
        for obs in obses:
            opt_axes[i].add_artist(obs)
            obs.set_alpha(0.5)
            obs.set_facecolor(np.random.rand(3))

        ode_num_timestamps = recon_control_points.shape[0]
        for j in range(ode_num_timestamps):
            opt_axes[i].plot(recon_control_points[j, i, :, 0], recon_control_points[j, i, :, 1], label="opt_{}".format(j))
            opt_axes[i].scatter(recon_control_points[j, i, :, 0], recon_control_points[j, i, :, 1])
        opt_axes[i].axis('equal')
        opt_axes[i].set_xlabel("x")
        opt_axes[i].set_ylabel("y")
        opt_axes[i].legend()

    loss_fig.tight_layout()
    opt_fig.tight_layout()
    writer.add_figure("ODE_opt/loss", loss_fig, epoch)
    writer.add_figure("ODE_opt/result", opt_fig, epoch)
    plt.close("all")


def plot_opt(writer, reference_pts, recon_control_points, loc, size, epoch):
    batch_size, num_obstacle, Dy = loc.size()
    if Dy != 2:
        pass

    if batch_size <= 3:
        idx = 0
    else:
        idx = np.random.randint(batch_size - 3)

    reference_pts = reference_pts[idx:idx + 3]
    recon_control_points = recon_control_points[:, idx:idx + 3]
    loc = loc[idx:idx + 3]
    size = size[idx:idx + 3]

    loc = to_numpy(loc)
    size = to_numpy(size)
    recon_control_points = to_numpy(recon_control_points)
    reference_pts = to_numpy(reference_pts)

    opt_fig, opt_axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = sns.color_palette('husl', n_colors=num_obstacle + 1)
    for i in range(3):
        opt_axes[i].plot(reference_pts[i, :, 0], reference_pts[i, :, 1], label="reference")
        opt_axes[i].scatter(reference_pts[i, :, 0], reference_pts[i, :, 1])

        obses = [Ellipse(xy=loc_, width=size_[0], height=size_[1]) for loc_, size_ in zip(loc[i], size[i])]
        for j, obs in enumerate(obses):
            opt_axes[i].add_artist(obs)
            obs.set_alpha(0.5)
            obs.set_facecolor(colors[j])

        ode_num_timestamps = recon_control_points.shape[0]
        for j in range(ode_num_timestamps):
            opt_axes[i].plot(recon_control_points[j, i, :, 0], recon_control_points[j, i, :, 1],
                             label="opt_{}".format(j))
            opt_axes[i].scatter(recon_control_points[j, i, :, 0], recon_control_points[j, i, :, 1])

        x_min = np.minimum(np.min(reference_pts[i, :, 0]), np.min(recon_control_points[:, i, :, 0]))
        x_max = np.maximum(np.max(reference_pts[i, :, 0]), np.max(recon_control_points[:, i, :, 0]))
        y_min = np.minimum(np.min(reference_pts[i, :, 1]), np.min(recon_control_points[:, i, :, 1]))
        y_max = np.maximum(np.max(reference_pts[i, :, 1]), np.max(recon_control_points[:, i, :, 1]))
        x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
        x_range, y_range = x_max - x_min, y_max - y_min
        x_min, x_max = x_mid - x_range, x_mid + x_range
        y_min, y_max = y_mid - y_range, y_mid + y_range

        opt_axes[i].axis('equal')
        opt_axes[i].set_xlabel("x")
        opt_axes[i].set_ylabel("y")

        opt_axes[i].set(xlim=[x_min, x_max], ylim=[y_min, y_max])
        opt_axes[i].legend()

    opt_fig.tight_layout()
    writer.add_figure("sample", opt_fig, epoch)
    plt.close("all")


def plot_obs_dist(writer, params, full_traj, loc_mu, loc_log_var, size_mu, size_log_var, epoch):
    batch_size, num_obstacle, Dy = loc_mu.size()
    if Dy != 2:
        pass
    if batch_size <= 3:
        idx = 0
    else:
        idx = np.random.randint(batch_size - 3)

    full_traj = to_numpy(full_traj[idx:idx + 3, :Dy])
    loc_prior_mu = np.mean(full_traj, axis=-1)
    loc_prior_var = np.var(full_traj, axis=-1) * params.model_params.obs_loc_prior_var_coef
    loc_prior_std = np.sqrt(loc_prior_var)
    loc_mu = to_numpy(loc_mu[idx:idx + 3])
    loc_log_var = to_numpy(loc_log_var[idx:idx + 3])
    loc_std = np.exp(0.5 * loc_log_var)
    size_mu = to_numpy(size_mu[idx:idx + 3])
    size_log_var = to_numpy(size_log_var[idx:idx + 3])
    size_std = np.exp(0.5 * size_log_var)

    dist_fig, dist_axes = plt.subplots(1, 3, figsize=(15, 5))

    def softplus(a):
        return np.log(1 + np.exp(a))

    colors = sns.color_palette('husl', n_colors=num_obstacle + 1)
    for i in range(3):
        obs_loc_prior = Ellipse(xy=loc_prior_mu[i], width=loc_prior_std[i, 0], height=loc_prior_std[i, 1],
                                facecolor='none')
        edge_c = colors[-1]
        dist_axes[i].add_artist(obs_loc_prior)
        obs_loc_prior.set_edgecolor(edge_c)

        obs_loc = [Ellipse(xy=loc_, width=size_[0], height=size_[1], facecolor='none')
                   for loc_, size_ in zip(loc_mu[i], loc_std[i])]
        obs_size_mu = [Ellipse(xy=loc_, width=size_[0], height=size_[1], facecolor='none')
                       for loc_, size_ in zip(loc_mu[i], softplus(size_mu[i]))]
        obs_size_s = [Ellipse(xy=loc_, width=size_[0], height=size_[1], facecolor='none')
                      for loc_, size_ in zip(loc_mu[i], softplus(size_mu[i] - size_std[i]))]
        obs_size_l = [Ellipse(xy=loc_, width=size_[0], height=size_[1], facecolor='none')
                      for loc_, size_ in zip(loc_mu[i], softplus(size_mu[i] + size_std[i]))]
        for j, (loc_, size_mu_, size_s, size_l) in enumerate(zip(obs_loc, obs_size_mu, obs_size_s, obs_size_l)):
            edge_c = colors[j]
            dist_axes[i].add_artist(loc_)
            dist_axes[i].add_artist(size_mu_)
            dist_axes[i].add_artist(size_s)
            dist_axes[i].add_artist(size_l)
            loc_.set_edgecolor(edge_c)
            size_mu_.set_edgecolor(edge_c)
            size_s.set_edgecolor(edge_c)
            size_l.set_edgecolor(edge_c)
            loc_.set_linestyle('--')

        x_min = np.min([loc_prior_mu[i, 0] - loc_prior_std[i, 0],
                        np.min(loc_mu[i, :, 0] - loc_std[i, :, 0]),
                        np.min(loc_mu[i, :, 0] - softplus(size_mu[i, :, 0] + size_std[i, :, 0]))])
        x_max = np.max([loc_prior_mu[i, 0] + loc_prior_std[i, 0],
                        np.max(loc_mu[i, :, 0] + loc_std[i, :, 0]),
                        np.max(loc_mu[i, :, 0] + softplus(size_mu[i, :, 0] + size_std[i, :, 0]))])
        y_min = np.min([loc_prior_mu[i, 1] - loc_prior_std[i, 1],
                        np.min(loc_mu[i, :, 1] - loc_std[i, :, 1]),
                        np.min(loc_mu[i, :, 1] - softplus(size_mu[i, :, 1] + size_std[i, :, 1]))])
        y_max = np.max([loc_prior_mu[i, 1] + loc_prior_std[i, 1],
                        np.max(loc_mu[i, :, 1] + loc_std[i, :, 1]),
                        np.max(loc_mu[i, :, 1] + softplus(size_mu[i, :, 1] + size_std[i, :, 1]))])
        x_mid, y_mid = (x_max + x_min) / 2, (y_max + y_min) / 2
        x_range, y_range = x_max - x_min, y_max - y_min
        x_min, x_max = x_mid - x_range * 0.6, x_mid + x_range * 0.6
        y_min, y_max = y_mid - y_range * 0.6, y_mid + y_range * 0.6

        dist_axes[i].axis('equal')
        dist_axes[i].set(xlim=[x_min, x_max], ylim=[y_min, y_max])

    dist_fig.tight_layout()
    writer.add_figure("distribution", dist_fig, epoch)
    plt.close("all")
