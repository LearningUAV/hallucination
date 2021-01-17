import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_ode_opt(writer, model, reference_pts, recon_control_points, loc, size, epoch):
    batch_size = reference_pts.size(0)
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


def plot_obs_dist(writer, reference_pts, recon_control_points,
                  loc_mu, loc_log_var, loc, size_mu, size_log_var, size,
                  epoch):
    batch_size = reference_pts.size(0)
    idx = np.random.randint(batch_size - 3)

    reference_pts = reference_pts[idx:idx + 3].cpu().detach().numpy()
    recon_control_points = recon_control_points[:, idx:idx + 3].cpu().detach().numpy()
    loc_mu = loc_mu[idx:idx + 3].cpu().detach().numpy()
    loc_log_var = loc_log_var[idx:idx + 3].cpu().detach().numpy()
    loc_std = np.exp(0.5 * loc_log_var)
    size_mu = size_mu[idx:idx + 3].cpu().detach().numpy()
    size_log_var = size_log_var[idx:idx + 3].cpu().detach().numpy()
    size_std = np.exp(0.5 * size_log_var)
    loc = loc[idx:idx + 3].cpu().detach().numpy()
    size = size[idx:idx + 3].cpu().detach().numpy()

    dist_fig, dist_axes = plt.subplots(1, 3, figsize=(18, 6))

    def softplus(a):
        return np.log(1 + np.exp(a))

    for i in range(3):
        obs_loc = [Ellipse(xy=loc_, width=size_[0], height=size_[1], facecolor='none')
                   for loc_, size_ in zip(loc_mu[i], loc_std[i])]
        obs_size_mu = [Ellipse(xy=loc_, width=size_[0], height=size_[1], facecolor='none')
                       for loc_, size_ in zip(loc_mu[i], softplus(size_mu[i]))]
        obs_size_s = [Ellipse(xy=loc_, width=size_[0], height=size_[1], facecolor='none')
                      for loc_, size_ in zip(loc_mu[i], softplus(size_mu[i] - size_std[i]))]
        obs_size_l = [Ellipse(xy=loc_, width=size_[0], height=size_[1], facecolor='none')
                      for loc_, size_ in zip(loc_mu[i], softplus(size_mu[i] + size_std[i]))]
        for loc_, size_mu_, size_s, size_l in zip(obs_loc, obs_size_mu, obs_size_s, obs_size_l):
            edge_c = np.random.rand(3)
            dist_axes[i].add_artist(loc_)
            dist_axes[i].add_artist(size_mu_)
            dist_axes[i].add_artist(size_s)
            dist_axes[i].add_artist(size_l)
            loc_.set_edgecolor(edge_c)
            size_mu_.set_edgecolor(edge_c)
            size_s.set_edgecolor(edge_c)
            size_l.set_edgecolor(edge_c)
            loc_.set_linestyle('--')
        dist_axes[i].set(xlim=[-10, 10], ylim=[-10, 10])

    dist_fig.tight_layout()
    writer.add_figure("ODE_opt/dist", dist_fig, epoch)
    plt.close("all")
