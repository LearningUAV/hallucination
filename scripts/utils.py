import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_ode_opt(writer, model, full_traj, reference_pts, epoch):
    loss_fig, loss_axes = plt.subplots(1, 3, figsize=(18, 4))
    opt_fig, opt_axes = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        try:
            reference_pts_np, loc, size, recon_control_points, losses = model.test(full_traj, reference_pts)
        except AssertionError:
            continue

        opt_params = model.params.optimization_params
        loss_axes[i].plot(np.linspace(0, opt_params.ode_t_end, opt_params.ode_num_timestamps), losses)
        loss_axes[i].set_xlabel("ODE t")
        loss_axes[i].set_ylabel("loss")

        opt_axes[i].plot(reference_pts_np[:, 0], reference_pts_np[:, 1], label="reference")
        opt_axes[i].scatter(reference_pts_np[:, 0], reference_pts_np[:, 1])

        obses = [Ellipse(xy=loc_, width=size_[0], height=size_[1]) for loc_, size_ in zip(loc, size)]
        for obs in obses:
            opt_axes[i].add_artist(obs)
            obs.set_alpha(0.5)
            obs.set_facecolor(np.random.rand(3))

        ode_num_timestamps = recon_control_points.shape[0]
        for j in range(ode_num_timestamps):
            opt_axes[i].plot(recon_control_points[j, :, 0], recon_control_points[i, :, 1], label="opt_{}".format(j))
            opt_axes[i].scatter(recon_control_points[j, :, 0], recon_control_points[i, :, 1])
        opt_axes[i].axis('equal')
        opt_axes[i].set_xlabel("x")
        opt_axes[i].set_ylabel("y")

    loss_fig.tight_layout()
    opt_fig.tight_layout()
    writer.add_figure("ODE_opt/loss", loss_fig, epoch)
    writer.add_figure("ODE_opt/result", opt_fig, epoch)
    plt.close("all")
