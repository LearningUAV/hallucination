import numpy as np
from scipy.interpolate import BSpline

import torch
import torch.nn as nn
import torch.nn.functional as F

from solver import Neural_ODE_Decoder, CVX_Decoder, EPS


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.model_params = model_params = params.model_params

        self.conv1d_layers = nn.ModuleList()
        output_len = model_params.full_traj_len
        prev_channel = 2 * params.Dy
        for channel, kernel_size, stride in \
                zip(model_params.encoder_channels, model_params.kernel_sizes, model_params.strides):
            self.conv1d_layers.append(nn.Conv1d(in_channels=prev_channel, out_channels=channel,
                                                kernel_size=kernel_size, stride=stride))
            prev_channel = channel
            output_len = int((output_len - kernel_size) / stride + 1)
        self.fc = nn.Linear(output_len * model_params.encoder_channels[-1], model_params.num_obs * params.Dy * 4)
        self.ln = nn.LayerNorm(model_params.num_obs * params.Dy * 4)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        output = input
        for conv1d_layer in self.conv1d_layers:
            output = F.leaky_relu(conv1d_layer(output))
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        # output = self.ln(output)
        output = output.view(output.size(0), self.model_params.num_obs, 4, self.params.Dy)

        loc_mu = output[:, :, 0]
        loc_log_var = output[:, :, 1]
        size_mu = output[:, :, 2]
        size_log_var = output[:, :, 3]

        loc_log_var = torch.clamp(loc_log_var, min=np.log(EPS))
        size_log_var = torch.clamp(size_log_var, min=np.log(EPS))

        loc = self.reparameterize(loc_mu, loc_log_var)
        size = torch.log(1 + torch.exp(self.reparameterize(size_mu, size_log_var)))
        return loc_mu, loc_log_var, loc, size_mu, size_log_var, size


class Hallucination(nn.Module):
    def __init__(self, params):
        super(Hallucination, self).__init__()
        self.params = params
        self.model_params = self.params.model_params
        params.model_params.num_control_pts = params.model_params.knot_end - params.model_params.knot_start
        if params.optimization_params.decoder == "Neural_ODE":
            Decoder = Neural_ODE_Decoder
        elif params.optimization_params.decoder == "CVX":
            Decoder = CVX_Decoder
        else:
            raise NotImplementedError

        self.encoder = Encoder(params).to(params.device)
        self.decoder = Decoder(params).to(params.device)
        self.coef = self._init_bspline_coef()

    def _init_bspline_coef(self):
        model_params = self.model_params

        # Find B-spline coef to convert control points to pos, vel, acc, jerk
        knots = np.arange(model_params.knot_start, model_params.knot_end) * model_params.knot_dt
        pos_bspline = BSpline(knots, np.eye(model_params.num_control_pts), 3, extrapolate=False)
        vel_bspline = pos_bspline.derivative(nu=1)
        # acc_bspline = pos_bspline.derivative(nu=2)
        label_t_min, label_t_max = knots[3], knots[-4]
        t = np.linspace(label_t_min, label_t_max, model_params.traj_len)
        # coef = np.array([pos_bspline(t), vel_bspline(t), acc_bspline(t)])             # (3, T, num_control_pts)
        coef = np.array([pos_bspline(t), vel_bspline(t)])                               # (2, T, num_control_pts)
        assert not np.isnan(coef).any()
        coef = torch.from_numpy(coef.astype(np.float32)[None, ...]).to(self.params.device)   # (1, 2, T, num_control_pts)
        return coef

    def forward(self, full_traj, reference_pts, decode=False):
        """

        :param full_traj: (batch_size, T_full, 2 * Dy) tensor, recorded trajectory, pos + vel
        :param reference_pts: (batch_size, num_control_pts, Dy) tensor, reference control_pts during the optimization
        :param decode: True when training
        :return: recon_traj: (batch_size, T, Dy) tensor
        """
        model_params = self.model_params
        loc_mu, loc_log_var, loc, size_mu, size_log_var, size = self.encoder(full_traj)
        if not decode:
            return loc_mu, loc_log_var, size_mu, size_log_var

        # initial traj before optimization, straight line from start to goal, (batch_size, num_control_pts, Dy)
        init_control_pts = reference_pts[:, None, 0] + \
                           torch.linspace(0, 1, model_params.num_control_pts)[None, :, None].to(self.params.device) * \
                           (reference_pts[:, None, -1] - reference_pts[:, None, 0])
        recon_control_points = self.decoder(init_control_pts, loc, size, reference_pts)

        # (batch_size, 1, num_control_pts, 3)
        last_recon_control_points = recon_control_points[-1, :, None]
        recon_traj = torch.matmul(self.coef, last_recon_control_points)

        return recon_traj, recon_control_points, loc_mu, loc_log_var, loc, size_mu, size_log_var, size

    def loss(self, full_traj, traj, recon_traj, reference_pts, loc_mu, loc_log_var, loc, size_mu, size_log_var, size):
        """
        :param full_traj: (batch_size, Dy, T_) tensor
        :param traj: (batch_size, T, Dy) tensor
        :param recon_traj: (batch_size, T, Dy) tensor
        :param loc_mu: (batch_size, num_obs, Dy) tensor
        :param loc_log_var: (batch_size, num_obs, Dy) tensor
        :param loc: (batch_size, num_obs, Dy) tensor
        :param size_mu: (batch_size, num_obs, Dy) tensor
        :param size_log_var: (batch_size, num_obs, Dy) tensor
        :param size: (batch_size, num_obs, Dy) tensor
        :return:
        """
        Dy = self.params.Dy
        device = self.params.device
        Dy = self.params.Dy
        batch_size = self.params.training_params.batch_size

        # reconstruction error
        recon_loss = torch.mean(torch.sum((traj - recon_traj) ** 2, dim=(1, 2)))

        # regularization loss
        # repulsion between obs
        loc_diff = loc[:, :, None] - loc[:, None]                                   # (batch_size, num_obs, num_obs, Dy)
        loc_diff_norm = torch.norm(loc_diff, dim=-1)
        # mask distance between the same obs to avoid numerical issues
        loc_diff[loc_diff_norm == 0] = size.detach().view(-1, Dy) * 3

        loc_diff_norm = torch.norm(loc_diff, dim=-1, keepdim=True)
        loc_diff_direction = loc_diff / loc_diff_norm                               # (batch_size, num_obs, num_obs, Dy)
        size_ = size[:, None, :]                                                    # (batch_size, 1, num_obs, Dy)
        tmp = torch.sqrt(torch.sum(loc_diff_direction ** 2 / size_ ** 2, dim=-1))   # (batch_size, num_obs, num_obs)
        radius_along_direction = 1 / tmp                                            # (batch_size, num_obs, num_obs)

        combined_radius_along_direction = radius_along_direction + torch.transpose(radius_along_direction, 1, 2)
        obs_overlap = combined_radius_along_direction - loc_diff_norm[..., 0]
        repulsive_loss = torch.clamp(obs_overlap, min=0) ** 2
        repulsive_loss = torch.sum(repulsive_loss) / batch_size
        repulsive_loss *= self.params.model_params.lambda_mutual_repulsion

        # repulsion between obs and reference_pts
        diff = reference_pts.view(batch_size, 1, -1, Dy) - loc.view(batch_size, -1, 1, Dy)
        diff_norm = torch.linalg.norm(diff, dim=-1)                                 # (B, num_obs, num_control_pts)
        direction = diff / torch.clamp(diff_norm, min=EPS)[..., None]

        # intersection = t. denote direction = (x, y, z) and obs_size = (a, b, c)
        # then (t * x)^2 / a^2 + (t * y)^2 / b^2 + (t * z)^2 / c^2 = 1
        # shape: (B, num_obs, num_control_pts)
        size = size.view(batch_size, -1, 1, Dy)
        intersection_inv = torch.sqrt(torch.sum(direction ** 2 / size ** 2, dim=-1))
        intersection_inv = torch.clamp(intersection_inv, min=EPS)
        intersection = 1 / intersection_inv

        clearance = self.params.optimization_params.clearance
        dist = diff_norm - intersection
        dist_error = clearance - dist
        reference_repulsion_loss = torch.sum(torch.clamp(dist_error, min=0) ** 2) / batch_size
        reference_repulsion_loss *= self.params.model_params.lambda_reference_repulsion

        # KL divergence from prior
        loc_var = torch.exp(loc_log_var)                                            # (batch_size, num_obs, Dy)

        loc_prior_mu = torch.mean(full_traj[:, :Dy], dim=-1)                        # (batch_size, Dy)
        loc_prior_var = torch.var(full_traj[:, :Dy], dim=-1)                        # (batch_size, Dy)
        loc_prior_var *= self.params.model_params.obs_loc_prior_var_coef
        loc_prior_mu = loc_prior_mu[:, None]                                        # (batch_size, 1, Dy)
        loc_prior_var = loc_prior_var[:, None]                                      # (batch_size, 1, Dy)

        # kl divergence between two diagonal Gaussian
        loc_kl_loss = 0.5 * (torch.sum(loc_var / loc_prior_var
                                       + (loc_mu - loc_prior_mu) ** 2 / loc_prior_var
                                       + torch.log(loc_prior_var) - torch.log(loc_var),
                                       dim=-1)
                             - self.params.Dy)
        loc_kl_loss = torch.mean(torch.sum(loc_kl_loss, dim=1))
        loc_kl_loss *= self.params.model_params.lambda_loc_kl

        size_var = torch.exp(size_log_var)                                          # (batch_size, num_obs, Dy)

        size_prior_mu = self.params.model_params.obs_size_prior_mu
        size_prior_var = self.params.model_params.obs_size_prior_var
        size_prior_std = np.sqrt(size_prior_var)
        size_prior_mu_ = np.log(np.exp(size_prior_mu) - 1.0).astype(np.float32)
        size_prior_std_ = np.log(np.exp(size_prior_mu + size_prior_std) - 1.0).astype(np.float32)
        size_prior_var_ = size_prior_std_ ** 2
        size_prior_mu = size_prior_mu_ * torch.ones(self.params.Dy).to(device)
        size_prior_var = size_prior_var_ * torch.ones(self.params.Dy).to(device)
        size_prior_mu = size_prior_mu[None, None, :]                                # (1, 1, Dy)
        size_prior_var = size_prior_var[None, None, :]

        # kl divergence between two diagonal Gaussian
        size_kl_loss = 0.5 * (torch.sum(size_var / size_prior_var
                                        + (size_mu - size_prior_mu) ** 2 / size_prior_var
                                        + torch.log(size_prior_var) - torch.log(size_var),
                                        dim=-1)
                              - self.params.Dy)
        size_kl_loss = torch.mean(torch.sum(size_kl_loss, dim=1))
        size_kl_loss *= self.params.model_params.lambda_size_kl

        loss = recon_loss + loc_kl_loss + size_kl_loss + repulsive_loss + reference_repulsion_loss
        loss_detail = {"loss": loss,
                       "recon_loss": recon_loss,
                       "loc_kl_loss": loc_kl_loss,
                       "size_kl_loss": size_kl_loss,
                       "repulsive_loss": repulsive_loss,
                       "reference_repulsion_loss": reference_repulsion_loss}
        loss_detail = dict([(k, v.item()) for k, v in loss_detail.items()])

        return loss, loss_detail

    def test(self, reference_pts, recon_control_points, loc, size):
        model_params = self.model_params
        opt_func = self.decoder.opt_func
        # initial traj before optimization, straight line from start to goal, (batch_size, num_control_pts, Dy)
        init_control_pts = reference_pts[:, None, 0] + \
                           torch.linspace(0, 1, model_params.num_control_pts)[None, :, None].to(self.params.device) * \
                           (reference_pts[:, None, -1] - reference_pts[:, None, 0])
        opt_func.update(loc, size, reference_pts, init_control_pts)
        ode_num_timestamps, batch_size, num_control_pts, Dy = recon_control_points.size()

        losses = np.zeros((batch_size, ode_num_timestamps))
        for i in range(ode_num_timestamps):
            for j in range(batch_size):
                losses[j, i] = opt_func.loss(recon_control_points[i, j].view(1, num_control_pts * Dy)).item()

        loc = loc.cpu().detach().numpy()
        size = size.cpu().detach().numpy()
        recon_control_points = recon_control_points.cpu().detach().numpy()
        reference_pts = reference_pts.cpu().detach().numpy()

        return reference_pts, loc, size, recon_control_points, losses

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.decoder.train(training)
