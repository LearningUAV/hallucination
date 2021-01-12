import numpy as np
from scipy.interpolate import BSpline

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

EPS = 1e-6


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


class OptimizationFunc(nn.Module):
    def __init__(self, params):
        super(OptimizationFunc, self).__init__()
        self.obs_loc = None
        self.obs_size = None
        self.params = params
        self.reference_pts = None
        self.direction = None
        self.intersection = None
        self.SECOND_DERIVATIVE_CONTINOUS = False

    def update(self, obs_loc, obs_size, reference_pts):
        """
        :param obs_loc: (B, num_obs, Dy) tensor
        :param obs_size: (B, num_obs, Dy) tensor
        :param reference_pts:  (B, num_control_pts, Dy) tensor
            pos on collected trajectory, used as the guidance for control points to get out of obstacles
        :return: 
        """
        self.obs_loc = obs_loc
        self.obs_size = obs_size
        self.reference_pts = reference_pts

        # direction: normalized, from obs_loc to reference_pts, (B, num_control_pts, num_obs, Dy)
        batch_size, _, Dy = list(reference_pts.size())
        diff = reference_pts.view(batch_size, -1, 1, Dy) - obs_loc.view(batch_size, 1, -1, Dy)
        self.direction = diff / torch.linalg.norm(diff, dim=-1, keepdims=True)

        # intersection = obs_loc + t * direction. denote direction = (x, y, z) and obs_size = (a, b, c)
        # then (t * x)^2 / a^2 + (t * y)^2 / b^2 + (t * z)^2 / c^2 = 1
        obs_size = obs_size.view(batch_size, 1, -1, Dy)
        t = 1 / torch.sqrt(torch.sum(self.direction ** 2 / obs_size ** 2, dim=-1, keepdim=True))
        self.intersection = obs_loc[:, None] + t * self.direction       # (B, num_control_pts, num_obs, Dy)

    def loss(self, control_pts):
        params = self.params
        device = params.device
        model_params = params.model_params
        optimization_params = params.optimization_params
        batch_size = control_pts.size(0)
        assert self.obs_loc is not None and self.obs_size is not None

        control_pts = control_pts.view(batch_size, -1, params.Dy)  # (B, num_control_pts, Dy)

        # smoothness
        # for some reason, ego-planner doesn't divide dt when calculating vel and acc, so keep the same here
        # https://github.com/ZJU-FAST-Lab/ego-planner/blob/master/src/planner/bspline_opt/src/bspline_optimizer.cpp#L413-L452
        vel = (control_pts[:, 1:] - control_pts[:, :-1])                    # (B, num_control_pts - 1, Dy)
        acc = (vel[:, 1:] - vel[:, :-1])                                    # (B, num_control_pts - 2, Dy)
        jerk = (acc[:, 1:] - acc[:, :-1])                                    # (B, num_control_pts - 2, Dy)

        smoothness_loss = torch.mean(torch.sum(jerk ** 2, dim=(1, 2)))

        # collision
        # not used, but may be a good future reference
        # https://math.stackexchange.com/questions/3722553/find-intersection-between-line-and-ellipsoid

        control_pts_ = control_pts.view(batch_size, -1, 1, params.Dy)

        # dist = (control_pts - intersection_pts).dot(direction to reference on traj), (B, num_control_pts, num_obs)
        dist = torch.sum((control_pts_ - self.intersection) * self.direction, dim=-1)
        clearance = optimization_params.clearance
        dist_error = clearance - dist
        gt_0_le_clearance_mask = (dist_error > 0) & (dist_error <= clearance)
        gt_clearance_mask = dist_error > clearance
        dist_gt_0_le_clearance = dist_error[gt_0_le_clearance_mask]
        dist_gt_clearance = dist_error[gt_clearance_mask]

        # see https://arxiv.org/pdf/2008.08835.pdf Eq 5
        a, b, c = 3 * clearance, -3 * clearance ** 2, clearance ** 3
        collision_loss = torch.sum(dist_gt_0_le_clearance ** 3) + \
            torch.sum(a * dist_gt_clearance ** 2 + b * dist_gt_clearance + c)
        collision_loss /= batch_size

        # feasibility, see https://arxiv.org/pdf/2008.08835.pdf Eq 10 and
        # https://github.com/ZJU-FAST-Lab/ego-planner/blob/master/src/planner/bspline_opt/src/bspline_optimizer.cpp#L462-L577

        knot_dt = model_params.knot_dt
        demarcation = optimization_params.demarcation
        max_vel = optimization_params.max_vel
        max_acc = optimization_params.max_acc

        vel /= knot_dt
        acc /= knot_dt ** 2

        if self.SECOND_DERIVATIVE_CONTINOUS:
            a, b, c = 3 * demarcation, -3 * demarcation ** 2, demarcation ** 3
            raise NotImplementedError("wait for ego_planner response")
        else:
            vel = torch.abs(vel)
            ge_max_vel_mask = vel >= max_vel
            vel_ge_max_vel = vel[ge_max_vel_mask]
            vel_feasibility_loss = (vel_ge_max_vel - max_vel) ** 2

            acc = torch.abs(acc)
            ge_max_acc_mask = acc >= max_acc
            acc_ge_max_acc = acc[ge_max_acc_mask]
            acc_feasibility_loss = (acc_ge_max_acc - max_acc) ** 2

            # extra "/ knot_dt ** 2": from ego_planner, to make vel and acc have similar magnitudes
            feasibility_loss = torch.sum(vel_feasibility_loss) / knot_dt ** 2 + torch.sum(acc_feasibility_loss)
            feasibility_loss /= batch_size

        loss = optimization_params.lambda_smoothness * smoothness_loss + \
               optimization_params.lambda_collision * collision_loss + \
               optimization_params.lambda_feasibility * feasibility_loss

        return loss

    def forward(self, t, control_pts):
        params = self.params
        device = params.device
        model_params = params.model_params
        optimization_params = params.optimization_params
        batch_size = control_pts.size(0)
        assert self.obs_loc is not None and self.obs_size is not None

        control_pts = control_pts.view(batch_size, -1, params.Dy)  # (B, num_control_pts, Dy)

        # smoothness
        # for some reason, ego-planner doesn't divide dt when calculating vel and acc, so keep the same here
        # https://github.com/ZJU-FAST-Lab/ego-planner/blob/master/src/planner/bspline_opt/src/bspline_optimizer.cpp#L413-L452
        vel = (control_pts[:, 1:] - control_pts[:, :-1])                    # (B, num_control_pts - 1, Dy)
        acc = (vel[:, 1:] - vel[:, :-1])                                    # (B, num_control_pts - 2, Dy)
        jerk = (acc[:, 1:] - acc[:, :-1])                                    # (B, num_control_pts - 2, Dy)

        smoothness_grad = torch.zeros_like(control_pts)
        smoothness_grad[:, :-3] += -1 * 2 * jerk
        smoothness_grad[:, 1:-2] += 3 * 2 * jerk
        smoothness_grad[:, 2:-1] += -3 * 2 * jerk
        smoothness_grad[:, 3:] += 1 * 2 * jerk
        smoothness_grad /= batch_size

        # collision
        # not used, but may be a good future reference
        # https://math.stackexchange.com/questions/3722553/find-intersection-between-line-and-ellipsoid

        control_pts_ = control_pts.view(batch_size, -1, 1, params.Dy)

        # dist = (control_pts - intersection_pts).dot(direction to reference on traj), (B, num_control_pts, num_obs)
        dist = torch.sum((control_pts_ - self.intersection) * self.direction, dim=-1)
        clearance = optimization_params.clearance
        dist_error = clearance - dist
        gt_0_le_clearance_mask = (dist_error > 0) & (dist_error <= clearance)
        gt_clearance_mask = dist_error > clearance
        dist_gt_0_le_clearance = dist_error[gt_0_le_clearance_mask]
        dist_gt_clearance = dist_error[gt_clearance_mask]

        # see https://arxiv.org/pdf/2008.08835.pdf Eq 5
        a, b, c = 3 * clearance, -3 * clearance ** 2, clearance ** 3

        collision_grad = torch.zeros(batch_size, model_params.num_control_pts, model_params.num_obs, params.Dy).to(device)
        collision_grad[gt_0_le_clearance_mask] += -3 * torch.unsqueeze(dist_gt_0_le_clearance ** 2, dim=-1) \
                                                  * self.direction[gt_0_le_clearance_mask]
        collision_grad[gt_clearance_mask] += -torch.unsqueeze(2 * a * dist_gt_clearance + b, dim=-1) \
                                             * self.direction[gt_clearance_mask]
        collision_grad = torch.sum(collision_grad, dim=2)
        collision_grad /= batch_size

        # feasibility, see https://arxiv.org/pdf/2008.08835.pdf Eq 10 and
        # https://github.com/ZJU-FAST-Lab/ego-planner/blob/master/src/planner/bspline_opt/src/bspline_optimizer.cpp#L462-L577

        knot_dt = model_params.knot_dt
        demarcation = optimization_params.demarcation
        max_vel = optimization_params.max_vel
        max_acc = optimization_params.max_acc

        vel /= knot_dt
        acc /= knot_dt ** 2

        if self.SECOND_DERIVATIVE_CONTINOUS:
            a, b, c = 3 * demarcation, -3 * demarcation ** 2, demarcation ** 3
            raise NotImplementedError("wait for ego_planner response")
            # # max_vel < v < vj
            # ge_max_vel_lt_vj_mask = (vel >= max_vel) & (vel < max_vel + params.demarcation)
            # # v >= vj
            # ge_vj_mask = vel >= max_vel + params.demarcation
            # # -vj < v <= -max_vel
            # le_neg_max_vel_gt_neg_vj_mask = (vel <= -max_vel) & (vel > -(max_vel + params.demarcation))
            # # v < -vj
            # le_neg_vj_mask = vel <= -(max_vel + params.demarcation)
            #
            # vel_ge_max_vel_lt_vj = vel[ge_max_vel_lt_vj_mask]
            # vel_ge_vj = vel[ge_vj_mask]
            # vel_le_neg_max_vel_gt_neg_vj = vel[le_neg_max_vel_gt_neg_vj_mask]
            # vel_le_neg_vj = vel[le_neg_vj_mask]
            #
            # feasibility_loss += torch.sum((vel_ge_max_vel_lt_vj - max_vel) ** 3) \
            #                     - torch.sum((vel_le_neg_max_vel_gt_neg_vj + max_vel) ** 3) \
            #                     + torch.sum(a * (vel_ge_vj - max_vel) ** 2 + b * (vel_ge_vj - max_vel) + c) \
            #                     + torch.sum(a * (vel_le_neg_vj + max_vel) ** 2 + b * (vel_le_neg_vj + max_vel) + c)
            # feasibility_loss /= knot_dt ** 3  # from ego_planner: to make vel and acc have similar magnitudes
            #
            # acc = torch.abs(acc)
            # ge_max_acc_lt_vj_mask = (acc > max_acc) & (acc < max_acc + params.demarcation)
            # ge_vj_mask = acc >= max_acc + params.demarcation
            # acc_ge_max_acc_lt_vj = acc[ge_max_acc_lt_vj_mask]
            # acc_ge_vj = acc[ge_vj_mask]
            # feasibility_loss += torch.sum((acc_ge_max_acc_lt_vj - max_acc) ** 3) + \
            #                     torch.sum(a * acc_ge_vj ** 2 + b * acc_ge_vj + c)
            # feasibility_loss /= batch_size
        else:
            vel = torch.abs(vel)
            ge_max_vel_mask = vel >= max_vel
            vel_ge_max_vel = vel[ge_max_vel_mask]

            vel_feasibility_grad = torch.zeros_like(control_pts).to(device)
            vel_feasibility_grad[:, :-1][ge_max_vel_mask] += -2 * (vel_ge_max_vel - max_vel) / knot_dt
            vel_feasibility_grad[:, 1:][ge_max_vel_mask] += 2 * (vel_ge_max_vel - max_vel) / knot_dt

            acc = torch.abs(acc)
            ge_max_acc_mask = acc >= max_acc
            acc_ge_max_acc = acc[ge_max_acc_mask]

            acc_feasibility_grad = torch.zeros_like(control_pts).to(device)
            acc_feasibility_grad[:, :-2][ge_max_acc_mask] += 2 * (acc_ge_max_acc - max_acc) / knot_dt ** 2
            acc_feasibility_grad[:, 1:-1][ge_max_acc_mask] += -4 * (acc_ge_max_acc - max_acc) / knot_dt ** 2
            acc_feasibility_grad[:, 2:][ge_max_acc_mask] += 2 * (acc_ge_max_acc - max_acc) / knot_dt ** 2

            # extra "/ knot_dt ** 2": from ego_planner, to make vel and acc have similar magnitudes
            feasibility_grad = vel_feasibility_grad / knot_dt ** 2 + acc_feasibility_grad

        grad = optimization_params.lambda_smoothness * smoothness_grad + \
               optimization_params.lambda_collision * collision_grad + \
               optimization_params.lambda_feasibility * feasibility_grad
        grad = grad.view(batch_size, -1)

        # gradient descent
        grad *= -1
        grad *= optimization_params.optimization_lr

        return grad


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.device = params.device
        self.opt_func = OptimizationFunc(params).to(params.device)
        opt_params = params.optimization_params
        self.ode_num_timestamps = opt_params.ode_num_timestamps
        self.t = np.linspace(0, opt_params.ode_t_end, opt_params.ode_num_timestamps).astype(np.float32)
        self.t = torch.from_numpy(self.t).to(params.device)

    def forward(self, init_control_pts, obs_loc, obs_size, reference_pts):
        self.opt_func.update(obs_loc, obs_size, reference_pts)
        batch_size, num_control_pts, Dy = init_control_pts.size()
        init_control_pts = init_control_pts.view(init_control_pts.size(0), -1)
        recon_control_points = odeint(self.opt_func, init_control_pts, self.t).to(self.device)
                                                                # (ode_num_timestamps, batch_size, num_control_pts * Dy)
        recon_control_points = recon_control_points.view(self.ode_num_timestamps, batch_size, num_control_pts, Dy)
        return recon_control_points


class Hallucination(nn.Module):
    def __init__(self, params):
        super(Hallucination, self).__init__()
        self.params = params
        self.model_params = self.params.model_params
        params.model_params.num_control_pts = params.model_params.knot_end - params.model_params.knot_start
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

        return recon_traj, loc_mu, loc_log_var, loc, size_mu, size_log_var, size, recon_control_points

    def loss(self, traj, recon_traj, loc_mu, loc_log_var, loc, size_mu, size_log_var, size):
        """

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
        device = self.params.device
        num_obs = self.params.model_params.num_obs
        # reconstruction error
        recon_loss = torch.mean(torch.sum((traj - recon_traj) ** 2, dim=(1, 2)))

        # regularization loss
        loc_diff = torch.cat([loc[:, :, None] * num_obs], dim=2) - torch.cat([loc[:, None] * num_obs], dim=1)
                                                                                    # (batch_size, num_obs, num_obs, Dy)
        loc_diff_norm = torch.clamp(torch.norm(loc_diff, dim=-1, keepdim=True), min=EPS)
        loc_diff_direction = loc_diff / loc_diff_norm                               # (batch_size, num_obs, num_obs, Dy)
        size_ = size[:, None, :]                                                    # (batch_size, 1, num_obs, Dy)
        tmp = torch.sqrt(torch.sum(loc_diff_direction ** 2 / size_ ** 2, dim=-1))   # (batch_size, num_obs, num_obs)
        tmp = torch.clamp(tmp, min=EPS)
        radius_along_direction = 1 / tmp                                            # (batch_size, num_obs, num_obs)
        radius_along_direction[radius_along_direction == 1 / EPS] = 0

        radius_along_direction[torch.logical_not(torch.isfinite(radius_along_direction))] = 0
        obs_distance = loc_diff_norm[:, :, :, 0] - \
                       (radius_along_direction + torch.transpose(radius_along_direction, 1, 2))
        obs_distance = loc_diff_norm[:, :, :, 0]
        repulsive_loss = -obs_distance / 2
        repulsive_loss = torch.mean(torch.sum(repulsive_loss, dim=(1, 2)))
        repulsive_loss *= self.params.model_params.lambda_repulsion

        loc_var = torch.exp(loc_log_var)                                            # (batch_size, num_obs, Dy)

        loc_prior_mu = self.params.model_params.obs_loc_prior_mu * torch.ones(self.params.Dy).to(device)
        loc_prior_var = self.params.model_params.obs_loc_prior_var * torch.ones(self.params.Dy).to(device)
        loc_prior_mu = loc_prior_mu[None, None, :]                                  # (1, 1, Dy)
        loc_prior_var = loc_prior_var[None, None, :]

        size_var = torch.exp(size_log_var)                                          # (batch_size, num_obs, Dy)

        size_prior_mu = self.params.model_params.obs_size_prior_mu * torch.ones(self.params.Dy).to(device)
        size_prior_var = self.params.model_params.obs_size_prior_var * torch.ones(self.params.Dy).to(device)
        size_prior_mu = size_prior_mu[None, None, :]                                # (1, 1, Dy)
        size_prior_var = size_prior_var[None, None, :]

        # kl divergence between two diagonal Gaussian
        loc_kl_loss = 0.5 * (torch.sum(loc_var / loc_prior_var
                                       + (loc_mu - loc_prior_mu) ** 2 / loc_prior_var
                                       + torch.log(loc_prior_var) - torch.log(loc_var),
                                       dim=-1)
                             - self.params.Dy)
        size_kl_loss = 0.5 * (torch.sum(size_var / size_prior_var
                                        + (size_mu - size_prior_mu) ** 2 / size_prior_var
                                        + torch.log(size_prior_var) - torch.log(size_var),
                                        dim=-1)
                              - self.params.Dy)
        kl_loss = torch.mean(torch.sum(loc_kl_loss + size_kl_loss, dim=1))
        kl_loss *= self.params.model_params.lambda_kl

        loss = recon_loss + kl_loss + repulsive_loss
        batch_size = self.params.training_params.batch_size

        return loss, (recon_loss, repulsive_loss, kl_loss)

    def test(self, reference_pts, recon_control_points, loc, size):
        opt_func = self.decoder.opt_func
        opt_func.update(loc, size, reference_pts)
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
