#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
import dynamic_reconfigure.client

import os
import scipy
import numpy as np
from scipy import stats

import torch

from LfD_main import TrainingParams, LfD_2D_model


class Predictor:
    def __init__(self, policy, params):
        self.policy = policy
        self.params = params

        self.global_path = []
        self.vel = 0
        self.ang_vel = 0

    def update_status(self, msg):
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z
        q0 = msg.pose.pose.orientation.w
        self.X = msg.pose.pose.position.x
        self.Y = msg.pose.pose.position.y
        self.PSI = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))

    @staticmethod
    def transform_lg(gp, X, Y, PSI):
        R_r2i = np.matrix([[np.cos(PSI), -np.sin(PSI), X], [np.sin(PSI), np.cos(PSI), Y], [0, 0, 1]])
        R_i2r = np.linalg.inv(R_r2i)

        pi = np.concatenate([gp, np.ones_like(gp[:, :1])], axis=-1)
        pr = np.matmul(R_i2r, pi.T)
        return np.asarray(pr[:2, :]).T

    def update_global_path(self, msg):
        gp = []
        for pose in msg.poses:
            gp.append([pose.pose.position.x, pose.pose.position.y])
        gp = np.array(gp)
        x = gp[:,0]
        try:
            xhat = scipy.signal.savgol_filter(x, 19, 3)
        except:
            xhat = x
        y = gp[:,1]
        try:
            yhat = scipy.signal.savgol_filter(y, 19, 3)
        except:
            yhat = y

        gphat = np.column_stack((xhat, yhat))
        gphat.tolist()
        self.global_path = self.transform_lg(gphat, self.X, self.Y, self.PSI)

    def get_local_goal(self, gp):
        local_goal = np.zeros(2)
        odom = np.zeros(2)
        if len(gp) > 0:
            if np.linalg.norm(gp[0] - odom) > 0.05:
                odom = gp[0]
            for wp in gp:
                dist = np.linalg.norm(wp - odom)
                if dist > self.params.local_goal_dist:
                    break
            local_goal = wp - odom
            local_goal /= np.linalg.norm(local_goal)

        return local_goal

    def update_cmd_vel(self, msg):
        if len(self.global_path) == 0:
            return

        scan = np.array(msg.ranges)
        scan = np.minimum(scan, self.params.laser_max_range).astype(np.float32)
        # scan = np.flip(scan).copy()
        local_goal = self.get_local_goal(self.global_path).astype(np.float32)
        scan = torch.from_numpy(scan[None]).to(self.params.device)
        local_goal = torch.from_numpy(local_goal[None]).to(self.params.device)
        cmd = self.policy(scan, local_goal)
        cmd = cmd[0]                    # remove batch size
        self.vel, self.ang_vel = cmd
        print("[INFO] current lin_vel: {:4.2f}, ang_vel: {:5.2f}".format(self.vel, self.ang_vel))


if __name__ == '__main__':
    model_path = "2021-02-22-12-07-52"
    model_fname = "model_500"
    update_dt = 0.04
    local_goal_dist = 1.5
    laser_max_range = 2.5

    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    params_path = os.path.join(repo_path, "rslts", "LfD_2D_rslts", model_path, "params.json")
    model_path = os.path.join(repo_path, "rslts", "LfD_2D_rslts", model_path, "trained_models", model_fname)

    params = TrainingParams(params_path, train=False)
    device = torch.device("cpu")

    params.device = device
    training_params = params.training_params
    training_params.load_model = model_path
    params.local_goal_dist = local_goal_dist
    params.laser_max_range = laser_max_range

    model = LfD_2D_model(params).to(device)
    if training_params.load_model is not None and os.path.exists(training_params.load_model):
        model.load_state_dict(torch.load(training_params.load_model, map_location=device))

    predictor = Predictor(model, params)

    rospy.init_node('context_classifier', anonymous=True)
    sub_robot = rospy.Subscriber("/odometry/filtered", Odometry, predictor.update_status)
    sub_gp = rospy.Subscriber("/move_base/TrajectoryPlannerROS/global_plan",
                              Path, predictor.update_global_path, queue_size=1)
    sub_scan = rospy.Subscriber("/front/scan", LaserScan, predictor.update_cmd_vel, queue_size=1)
    velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    client = dynamic_reconfigure.client.Client('move_base/TrajectoryPlannerROS')
    client2 = dynamic_reconfigure.client.Client('move_base/local_costmap/inflater_layer')

    prev_cmd_time = None
    while not rospy.is_shutdown():
        try:
            now = rospy.Time.now()
            if prev_cmd_time is None or (now - prev_cmd_time).to_sec() >= update_dt:
                vel_msg = Twist()
                vel_msg.linear.x = predictor.vel
                vel_msg.angular.z = predictor.ang_vel
                velocity_publisher.publish(vel_msg)

                prev_cmd_time = now
        except rospy.exceptions.ROSInterruptException:
            break
