#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from cv_bridge import CvBridge
from drone_LfH.msg import Bspline

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from skimage import transform

import torch
from LfD_main import TrainingParams, Encoder, Decoder, Model

model_path = "2021-03-02-03-27-39"
model_fname = "model_1000"
update_dt = 0.08
depth_max_range = 2.0


class Predictor:
    def __init__(self, model, params):
        self.model = model
        self.bridge = CvBridge()
        self.image_size = params.model_params.image_size
        assert isinstance(self.image_size, (int, tuple))
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)
        self.max_depth = params.model_params.max_depth
        self.need_update_bspline = False
        self.device = params.device

        model.train(training=False)
        self.depth = None
        self.goal = None
        self.pos = None
        self.ori = None
        self.lin_vel = None
        self.ang_vel = None
        self.depth_pos = None
        self.depth_quat = None

    def update_depth(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth = np.asarray(cv_image)
        self.notify_bspline_update()

    def update_depth_pose(self, msg):
        depth_pose = msg.pose
        self.depth_pos = np.array([depth_pose.position.x, depth_pose.position.y, depth_pose.position.z])
        self.depth_quat = np.array([depth_pose.orientation.x, depth_pose.orientation.y, depth_pose.orientation.z,
                                    depth_pose.orientation.w])
        self.notify_bspline_update()

    def notify_bspline_update(self):
        if np.any([ele is None for ele in [self.depth, self.goal, self.ori, self.lin_vel, self.ang_vel]]):
            return
        self.need_update_bspline = True

    def update_goal(self, msg):
        goal = msg.poses[0].pose.position
        self.goal = np.array([goal.x, goal.y, goal.z])
        self.notify_bspline_update()

    def update_odom(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.pos = np.array([pos.x, pos.y, pos.z])
        self.ori = np.array([ori.x, ori.y, ori.z, ori.w])

        twist = msg.twist.twist
        self.lin_vel = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
        self.ang_vel = np.array([twist.angular.x, twist.angular.y, twist.angular.z])
        self.notify_bspline_update()

    def update_bspline(self):
        r = R.from_quat(self.ori).inv()
        goal_local = r.apply(self.goal - self.pos)
        goal_local /= np.linalg.norm(goal_local)
        goal_local = torch.from_numpy(goal_local[None].astype(np.float32)).to(self.device)

        lin_vel = r.apply(self.lin_vel)
        ang_vel = r.apply(self.ang_vel)
        lin_vel = torch.from_numpy(lin_vel[None].astype(np.float32)).to(self.device)
        ang_vel = torch.from_numpy(ang_vel[None].astype(np.float32)).to(self.device)

        depth = transform.resize(self.depth, self.image_size)

        depth = np.clip(depth, 0, self.max_depth)
        depth /= self.max_depth
        depth = torch.from_numpy(depth[None, None].astype(np.float32),).to(self.device)

        with torch.no_grad():
            bspline_local = self.model(depth, goal_local, lin_vel, ang_vel)
        bspline_local = bspline_local[0].cpu().numpy()  # get rid of the batch size dim

        r_inv = R.from_quat(self.ori)
        bspline = r_inv.apply(bspline_local) + self.pos
        return bspline


if __name__ == '__main__':
    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    params_path = os.path.join(repo_path, "rslts", "LfD_3D_rslts", model_path, "params.json")
    model_path = os.path.join(repo_path, "rslts", "LfD_3D_rslts", model_path, "trained_models", model_fname)

    params = TrainingParams(params_path, train=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params.device = device

    encoder = Encoder(params).to(device)
    decoder = Decoder(params).to(device)
    model = Model(params, encoder, decoder).to(device)
    assert os.path.exists(model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print "[INFO] LfH policy loaded"

    predictor = Predictor(model, params)
    rospy.init_node('LfH_policy', anonymous=True)

    sub_goal = rospy.Subscriber("/waypoint_generator/waypoints", Path, predictor.update_goal)
    sub_depth = rospy.Subscriber("/pcl_render_node/depth", Image, predictor.update_depth, queue_size=1)
    sub_scan = rospy.Subscriber("/pcl_render_node/camera_pose", PoseStamped, predictor.update_depth_pose, queue_size=1)
    sub_scan = rospy.Subscriber("/visual_slam/odom", Odometry, predictor.update_odom, queue_size=1)
    bspline_pub = rospy.Publisher('/planning/bspline', Bspline, queue_size=1)

    traj_id = 0
    model_params = params.model_params
    knots_fitted = np.arange(model_params.knot_start, model_params.knot_end, 1) * model_params.knot_dt * 1.25
    num_control_pts = model_params.knot_end - model_params.knot_start - 3 - 1
    knots_fitted = list(knots_fitted.astype(np.float64))
    prev_msg_time = None
    print_time = rospy.get_time()
    while not rospy.is_shutdown():
        try:
            now = rospy.get_time()
            if (prev_msg_time is None or now - prev_msg_time >= update_dt) and predictor.need_update_bspline:
                traj_id += 1
                bspline_msg = Bspline()
                bspline_msg.start_time = rospy.Time.now()
                bspline_msg.order = 3
                bspline_msg.traj_id = traj_id
                bspline_msg.knots = knots_fitted
                pos_pts = []
                bspline = predictor.update_bspline()
                bspline = bspline[:num_control_pts]
                for control_pt in bspline:
                    pt = Point()
                    pt.x, pt.y, pt.z = control_pt.astype(np.float64)
                    pos_pts.append(pt)
                bspline_msg.pos_pts = pos_pts
                bspline_pub.publish(bspline_msg)

                if prev_msg_time is not None and now - print_time > 1:
                    print_time = now
                    print "[INFO] Update bspline, frequency = %d Hz" % int(1.0 / (now - prev_msg_time))
                predictor.need_update_bspline = False
                prev_msg_time = now
        except rospy.exceptions.ROSInterruptException:
            break
