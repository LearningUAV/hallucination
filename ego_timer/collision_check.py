from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, ChannelFloat32
from cv_bridge import CvBridge

import sys
import rospy
import random
import argparse
import numpy as np


class CollisionChecker(object):
    """
    The input is a 480x640 frame, we use a 100x100 pixel square
    and check if object is in that square range
    """
    def __init__(self, init_pos):
        rospy.init_node("listener", anonymous=True)
        sub_odom = rospy.Subscriber("/visual_slam/odom", Odometry, self.update_odom, queue_size=1)
        sub_depth = rospy.Subscriber("/pcl_render_node/depth", Image, self.update_collision, queue_size=1)
        self.bridge = CvBridge()
        self.pos = init_pos
        self.is_collide = False
        self.distance = 0

    def update_odom(self, msg):
        pos = msg.pose.pose.position
        pos = np.array([pos.x, pos.y, pos.z])
        self.distance += np.linalg.norm(pos - self.pos)
        self.pos = pos

    def update_collision(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(cv_image)
        depth_center = depth[190:290, 270:370]
        collision_mask = (depth_center > 0) & (depth_center < 0.25)

        if np.sum(collision_mask) >= 200:
            self.is_collide = True


class WaypointFeed(object):

    def __init__(self, init_pos):
        print('[INFO] Initializing object...')
        self.waypoint_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        sub_robot = rospy.Subscriber("/visual_slam/odom", Odometry, self.update_status, queue_size=1)
        self.map_x_size = rospy.get_param("/pcl_render_node/map/x_size", 40.0)
        self.map_y_size = rospy.get_param("/pcl_render_node/map/y_size", 40.0)
        self.goal = None
        self.pos = init_pos

    def update_status(self, msg):
        pos = msg.pose.pose.position
        pos = np.array([pos.x, pos.y, pos.z])

        if self.goal is None or np.linalg.norm(pos - self.goal) < 2:
            self.publish_waypoint()

    def publish_waypoint(self):
        waypoint_msg = PoseStamped()
        waypoint_msg.header.seq = 1
        waypoint_msg.header.stamp = rospy.Time.now()
        waypoint_msg.header.frame_id = "world"
        while True:
            cur_x = random.uniform(self.map_x_size / -2, self.map_x_size / 2)
            cur_y = random.uniform(self.map_y_size / -2, self.map_y_size / 2)
            if self.goal is None:
                break
            if cur_x * self.goal[0] < 0 or cur_y * self.goal[1] < 0:
                break
        self.goal = np.array([cur_x, cur_y, 0])
        waypoint_msg.pose.position.x = cur_x
        waypoint_msg.pose.position.y = cur_y
        waypoint_msg.pose.position.z = 0
        waypoint_msg.pose.orientation.w = 1.0
        self.waypoint_pub.publish(waypoint_msg)
        print('[INFO] Waypoint published:', self.goal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='seed', type=int, default=0)

    args, unknown = parser.parse_known_args()
    print("[INFO] Using seed", args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    init_x = rospy.get_param("/quadrotor_simulator_so3/simulator/init_state_x", -18.0)
    init_y = rospy.get_param("/quadrotor_simulator_so3/simulator/init_state_y", 0.0)
    init_z = rospy.get_param("/quadrotor_simulator_so3/simulator/init_state_z", 0.0)
    init_pos = np.array([init_x, init_y, init_z])

    collision_checker = CollisionChecker(init_pos)
    time_pub = rospy.Publisher("collision_status", ChannelFloat32, queue_size=10)
    waypoint_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

    start_time = rospy.get_time()
    while not rospy.is_shutdown():
        duration = rospy.get_time() - start_time

        # only start timer once jackal starts moving
        if np.linalg.norm(collision_checker.pos - init_pos) < 0.1:
            duration = 0
            start_time = rospy.get_time()

        time_pub.publish(ChannelFloat32("duration+distance", [duration, collision_checker.distance]))

        if collision_checker.is_collide:
            time_pub.publish(ChannelFloat32("duration+distance", [-1.0, collision_checker.distance]))
            rospy.signal_shutdown("Done")
            exit()
