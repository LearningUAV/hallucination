#!/usr/bin/env python

import os
import roslaunch
import rospy
import rosnode
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import ChannelFloat32


uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)

# whenever listener node receives a message, update duration
curr_duration = 0.0
curr_distance = 0.0
trial_running = True


def callback(msg):
    global curr_duration
    global trial_running
    global curr_distance

    duration, distance = msg.values
    if duration == -1.0:
        trial_running = False
    else:
        curr_duration = duration
        curr_distance = distance


c_num = 400
p_num = 400
random_forest = True
use_LfH = True
rospy.init_node("duration_listener", anonymous=True)
rospy.Subscriber("collision_status", ChannelFloat32, callback)


for i in range(1):
    curr_duration = 0.0
    curr_distance = 0.0
    trial_running = True

    args_list = ["run_in_sim_test.launch", "c_num:=" + str(c_num), "seed:=" + str(i),
                 "bspline_topic:=" + str("/planning/bspline_truth" if use_LfH else "/planning/bspline"),
                 "use_lfh:=" + str("true" if use_LfH else "false"),
                 "random_forest:=" + str("true" if random_forest else "false"),
                 ]
    lifelong_args = args_list[1:]
    launch_files = [(roslaunch.rlutil.resolve_launch_arguments(args_list)[0], lifelong_args)]

    # launch the launch file
    parent = roslaunch.parent.ROSLaunchParent(uuid, launch_files)
    parent.start()

    trial_start = rospy.get_time()

    while trial_running:
        if rospy.get_time() - trial_start > 60.0 * 15:
            break

    parent.shutdown()
    fout = open("{}.txt".format("LfH" if use_LfH else "ego"), "a")
    if not trial_running:
        fout.write("%d\n%f\n%f\n" % (i, curr_duration, curr_distance))
    fout.close()
    print "Finished %d" % i
