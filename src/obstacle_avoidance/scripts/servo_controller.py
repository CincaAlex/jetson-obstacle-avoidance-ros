#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def callback(msg):
    if msg.data == "TURN_LEFT":
        angle = 30
    elif msg.data == "TURN_RIGHT":
        angle = 150
    elif msg.data == "FORWARD":
        angle = 90
    else:
        angle = 90

    rospy.loginfo(f"Servo angle: {angle}")

rospy.init_node("servo_controller")
rospy.Subscriber("/obstacle_decision", String, callback)
rospy.spin()
