#!/usr/bin/env python

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya
# Edits by: Lukasz Zmudzinski

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *


def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:
        # Create symbols
    	d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
    	a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
    	ap0, ap1, ap2, ap3, ap4, ap5, ap6 = symbols('ap0:7')
    	q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')

    	# Create Modified DH parameters
    	DH = {
    		ap0: 0,      a0: 0,      d1: 0.75,    q1: q1,
    		ap1: -pi/2,  a1: 0.35,   d2: 0,       q2: q2 - pi/2,
    		ap2: 0,      a2: 1.25,   d3: 0,       q3: q3,
    		ap3: -pi/2,  a3: -0.054, d4: 1.5,     q4: q4,
    		ap4: pi/2,   a4: 0,      d5: 0,       q5: q5,
    		ap5: -pi/2,  a5: 0,      d6: 0,       q6: q6,
    		ap6: 0,      a6: 0,      d7: 0.303,   q7: 0
            }

        # Define Modified DH Transformation matrix
    	def TF(ap, a, d, q):
    		return Matrix([[ cos(q), -sin(q), 0, a],
    			    [sin(q)*cos(ap), cos(q)*cos(ap), -sin(ap), -sin(ap)*d],
    			    [sin(q)*sin(ap), cos(q)*sin(ap), cos(ap), cos(ap)*d],
    			    [0, 0, 0, 1]]).subs(DH)

    	# Create individual transformation matrices
    	T01 = TF(ap0, a0, d1, q1)
    	T12 = TF(ap1, a1, d2, q2)
    	T23 = TF(ap2, a2, d3, q3)
    	T34 = TF(ap3, a3, d4, q4)
    	T45 = TF(ap4, a4, d5, q5)
    	T56 = TF(ap5, a5, d6, q6)
    	T6E = TF(ap6, a6, d7, q7)

    	T0E = T01 * T12 * T23 * T34 * T45 * T56 * T6E

    	# Extract rotation matrices from the transformation matrices
        def rot_roll(r):
            return Matrix([[1,      0,       0],
        		           [0, cos(r), -sin(r)],
        		           [0, sin(r),  cos(r)]])

        def rot_pitch(p):
            return Matrix([[cos(p),  0, sin(p)],
        			       [0,       1,      0],
        			       [-sin(p), 0, cos(p)]])

        def rot_yaw(y):
            return Matrix([[cos(y), -sin(y), 0],
        			       [sin(y),  cos(y), 0],
        			       [0,            0, 1]])

        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

    	    # Extract end-effector position and orientation from request
    	    # px,py,pz = end-effector position
    	    # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

    	    # Compensate for rotation discrepancy between DH parameters and Gazebo
            rot_ee = rot_yaw(yaw) * rot_pitch(pitch) * rot_roll(roll)

    	    rot_err = rot_yaw(yaw).subs(yaw, radians(180)) * rot_pitch(pitch).subs(pitch, radians(-90))
    	    rot_ee = rot_ee * rot_err
    	    #ROT_EE = ROT_EE.subs({'r': roll, 'p': pitch, 'y': yaw})

    	    EE = Matrix([[px],
    			[py],
    			[pz]])
            WC = EE - (0.303) * rot_ee[:,2]

    	    # Calculate joint angles using Geometric IK method

    	    theta1 = atan2(WC[1], WC[0])

    	    # Theta 2 and 3 calculations
    	    side_a = 1.501
    	    side_b = sqrt(pow((sqrt(WC[0] * WC[0] + WC[1] * WC[1]) - 0.35), 2) + pow((WC[2] - 0.75), 2))
    	    side_c = 1.25
    	    angle_a = acos((side_b * side_b + side_c * side_c - side_a * side_a) / (2 * side_b * side_c))
    	    angle_b = acos((side_a * side_a + side_c * side_c - side_b * side_b) / (2 * side_a * side_c))
    	    angle_c = acos((side_a * side_a + side_b * side_b - side_c * side_c) / (2 * side_a * side_b))
    	    theta2 = pi / 2 - angle_a - atan2(WC[2] - 0.75, sqrt(WC[0] * WC[0] + WC[1] * WC[1]) - 0.35)
    	    theta3 = pi / 2 - (angle_b + 0.036)

    	    # Theta 4 to 6 rotation matrixes
    	    R03 = T01[0:3, 0:3] * T12[0:3, 0:3] * T23[0:3, 0:3]
    	    R03 = R03.evalf(subs={q1: theta1, q2: theta2, q3: theta3})
    	    R36 = R03.inv("LU") * rot_ee
    	    theta4 = atan2(R36[2,2], -R36[0,2])
    	    theta5 = atan2(sqrt(R36[0,2]*R36[0,2] + R36[2,2]*R36[2,2]), R36[1,2])
    	    theta6 = atan2(-R36[1,1], R36[1,0])
                # Populate response for the IK request
                # In the next line replace theta1,theta2...,theta6 by your joint angle variables
    	    joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
    	    joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
