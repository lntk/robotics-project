import gym
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class GazeboObstacleMazeEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "GazeboTurtlebotObstacleMaze.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.reward_range = (-np.inf, np.inf)

        self._seed()

    def calculate_observation(self,data):
        min_range = 0.2
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return data.ranges,done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # # ===== SET UP FOR LMAZE ======
        # max_ang_speed = 1.2
        # ang_vel = (action-9)*max_ang_speed*0.1 #from (-0.33 to + 0.33)
        # vel_cmd = Twist()
        # vel_cmd.linear.x = 
        # vel_cmd.angular.z = ang_vel
        # self.vel_pub.publish(vel_cmd)
        # time.sleep(0.5)
        # vel_cmd = Twist()
        # vel_cmd.linear.x = 0.5
        # vel_cmd.angular.z = ang_vel
        # self.vel_pub.publish(vel_cmd)

        # # ===== ACTION TO OBSTACLE AVOIDANCE =====
        # max_ang_speed = 0.3
        # linear_x, angular_z = action
        # ang_vel = (angular_z-50)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        # vel_cmd = Twist()
        # vel_cmd.linear.x = linear_x
        # vel_cmd.angular.z = ang_vel
        # self.vel_pub.publish(vel_cmd)


        # ===== SETUP FOR NORL =====
        # 4.8 = 90 degree
        # linear_x, angular_z = action
        # vel_cmd = Twist()
        # vel_cmd.linear.x = linear_x
        # # vel_cmd.angular.z = (angular_z - 6) * 0.72
        # vel_cmd.angular.z = angular_z
        # print(action)
        # self.vel_pub.publish(vel_cmd)
        # time.sleep(2)

        # ===== ACTION FOR DQN =====
        max_ang_speed = 0.6
        ang_vel = (action - 10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.2
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)



        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state,done = self.calculate_observation(data)
        # total = sum(state)

        # # ===== REWARD FOR LMAZE =====
        # count = 0
        # max_count = 0
        # for dist in state:
        #     if dist < 0.75:
        #         count += 1
        #     else:
        #         max_count = max(max_count, count)
        #         count = 0

        # ===== CALCULATE REWARD =====
        laser_ranges = np.asarray(state)
        # laser_ranges = np.asarray([min(3, x) for x in laser_ranges])
        # variance_range = np.var(laser_ranges)
        # penalty = 0
        # for x in laser_ranges:
        #     if x < 0.5:
        #         penalty -= 1

        if not done:
            reward = 1
        else:
            reward = -200

        return np.asarray(state), reward, done, {}

        # ===== OBSTACLE AVOIDANCE RETURN =====
        # return data, reward, done, {}

    def _reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.calculate_observation(data)

        return np.asarray(state)

        # ===== OBSTACLE AVOIDANCE RETURN =====
        # return data
