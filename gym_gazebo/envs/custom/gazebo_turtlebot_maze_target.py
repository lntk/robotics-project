import gym
import rospy
import roslaunch
import time
import math
import numpy as np
import cv2

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError


from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetPhysicsProperties
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class GazeboTurtlebotMazeTargetEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "TurtlebotMazeTarget_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.name_model = 'mobile_base'
        
        #get model state service
        self.model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        self.physics_properties = rospy.ServiceProxy('gazebo/get_physics_properties', GetPhysicsProperties)

        self.action_space = spaces.Discrete(19)
        self.reward_range = (-np.inf, np.inf)
        self.count_loop = [0] * 50

        self.target = [4, 5]
        self.hint = [[3,1], [4,3]]

        self.channel = 1
        self.width = 32
        self.height = 32
        self.num_action = 21
        self.last_distance = 10000
        self.old_pose = 0
        self.num_state = [100, 2]

        self._seed()

    def calculate_observation(self,data, distance):
        min_range = 0.21
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        if (distance < min_range):
            done = True
        return done

    def discretize_observation(self,data, vel_cmd, pos, new_pose, twist):
        observation = []
        mod = len(data.ranges)/100
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf') or np.isinf(data.ranges[i]):
                    observation.append(21)
                elif np.isnan(data.ranges[i]):
                    observation.append(0)
                else:
                    observation.append(data.ranges[i])
        # observation.append(vel_cmd.linear.x)
        # observation.append(vel_cmd.angular.z)
        # observation.append(self.target[0])
        # observation.append(self.target[1])
        """observation = [self.old_pose.position.x, self.old_pose.position.y, self.old_pose.orientation.x,  self.old_pose.orientation.y,  self.old_pose.orientation.z,  self.old_pose.orientation.w, new_pose.position.x, new_pose.position.y, new_pose.orientation.x, new_pose.orientation.y, new_pose.orientation.z, new_pose.orientation.w]"""
        return observation

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def calculate_distance(self, x, y, a, b):
        return math.sqrt((x-a)*(x-a) +(y-b) * (y-b))

    def calculate_reward(self, done, pos):
        min_range = 0.21
        reward = -1
        distance = self.calculate_distance(pos.x, pos.y, self.target[0], self.target[1])
        reward = (self.last_distance - distance) * 10
        if (done):
            if (distance < min_range):
                reward = 200
            else:
                reward = -200
        
        """else:
            for i in range(2):
                if (self.calculate_distance(pos.x, pos.y, self.hint[i][0], self.hint[i][1]) < min_range):
                    reward = 10"""
        
        return reward
    def _step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")
        #print action
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.2
        max_ang_speed = 0.3
        ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)

        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        pos = self.model_state(self.name_model, "world").pose
        twist = self.model_state(self.name_model, "world").twist
        new_pose = pos
        pos = pos.position
        #print pos
        distance = math.sqrt((pos.x - self.target[0])*(pos.x - self.target[0]) + (pos.y - self.target[1]) * (pos.y - self.target[1]))

        done = self.calculate_observation(data, distance)
        state = self.discretize_observation(data, vel_cmd, pos, new_pose, twist)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")

        reward = self.calculate_reward(done, pos)

        """if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
        else:
            reward = -200"""
        self.last_distance = distance
        self.old_pose = new_pose
        #print(len(data.ranges))
        #print(data.ranges)

        return [np.asarray(state), np.asarray([self.target[0] - pos.x, self.target[0] - pos.y])], reward, done, {}

    def _reset(self):

        #cv2.destroyAllWindows()
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except rospy.ServiceException, e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")
        #read laser data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        pos = self.model_state(self.name_model, "world").pose
        twist = self.model_state(self.name_model, "world").twist
        self.old_pose = pos
        pos = pos.position
        self.last_distance = math.sqrt((pos.x - self.target[0])*(pos.x - self.target[0]) + (pos.y - self.target[1]) * (pos.y - self.target[1]))
        vel_cmd = Twist()
        state = self.discretize_observation(data, vel_cmd, pos, self.old_pose, twist)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")

        return [np.asarray(state), np.asarray([self.target[0] - pos.x, self.target[0] - pos.y])]
