import gym
import rospy
import roslaunch
import time
import math
import numpy as np
import cv2
import random

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError


from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, GetPhysicsProperties, SetModelState
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class GazeboTurtlebotComplicatedMazeEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "TurtlebotComplicatedMaze_v0.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.name_model = 'mobile_base'
        # self.name_target = 'TargetCylinder'
        self.name_hint = 'Hint'
        
        #get model state service
        self.model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        self.physics_properties = rospy.ServiceProxy('gazebo/get_physics_properties', GetPhysicsProperties)

        self.action_space = spaces.Discrete(19)
        self.reward_range = (-np.inf, np.inf)
        self.count_loop = [0] * 50

        self.target_pos = [[-3, -3.5], [-3.5, 0], [3.5, -3.5], [3, 1.5]]
        self.rand_target = 0
        self.hint = [[3,1], [4,3]]

        self.set_model = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)

        rospy.wait_for_service('gazebo/set_model_state')

        self.channel = 1
        self.width = 32
        self.height = 32
        self.num_action = 7
        self.last_distance = 10000
        self.old_pose = 0
        self.num_state = 105
        self.num_target = 4

        self._seed()

    def calculate_observation(self,data, distance):
        min_range = 0.21
        target_min_range = 0.21
        done = 0
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = 1
        if (distance < target_min_range):
            done = 2
        return done

    def setTarget(self):
        state = ModelState()
        state.model_name = self.name_target
        state.reference_frame = "world"
        state.pose.position.x = self.target_pos[self.rand_target][0]
        state.pose.position.y = self.target_pos[self.rand_target][1]
        self.set_model(state)

    def discretize_observation(self,data, vel_cmd, pos, new_pose, twist, num_step, distance):
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
        observation.append(twist.linear.x)
        observation.append(twist.angular.z)
        observation.append(self.target_pos[self.rand_target][0] - pos.x)
        observation.append(self.target_pos[self.rand_target][1] - pos.y)
        observation.append(distance)
        #observation.append(num_step)
        # print vel_cmd
        # print twist
        # observation = [self.old_pose.position.x, self.old_pose.position.y, self.old_pose.orientation.x,  self.old_pose.orientation.y,  self.old_pose.orientation.z,  self.old_pose.orientation.w, new_pose.position.x, new_pose.position.y, new_pose.orientation.x, new_pose.orientation.y, new_pose.orientation.z, new_pose.orientation.w]
        """image_data = None
        cv_image = None
        n = 0
        while image_data is None:
            try:
                image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            except:
                n += 1
                if n == 10:
                    print "Camera error"
                    state = []
                    done = True
                    return state
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.height, self.width))
        cv2.imshow("image", cv_image)
        cv2.waitKey(3)
        cv_image = cv_image.reshape(1, 32, 32, 1)
        # print cv_image.shape"""
        return np.asarray(observation)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def calculate_distance(self, x, y, a, b):
        return math.sqrt((x-a)*(x-a) +(y-b) * (y-b))

    def calculate_reward(self, done, pos, vel_cmd, num_step):
        min_range = 0.21
        target_min_range = 0.21
        reward = -1
        distance = self.calculate_distance(pos.x, pos.y, self.target_pos[self.rand_target][0], self.target_pos[self.rand_target][1])
        reward = (self.last_distance - distance) * 100
        if done == 2:
            reward = 200
        elif done ==1:
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
        num_step = action[1]
        action = action[0]
        vel_cmd = Twist()
        max_linear_speed = 2
        linear_vel = (math.floor(action/7))*max_linear_speed*0.1 #(0, 0.2, 0.4)
        vel_cmd.linear.x = 0.2
        max_ang_speed = 3.3
        ang_vel = (action - 3)*max_ang_speed*0.1 #from (-1 to + 1)

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
        distance = math.sqrt((pos.x - self.target_pos[self.rand_target][0])*(pos.x - self.target_pos[self.rand_target][0]) + (pos.y - self.target_pos[self.rand_target][1]) * (pos.y - self.target_pos[self.rand_target][1]))

        done = self.calculate_observation(data, distance)
        state = self.discretize_observation(data, vel_cmd, pos, new_pose, twist, num_step, distance)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")

        reward = self.calculate_reward(done, pos, vel_cmd, num_step)

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
        info = {"target": self.rand_target, "touched": done}
        if done != 0:
            done = True
        else:
            done = False
        return state, reward, done, info

    def _reset(self):

        cv2.destroyAllWindows()
        # Resets the state of the environment and returns an initial observation.
        self.rand_target = np.random.randint(self.num_target)
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
        self.last_distance = math.sqrt((pos.x - self.target_pos[self.rand_target][0])*(pos.x - self.target_pos[self.rand_target][0]) + (pos.y - self.target_pos[self.rand_target][1]) * (pos.y - self.target_pos[self.rand_target][1]))
        vel_cmd = Twist()
        state = self.discretize_observation(data, vel_cmd, pos, self.old_pose, twist, 0, self.last_distance)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")
        rospy.wait_for_service('gazebo/set_model_state')
        # self.setTarget()

        return state
