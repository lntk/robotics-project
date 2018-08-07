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

class GazeboTurtlebotAroundCameraEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "TurtlebotWorldCamera_v1.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.name_model = 'mobile_base'
        
        #get model state service
        self.model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        self.physics_properties = rospy.ServiceProxy('gazebo/get_physics_properties', GetPhysicsProperties)

        self.action_space = spaces.Discrete(6)
        self.reward_range = (-np.inf, np.inf)
        self.count_loop = [0] * 50

        self.channel = 1
        self.width = 32
        self.height = 32
        self.num_action = 5

        self._seed()

    def calculate_observation(self,data):
        min_range = 0.21
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True

        return done

    def discretize_observation(self,data):

        """image_data = None
        cv_image = None
        n = 0
        while image_data is None:
            try:
                image_data = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "32FC1")
            except:
                n += 1
                if n == 10:
                    print "Depth error"
                    state = []
                    done = True
                    return state, done
        cv_image = np.array(cv_image, dtype=np.float32)
        cv2.normalize(cv_image, cv_image, 0, 1, cv2.NORM_MINMAX)
        cv_image = cv2.resize(cv_image, (160, 120))
        for i in range(120):
            for j in range(160):
                if np.isnan(cv_image[i][j]):
                    cv_image[i][j] = 0
                elif np.isinf(cv_image[i][j]):
                    cv_image[i][j] = 1
        """
        image_data = None
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
        cv_image = cv_image.reshape(self.width, self.height, self.channel)
        # print cv_image.shape
        #cv2.imshow("image", cv_image)
        #cv2.waitKey(3)
        return cv_image


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException, e:
            print ("/gazebo/unpause_physics service call failed")
        #print action
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.3
        if action == 0: #Turn 90
            vel_cmd.angular.z = math.pi/2
        elif action == 1: #Turn 45
            vel_cmd.angular.z = math.pi/4
        elif action == 2: #Turn 0
            vel_cmd.angular.z = 0
        elif action == 3: #Turn -45
            vel_cmd.angular.z = -math.pi/4
        elif action == 4: #Turn -90
            vel_cmd.angular.z = -math.pi/2
        elif action == 5: #Turn -90
            vel_cmd.angular.z = 0
            vel_cmd.linear.x*=-1

        self.vel_pub.publish(vel_cmd)

        if (action == 2 or action == 5):
            self.count_loop.append(0)
        else:
            self.count_loop.append(1)
        
        if (len(self.count_loop) > 50):
            self.count_loop = self.count_loop[1:]

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass
        done = self.calculate_observation(data)
        state = self.discretize_observation(data)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")
        #print(len(data.ranges))
        #print(data.ranges)
        #print state
        laser_len = len(data.ranges)
        left_sum = sum(data.ranges[laser_len-(laser_len/5):laser_len-(laser_len/10)]) #80-90
        right_sum = sum(data.ranges[(laser_len/10):(laser_len/5)]) #10-20

        center_detour = abs(right_sum - left_sum)/5

        if done:
            reward = -1
        else:
            if (action == 2):
                reward = 1 / float(center_detour+1)
            elif (self.count_loop.count(1) > 45):
                reward = -0.5
            else:
                reward = 0.5 / float(center_detour+1)

        return state, reward, done, {}

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
        done = self.calculate_observation(data)
        state = self.discretize_observation(data)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")

        return state
