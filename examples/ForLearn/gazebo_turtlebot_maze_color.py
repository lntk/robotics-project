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
from gazebo_msgs.srv import GetModelState, GetPhysicsProperties, SetModelState
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan

from gym.utils import seeding

class GazeboTurtlebotMazeColorEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "MazeColor.launch")
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.name_model = 'mobile_base'
        self.name_target = 'Target'
        self.name_hint = 'Hint'

        self.target_pos = [[-0.25, -2], [6.5, -1.75], [6.5, 1.75], [8.5, 1.5]]
        self.hint_pos = []
        hint_target = [[1.5, 0], [1.5, -2], [3, -2], [3, -3.75], [-0.25, -3.75]]
        self.hint_pos.append(hint_target);
        hint_target = [[1.5, 0], [4.75, 0], [4.75, -3.75], [8, -3.75], [8, 0], [6.5, 0]]
        self.hint_pos.append(hint_target)
        self.hint_pos.append(hint_target)
        hint_target = [[1.5, 0], [1.5, 2], [5, 2], [5, 3.5], [8.5, 3.5]]
        self.hint_pos.append(hint_target)
        
        self.set_model = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)

        rospy.wait_for_service('gazebo/set_model_state')

        self.num_target = np.random.randint(3)

        self.num_hint = len(self.hint_pos[self.num_target])
        self.setTarget()


        #get model state service
        self.model_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        self.physics_properties = rospy.ServiceProxy('gazebo/get_physics_properties', GetPhysicsProperties)
        self.turtlebot_state = self.model_state(self.name_model, "world")
        self.action_space = spaces.Discrete(6)
        self.reward_range = (-np.inf, np.inf)
        self.count_loop = [0] * 50

        self.channel = 3
        self.width = 32
        self.height = 32
        self.num_action = 5
        self.num_state = [[32, 32, 3], 100, 2]

        self._seed()

    def setTarget(self):
        state = ModelState()
        state.model_name = self.name_target
        state.reference_frame = "world"
        state.pose.position.x = self.target_pos[self.num_target][0]
        state.pose.position.y = self.target_pos[self.num_target][1]
        self.set_model(state)
        for i in range(self.num_hint):
            state.model_name = self.name_hint + str(i)
            state.pose.position.x = self.hint_pos[self.num_target][i][0]
            state.pose.position.y = self.hint_pos[self.num_target][i][1]
            self.set_model(state)

    def calculate_observation(self, data, pos):
        min_range = 0.21
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        if (self.calculate_distance(pos.x, pos.y, self.target_pos[self.num_target][0], self.target_pos[self.num_target][1]) < min_range):
            done = True
        return done

    def calculate_distance(self, x, y, a, b):
        return math.sqrt((x-a)*(x-a) +(y-b) * (y-b))

    def calculate_reward(self, done, pos):
        min_range = 0.21
        reward = -1
        if (done):
            if (self.calculate_distance(pos.x, pos.y, self.target_pos[self.num_target][0], self.target_pos[self.num_target][1]) < min_range):
                reward = 200
            else:
                reward = -200
        else:
            for i in range(self.num_hint):
                if (self.calculate_distance(pos.x, pos.y, self.hint_pos[self.num_target][i][0], self.hint_pos[self.num_target][i][1]) < min_range):
                    reward = 10
        return reward

    def discretize_observation(self,data):
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
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.height, self.width))
        cv_image = cv_image.reshape(self.width, self.height, self.channel)
        # print cv_image.shape
        # cv2.imshow("image", cv_image)
        # cv2.waitKey(3)
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


        #######Action defination######
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.2
        max_ang_speed = 0.3
        ang_vel = (action-2)*max_ang_speed*0.1 #from (-0.33 to + 0.33)
        vel_cmd.angular.z = ang_vel
        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        ########Getting state#########
        pos = self.model_state(self.name_model, "world").pose.position
        done = self.calculate_observation(data, pos)
        image = self.discretize_observation(data)
        laser = data.ranges
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")
        

        ########Reward calculation#####
        reward = self.calculate_reward(done, pos)
        # print reward

        return [image, np.asarray(laser), np.asarray(pos)], reward, done, {}

    def _reset(self):

        #cv2.destroyAllWindows()
        # Resets the state of the environment and returns an initial observation.
        # rospy.wait_for_service('/gazebo/reset_simulation')
        # try:
        #     #reset_proxy.call()
        #     self.reset_proxy()
        # except rospy.ServiceException, e:
        #     print ("/gazebo/reset_simulation service call failed")

        rospy.wait_for_service('gazebo/set_model_state')
        state = ModelState()
        state.model_name = self.name_model
        state.reference_frame = "world"
        state.pose= self.turtlebot_state.pose
        state.twist = self.turtlebot_state.twist
        self.set_model(state)

        # self.num_target = np.random.randint(3)

        # self.num_hint = len(self.hint_pos[self.num_target])
        # self.setTarget()
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
        pos = self.model_state(self.name_model, "world").pose.position
        done = self.calculate_observation(data, pos)
        image = self.discretize_observation(data)
        laser = data.ranges
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException, e:
            print ("/gazebo/pause_physics service call failed")

        return [image, np.asarray(laser), np.asarray(pos)]
