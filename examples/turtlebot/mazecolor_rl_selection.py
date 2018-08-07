import gym
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import random
import numpy as np
import memory
import cv2
import qlearn

import sys, select, termios, tty
from itertools import *
from operator import itemgetter



def getPoints(image, red, green, blue):
    epsilon = 0.00001
    height, width, depth = image.shape
    sumY = 0
    countY = 0 
    mid = float(width/2)
    hint_vector = np.zeros(shape=(width,))
    for x in range(height):
        for y in range(width):
            if (image[x][y][0] == red) and (image[x][y][1] <= green) and (image[x][y][2] <= blue):
                hint_vector[y] = 1
                sumY += y
                countY += 1

    component = []
    sum_elements = 0
    count_elements = 0
    i = 0
    while i < width:
        if hint_vector[i] == 1:
            break
        i += 1
    while i < width:
        x = hint_vector[i]
        if x == 1:
            sum_elements += i
            count_elements += 1
            i += 1
        else:
            component.append(float(sum_elements)/count_elements)
            sum_elements = 0
            count_elements = 0
            while i < width:
                if hint_vector[i] == 1:
                    break
                i += 1

    if len(component) == 0:
        return 0
    else:
        return np.average(np.asarray(component))

def getTargetPoints(image):
    return getPoints(image, 102, 20, 20)

def getHintPoints(image):
    return getPoints(image, 255, 120, 120)

def get_image_action(width, y):
    if y == 0:
        return 10
    mid = width/2
    return int((y - mid) * 10 / mid + 10)

def checkDanger(laser):
    for distance in laser:
        if distance < 0.5:
            return True
    return False


def normalize(array):
    epsilon = 1e-4
    max = np.amax(array)
    min = np.amin(array)
    return (array - min)/(max - min + epsilon)

# ===== OBSTACLE AVOIDANCE =====
def get_expert_action(data):
    THRESHOLD = 3
    total_range = 180
    num_actions = 21
    num_ranges = len(data)
    
    range_angles = np.arange(len(data))
    ranges = data.copy()

    largest_gap = []
    count = 0
    while len(largest_gap) < 6:
        range_mask = (ranges > THRESHOLD)
        ranges_list = list(range_angles[range_mask])
        max_gap = 40

        gap_list = []

        # groupby: https://stackoverflow.com/questions/41411492/what-is-itertools-groupby-used-for
        # enumerate: adds a counter to an iterable: [0 x, 1 y, 2 z ...]
        for k, g in groupby(enumerate(ranges_list), lambda(i,x):i-x):
            gap_list.append(map(itemgetter(1), g))

        gap_list.sort(key=len)

        # gap_list: [[gap1], [gap2], ....]
        if len(gap_list) == 0:
            THRESHOLD -= 0.2
            continue
        largest_gap = gap_list[-1]
        THRESHOLD -= 0.2

    unit_angle = float(total_range)/(num_actions-1)
    mid_largest_gap = int((largest_gap[0] + largest_gap[-1]) / 2)
    mid_angle = mid_largest_gap * unit_angle
    turn_angle = mid_angle - total_range/2
    angular_z = 2.4/90 * turn_angle
    # 4.8 = 90 degree

    state = data.copy()    
    linear_x = np.amin([state[i] for i in largest_gap]) * 0.2
    angular_z = mid_largest_gap
    action = int(float(mid_largest_gap) / num_ranges * num_actions)
    return action

def threshold_laser(laser, value):
    binary_vector = laser >= value
    result = ''
    for x in binary_vector:
        if x: 
            result += '1'
        else: 
            result += '0'
    return result

if __name__ == '__main__':
    env = gym.make('GazeboTurtlebotMazeColor-v0')
    observation = env.reset()
    qlearn = qlearn.QLearn(actions=np.arange(2),
                alpha=0.2, gamma=0.8, epsilon=0.9)
    initial_epsilon = qlearn.epsilon
    epsilon_discount = 0.98

    total_episodes = 10000
    total_steps = 1000
    threshold_value = 2
    start_time = time.time()

    # Load trained q-values
    qlearn.load_q('/home/lntk/Desktop/q_values_900EP_2355.txt')
    qlearn.epsilon = 0.5

    for episode in range(total_episodes):
        done = False
        cumulated_reward = 0
        step_counter = 0

        observation = env.reset()

        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        for t in range(total_steps):

            step_counter += 1

            [image, laser] = observation
            hint_pos = getTargetPoints(image)
            if hint_pos == 0:
                hint_pos = getHintPoints(image)

            height, width, depth = image.shape
            image_action = get_image_action(width, hint_pos)
            laser_action = get_expert_action(laser)


            state = threshold_laser(laser, threshold_value)

            rl_choice = qlearn.chooseAction(state)

            if rl_choice == 1:
                action = 20 - image_action
            else:
                action = laser_action

            next_observation, reward, done, info = env.step(action)

            cumulated_reward += reward
            next_image, next_laser = next_observation
            next_state = threshold_laser(next_laser, threshold_value)
            
            qlearn.learn(state, rl_choice, reward, next_state)

            if not done:
                observation = next_observation
            else:
                break

        if episode%100==0:
            qlearn.save_q()

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP "+str(episode+1) + ":  " + str(step_counter)+"  timesteps" + " - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(round(cumulated_reward,2))+"     Time: %d:%02d:%02d" % (h, m, s))

    env.close()
