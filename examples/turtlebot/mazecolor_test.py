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
                continue    

    return centerY

def getTargetPoints(image):
    return getPoints(image, 102, 20, 20)

def getHintPoints(image):
    return getPoints(image, 255, 120, 120)

def getImageAction(width, y):
    if y == 0:
        return 0
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
def expert_action(data):
    THRESHOLD = 4
    total_range = 180
    num_actions = 21
    num_ranges = len(data)
    
    range_angles = np.arange(len(data))
    ranges = data.copy()

    largest_gap = []
    count = 0
    while len(largest_gap) < 30:
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

if __name__ == '__main__':
    env = gym.make('GazeboTurtlebotMazeColor-v0')
    observation = env.reset()

    while True:
        [image, laser] = observation

        x, y = getTargetPoints(image)
        if x == 0 and y == 0:
            x, y = getHintPoints(image)

        height, width, depth = image.shape
        imageAction = getImageAction(height, width, x, y)

        min_laser = np.amin(laser)
        if imageAction == 100:
            action = expert_action(laser)
            # print("expert action: " + str(action))
        else:
            if min_laser < 0.8:
                action = expert_action(laser)
                # print("expert action: " + str(action))
            else:
                action = 20 - imageAction
                # print("image action: " + str(action))

        observation, reward, done, info = env.step(action)
        lastAction = action
        if done:
            observation = env.reset()

    env.close()
