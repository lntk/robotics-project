#!/usr/bin/env python
import gym
import gym_gazebo
import time
import numpy as np
import random
import time
import sys, select, termios, tty
import random
from itertools import *
from operator import itemgetter

# ===== MOVE BY KEY =====
moveBindings = {
    'a': 0,
    's': 1,
    'd': 2,
    'f': 3,
    'g': 4,
    'h': 5,
    'j': 6
}

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def chooseAction(state):
    sub_state = np.asarray(state)
    sub_state = sub_state[4:16]
    max_angular = np.argmax(sub_state)
    min_angular = np.argmin(sub_state)
    mean = np.sum(sub_state) / sub_state.shape[0]

    max_in_state = max_angular + 4
    left_nb = max_in_state - 1
    right_nb = max_in_state + 1
    min_left = np.amin(state[(left_nb - 2) : (left_nb + 2)])
    min_right = np.amin(state[(right_nb - 2) : (right_nb + 2)]) 
    
    if state[left_nb] < state[right_nb]:
        max_angular += 2
        if min_left > min_right:
            max_angular -= 1
    else:
        max_angular -= 1
        if min_right > min_left:
            max_angular += 1
    max_angular = max(max_angular, 0)
    max_angular = min(max_angular, 12)
    # print(max_angular)
    max_linear = sub_state[max_angular] - 0.3
    # return max_linear, 12 - max_angular
    return 0.3, max_angular


# ===== IMAGE ACTION =====
def checkDanger(data):
    ranges = np.asarray(data)
    min_range = np.amin(ranges)
    if min_range < 0.5:
        return True
    return False

def getPoints(image, red, green, blue):    
    height, width, depth = image.shape
    sumX = 0
    sumY = 0
    countX = 0
    countY = 0 
    mid = float(width/2)
    for x in range(height):
        for y in range(width):
            if (image[x][y][0] == red) and (image[x][y][1] <= green) and (image[x][y][2] <= blue):
                sumX += x
                weightY = abs(float(y - mid)) / mid
                sumY += y * weightY
                countX += 1
                countY += weightY

    if countX == 0 and countY == 0:
        return np.asarray([0, 0])

    centerX = float(sumX)/countX
    centerY = float(sumY)/countY

    return np.asarray([centerX, centerY])

def getTargetPoints(image):
    return getPoints(image, 102, 20, 20)

def getHintPoints(image):
    return getPoints(image, 255, 120, 120)

def getImageAction(height, width, x, y):
    if x == 0 and y == 0:
        return 100
    mid = width/2
    return np.rint((y - mid) * 10 / mid + 10)
# ===== ~ =====


# ===== OBSTACLE AVOIDANCE =====
def LaserScanProcess(data):
    linear_x = 1 
    THRESHOLD = 1.5
    PI = 3.14
    Kp = 0.05
    angular_z = 0
    
    range_angles = np.arange(len(data))
    ranges = np.array(data)
    range_mask = (ranges > THRESHOLD)
    ranges = list(range_angles[range_mask])
    max_gap = 40

    gap_list = []

    # groupby: https://stackoverflow.com/questions/41411492/what-is-itertools-groupby-used-for
    # enumerate: adds a counter to an iterable: [0 x, 1 y, 2 z ...]
    for k, g in groupby(enumerate(ranges), lambda(i,x):i-x):
        gap_list.append(map(itemgetter(1), g))

    gap_list.sort(key=len)

    # gap_list: [[gap1], [gap2], ....]
    largest_gap = gap_list[-1]

    unit_angle = float(270)/20
    mid_largest_gap = int((largest_gap[0] + largest_gap[-1]) / 2)
    mid_angle = mid_largest_gap * unit_angle
    turn_angle = mid_angle - 270/2
    angular_z = 2.4/90 * turn_angle
    # print(largest_gap)

    return mid_largest_gap


if __name__ == '__main__':
    env = gym.make('GazeboTurtlebotMazeColor-v0')
    num_episodes = 200
    num_steps = 2000
    start_time = time.time()
    settings = termios.tcgetattr(sys.stdin)

    for x in range(num_episodes):
        observation = env.reset()
        cumulated_reward = 0

        for i in range(num_steps):
            # # ===== MOVE BY KEY =====
            # while True:
            #     key = getKey()
            #     if key in moveBindings.keys():
            #         angular_z = moveBindings[key]
            #         break
            # action = [0, angular_z]
            
            # action = chooseAction(observation)
            image, laser = observation

            # ===== GET LASER ACTION =====
            laserAction = LaserScanProcess(laser)
            
            # ===== GET IMAGE ACTION =====
            x, y = getTargetPoints(image)
            if x == 0 and y == 0:
                x, y = getHintPoints(image)

            height, width, depth = image.shape
            imageAction = getImageAction(height, width, x, y)

            # ===== TAKE ACTION =====
            if checkDanger(laser):
                print("in danger")
                action = [laserAction, True]
            else:
                if imageAction == 100:
                    action = [laserAction, True]
                else:
                    action = [imageAction, False]

            print(imageAction)
            print(laserAction)
            print(action)
            # time.sleep(3)
            newObservation, reward, done, info = env.step(action)
            cumulated_reward += reward
            if done:
                break

            observation = newObservation

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: "+str(x+1)+" - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))

    env.close()
