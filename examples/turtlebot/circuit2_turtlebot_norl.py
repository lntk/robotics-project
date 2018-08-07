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



# ===== OBSTACLE AVOIDANCE =====
def get_teacher_action(state):
    THRESHOLD = 4
    total_range = 200
    num_actions = 21
    num_ranges = len(state)
    
    range_angles = np.arange(len(state))
    ranges = state.copy()

    largest_gap = []
    count = 0
    while len(largest_gap) < 30:
        range_mask = (ranges > THRESHOLD)
        ranges_list = list(range_angles[range_mask])
        max_gap = 40
        gap_list = []

        for k, g in groupby(enumerate(ranges_list), lambda(i,x):i-x):
            gap_list.append(map(itemgetter(1), g))

        gap_list.sort(key=len)

        # gap_list: [[gap1], [gap2], ....]
        if len(gap_list) == 0:
            THRESHOLD -= 0.2
            continue
        largest_gap = gap_list[-1]
        THRESHOLD -= 0.2

    unit_angle = float(total_range)/20
    mid_largest_gap = int((largest_gap[0] + largest_gap[-1]) / 2) # index of the laser at the middle of the gap
    action = int(float(mid_largest_gap) / num_ranges * num_actions) # scale to action 0..20
    return action


if __name__ == '__main__':
    env = gym.make('GazeboCircuit2TurtlebotLidarNn-v0')
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
            action = get_teacher_action(observation)
            newObservation, reward, done, info = env.step(action)
            cumulated_reward += reward
            if done:
                break

            observation = newObservation

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: "+str(x+1)+" - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))

    env.close()
