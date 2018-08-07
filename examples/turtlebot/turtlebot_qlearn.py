#!/usr/bin/env python
import gym
import gym_gazebo
import time
import numpy
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import qlearn
import numpy as np
import sys, select, termios, tty
from itertools import *
from operator import itemgetter

def binarize_observation(observation):
    min_range = 0.7
    binary_ranges = ''
    for x in observation:
        if min_range > x > 0:
            binary_ranges += '0'
        else:
            binary_ranges += '1'
    return binary_ranges

# ===== OBSTACLE AVOIDANCE =====
def expert_action(observation):
    THRESHOLD = 4
    total_range = 180
    num_actions = 7
    num_ranges = len(observation)
    
    range_angles = np.arange(len(observation))
    ranges = observation.copy()

    largest_gap = []
    count = 0
    while len(largest_gap) < 5:
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

    unit_angle = float(total_range)/(num_ranges - 1)
    mid_largest_gap = int((largest_gap[0] + largest_gap[-1]) / 2)
    mid_angle = mid_largest_gap * unit_angle
    turn_angle = mid_angle - total_range/2
    angular_z = 2.4/90 * turn_angle
    # 4.8 = 90 degree

    state = observation.copy()    
    linear_x = np.amin([state[i] for i in largest_gap]) * 0.2
    angular_z = mid_largest_gap
    action = int(float(mid_largest_gap) / num_ranges * num_actions)
    return action

if __name__ == '__main__':

    env = gym.make('GazeboCircuit2TurtlebotLidar-v0')

    last_time_steps = numpy.ndarray(0)

    qlearn = qlearn.QLearn(actions=np.arange(7),
                    alpha=0.2, gamma=0.8, epsilon=0.9)

    initial_epsilon = qlearn.epsilon

    epsilon_discount = 0.9986

    start_time = time.time()
    total_episodes = 10000
    highest_reward = 0
    teach_episodes = 2

    print("Teaching...")
    for x in range(teach_episodes):
        done = False
        observation = env.reset()

        #render() #defined above, not env.render()
        step_counter = 0

        for i in range(200):
            step_counter += 1
            action = expert_action(observation)
            binarized_observation = binarize_observation(observation)
            newObservation, reward, done, info = env.step(action)
            binarized_new_observation = binarize_observation(newObservation)

            qlearn.learn(binarized_observation, action, reward, binarized_new_observation)

            if not(done):
                observation = newObservation
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break


    print("Learning...")
    for x in range(total_episodes):
        done = False

        cumulated_reward = 0 #Should going forward give more reward then L/R ?

        observation = env.reset()

        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        step_counter = 0

        for i in range(1500):
            step_counter += 1
            binarized_observation = binarize_observation(observation)
            action = qlearn.chooseAction(binarized_observation)
            newObservation, reward, done, info = env.step(action)
            binarized_new_observation = binarize_observation(newObservation)
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            qlearn.learn(binarized_observation, action, reward, binarized_new_observation)
            if not(done):
                observation = newObservation
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break

        if x%100==0:
            qlearn.save_q()

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP "+str(x+1) + ":  " + str(step_counter)+"  timesteps" + " - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(round(cumulated_reward,2))+"     Time: %d:%02d:%02d" % (h, m, s))

    #Github table content
    print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |")

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    # env.monitor.close()
    env.close()
