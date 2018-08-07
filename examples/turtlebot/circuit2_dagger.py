#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
import gym_gazebo
import os
import random
import numpy as np
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import keras

import sys, select, termios, tty
import random
from itertools import *
from operator import itemgetter


class DeepQ:
    def __init__(self, inputs, outputs, learningRate):
        self.input_size = inputs
        self.output_size = outputs
        self.learningRate = learningRate
        self.states = np.empty((0,self.input_size), dtype = np.float64)
        self.actions = np.empty((0,self.output_size), dtype = np.float64)

    def initNetworks(self, hiddenLayers):
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.model = model

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        model = Sequential()
        if len(hiddenLayers) == 0:
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform'))
            model.add(Activation("linear"))
        else :
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform'))
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))
            model.add(Dropout(0.25))

            for index in range(1, len(hiddenLayers)):
                # print("adding layer "+str(index))
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, init='lecun_uniform'))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))

                model.add(Dropout(0.25))
            model.add(Dense(self.output_size, init='lecun_uniform'))
            model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        model.summary()
        return model

    def getAction(self, state):
        predicted = self.model.predict(state.reshape(1,len(state)))
        return np.argmax(predicted[0])

    def selectAction(self, state, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getAction(state)
        return action

    def addMemory(self, state, action):
        state_copy = np.array([state.copy()])
        state_copy = state_copy / np.amax(state_copy)
        self.states = np.append(self.states, state_copy, axis=0)
        action_category = keras.utils.to_categorical(action, num_classes=11)
        action_category = action_category.reshape((1, len(action_category)))
        self.actions = np.append(self.actions, action_category, axis=0)
        # print(state_copy)
        # print(action)

    def learn(self):
        # print(self.states.shape)
        # print(self.actions.shape)
        self.model.fit(self.states, self.actions, batch_size=64, nb_epoch=100, verbose=1)

    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())

# ===== OBSTACLE AVOIDANCE =====
def get_expert_action(data):
    THRESHOLD = 3
    total_range = 180
    num_actions = 11
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

if __name__ == '__main__':
    env = gym.make('GazeboCircuit2TurtlebotLidarNn-v0')

    continue_execution = True

    weights_path = '/home/lntk/Desktop/turtle_dagger_dqn.h5'

    if not continue_execution:
        epochs = 1000
        steps = 1000
        explorationRate = 1
        learningRate = 0.00025
        network_inputs = 20
        network_outputs = 11
        network_structure = [15, 15]
        current_epoch = 0

        deepQ = DeepQ(network_inputs, network_outputs, learningRate)
        deepQ.initNetworks(network_structure)
        # env.monitor.start(outdir, force=True, seed=None)
    else:
        epochs = 1000
        steps = 1000
        explorationRate = 0.5
        learningRate = 0.00025
        network_inputs = 20
        network_outputs = 11
        network_structure = [15, 15]
        current_epoch = 0
        deepQ = DeepQ(network_inputs, network_outputs, learningRate)
        deepQ.initNetworks(network_structure)
        deepQ.loadWeights(weights_path)
        print ("Import sucess.")



    # ===== INITIAL PHASE =====
    num_teach_episode = 5
    num_step = 500
    print("Initial Phase Started.")
    for episode in range(num_teach_episode):
        observation = env.reset()
        for t in range(num_step):
            action = get_expert_action(observation)
            action = deepQ.selectAction(observation, 0)
            
            newObservation, reward, done, info = env.step(action)
            deepQ.addMemory(observation, action)
            observation = newObservation
            if (t >= num_step - 1):
                done = True

            if done:        
                print("Teaching Episode " + str(episode + 1) + " completed.")
                break
    deepQ.learn()

    # ===== TRAINING PHASE =====
    print("Traning Phase Started.")
    num_iteration = 5
    num_episode = 20
    num_step = 500

    for i in range(num_iteration):
        for episode in range(num_episode):
            observation = env.reset()

            for t in range(num_step):
                expert_action = get_expert_action(observation)
                robot_action = deepQ.selectAction(observation, explorationRate)
                newObservation, reward, done, info = env.step(robot_action)
                    
                deepQ.addMemory(observation, expert_action)
                observation = newObservation

                if (t >= num_step - 1):
                    print ("reached the end! :D")
                    done = True

                if done:         
                    print ("EP "+str(episode)+ " - {} timesteps".format(t+1))
                    break

            explorationRate *= 0.98 #epsilon decay
            # explorationRate -= (2.0/epochs)
            explorationRate = max (0.05, explorationRate)
        deepQ.learn()
        print ("Saving model ...")  
        deepQ.saveModel('/home/lntk/Desktop/turtle_dagger_dqn'+'.h5')

    env.close()
