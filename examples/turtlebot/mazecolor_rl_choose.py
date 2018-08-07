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

from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import keras
import memory

import sys, select, termios, tty
from itertools import *
from operator import itemgetter

class LaserNet:
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


class SelectionNet:
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        self.input_size = inputs
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate

    def initNetworks(self, hiddenLayers):
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.model = model

        targetModel = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.targetModel = targetModel

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        model = Sequential()
        model.add(Dense(10, input_dim=20, kernel_initializer='normal', activation='relu'))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        model.summary()
        return model

    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, state):
        predicted = self.model.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getTargetQValues(self, state):
        #predicted = self.targetModel.predict(state.reshape(1,len(state)))
        predicted = self.targetModel.predict(state.reshape(1,len(state)))

        return predicted[0]

    def getAction(self, state):
        state_copy = np.array([state.copy()])
        state_copy = state_copy / np.amax(state_copy)
        predicted = self.model.predict(state_copy.reshape(1,len(state)))
        # predicted = self.model.predict(state.reshape(1,len(state)))
        # print(predicted)
        return int(predicted[0] > 0.5)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else :
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    def selectAction(self, state, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getAction(state)
        return action

    def addMemory(self, state, action, reward, newState, isFinal):
        state_copy = np.array([state.copy()])
        state_copy = state_copy / np.amax(state_copy)
        new_state_copy = np.array([newState.copy()])
        new_state_copy = new_state_copy / np.amax(new_state_copy)

        self.memory.addMemory(state_copy, action, reward, new_state_copy, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((0,self.input_size), dtype = np.float64)
            Y_batch = np.empty((0,self.output_size), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)

    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())


class SelectionNet1:
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        self.input_size = inputs
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate

    def initNetworks(self, hiddenLayers):
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.model = model

        targetModel = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.targetModel = targetModel

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        model = Sequential()
        model.add(Dense(10, input_dim=20, kernel_initializer='normal', activation='relu'))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        model.add(Dense(2, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        model.summary()
        return model

    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, state):
        predicted = self.model.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getTargetQValues(self, state):
        #predicted = self.targetModel.predict(state.reshape(1,len(state)))
        predicted = self.targetModel.predict(state.reshape(1,len(state)))
        return predicted[0]

    def getAction(self, state):
        predicted = self.model.predict(state.reshape(1,len(state)))
        return np.argmax(predicted[0])

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else :
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    def selectAction(self, state, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getAction(state)
        return action

    def addMemory(self, state, action, reward, newState, isFinal):
        state_copy = np.array([state.copy()])
        state_copy = state_copy / np.amax(state_copy)
        new_state_copy = np.array([newState.copy()])
        new_state_copy = new_state_copy / np.amax(new_state_copy)

        self.memory.addMemory(state_copy, action, reward, new_state_copy, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((0,self.input_size), dtype = np.float64)
            Y_batch = np.empty((0,self.output_size), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)

    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())




# Util Functions 
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
    num_actions = 11
    half_action = (num_actions - 1) / 2
    if y == 0:
        return 5
    mid = width/2
    return int((y - mid) *  half_action/ mid + half_action)

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

if __name__ == '__main__':
    # Init nets
    laser_weights = '/home/lntk/Desktop/Robot/model/mazecolor_laser.h5'
    laser_params = "/home/lntk/Desktop/Robot/model/mazecolor_laser.json"

    laser_learning_rate = 0.00025
    laser_inputs = 20
    laser_outputs = 11
    laser_network_structure = [15, 15]

    laser_net = LaserNet(laser_inputs, laser_outputs, laser_learning_rate)
    laser_net.initNetworks(laser_network_structure)
    laser_net.loadWeights(laser_weights)

    selection_weights = "/home/lntk/Desktop/Robot/model/mazecolor_selection.h5"
    selection_params = "/home/lntk/Desktop/Robot/model/mazecolor_selection.json"

    selection_update_target_network = 1000
    selection_current_episode = 0
    selection_minibatch_size = 64
    selection_exploration_rate = 1.0
    selection_learn_start = 64
    selection_inputs = 20
    selection_outputs = 1
    selection_learning_rate = 0.00025
    selection_discount_factor = 0.99
    selection_memory_size = 1000000
    selection_network_structure = [10, 10]

    selection_net = SelectionNet(selection_inputs, selection_outputs, selection_memory_size, selection_discount_factor, selection_learning_rate, selection_learn_start)
    selection_net.initNetworks(selection_network_structure)
    selection_net.loadWeights(selection_weights)

    env = gym.make('GazeboTurtlebotMazeColor-v0')
    observation = env.reset()

    while True:
        [image, laser] = observation
        hint_pos = getTargetPoints(image)
        if hint_pos == 0:
            hint_pos = getHintPoints(image)

        height, width, depth = image.shape
        image_action = get_image_action(width, hint_pos)
        laser_action = laser_net.selectAction(laser, 0)


        rl_choice = selection_net.selectAction(laser, 0)

        if rl_choice == 1:
            action = 10 - image_action
        else:
            action = laser_action

        next_observation, reward, done, info = env.step(action)
        next_image, next_laser = next_observation            
        observation = next_observation

        if done:       
            observation = env.reset()

    env.close()
