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
        if len(hiddenLayers) == 0:
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform'))
            model.add(Activation("linear"))
        else :
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform'))
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))

            for index in range(1, len(hiddenLayers)):
                # print("adding layer "+str(index))
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, init='lecun_uniform'))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
            model.add(Dense(self.output_size, init='lecun_uniform'))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
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

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else :
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

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
        if len(hiddenLayers) == 0:
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform'))
            model.add(Activation("linear"))
        else :
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform'))
            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))

            for index in range(1, len(hiddenLayers)):
                # print("adding layer "+str(index))
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, init='lecun_uniform'))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
            model.add(Dense(self.output_size, init='lecun_uniform'))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
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

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else :
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

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



class DeepQ:
    def __init__(self, inputs, outputs, learningRate):
        self.input_size = inputs
        self.output_size = outputs
        self.learningRate = learningRate
        self.states = np.empty((0,self.input_size), dtype = np.float64)
        # self.actions = np.empty((0,self.output_size), dtype = np.float64)
        self.actions = np.empty((0, 1), dtype = np.float64)

    def initNetworks(self, hiddenLayers):
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.model = model

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        # model = Sequential()
        # if len(hiddenLayers) == 0:
        #     model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform'))
        #     model.add(Activation("linear"))
        # else :
        #     model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform'))
        #     if (activationType == "LeakyReLU") :
        #         model.add(LeakyReLU(alpha=0.01))
        #     else :
        #         model.add(Activation(activationType))
        #     model.add(Dropout(0.25))

        #     for index in range(1, len(hiddenLayers)):
        #         # print("adding layer "+str(index))
        #         layerSize = hiddenLayers[index]
        #         model.add(Dense(layerSize, init='lecun_uniform'))
        #         if (activationType == "LeakyReLU") :
        #             model.add(LeakyReLU(alpha=0.01))
        #         else :
        #             model.add(Activation(activationType))

        #         model.add(Dropout(0.25))
        #     model.add(Dense(self.output_size, init='lecun_uniform'))
        #     model.add(Activation('softmax'))

        # model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='SGD')
        model = Sequential()
        model.add(Dense(10, input_dim=20, kernel_initializer='normal', activation='relu'))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        model.summary()
        return model

    def getAction(self, state):
        state_copy = np.array([state.copy()])
        state_copy = state_copy / np.amax(state_copy)
        predicted = self.model.predict(state_copy.reshape(1,len(state)))
        # predicted = self.model.predict(state.reshape(1,len(state)))
        # print(predicted)
        return int(predicted[0] > 0.5)

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
        action_category = keras.utils.to_categorical(action, num_classes=2)
        action_category = action_category.reshape((1, len(action_category)))
        # self.actions = np.append(self.actions, action_category, axis=0)
        action_array = np.asarray([action])
        action_array = action_array.reshape((1, len(action_array)))
        self.actions = np.append(self.actions, action_array, axis=0)
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

def threshold_laser(laser, value):
    binary_vector = laser >= value
    result = ''
    for x in binary_vector:
        if x: 
            result += '1'
        else: 
            result += '0'
    return result

def select_action(laser):
    # binary_vector = laser >= 1.5
    # i = 0
    # length = len(laser)
    # wide_range = 0
    # max_wide_range = 0

    # while i < length:
    #     if binary_vector[i]:
    #         break
    #     i += 1
    # while i < length:
    #     if binary_vector[i]:
    #         wide_range += 1
    #         i += 1
    #     else:
    #         max_wide_range = max(wide_range, max_wide_range)
    #         wide_range = 0
    #         while i < length:
    #             if binary_vector[i]:
    #                 break
    #             i += 1
    # if max_wide_range >= 14:
    #     return 1

    binary_vector = laser >= 2.5
    i = 0
    length = len(laser)
    num_component = 0
    while i < length:
        if binary_vector[i]:
            break
        i += 1
    while i < length:
        if binary_vector[i]:
            i += 1
        else:
            num_component += 1
            while i < length:
                if binary_vector[i]:
                    break
                i += 1
    if num_component > 1:
        return 1
    else:
        return 0


if __name__ == '__main__':
    # DAgger part
    env = gym.make('GazeboTurtlebotMazeColor-v0')
    continue_execution = True

    weights_path = '/home/lntk/Desktop/mazecolor_dagger.h5'

    if not continue_execution:
        epochs = 1000
        steps = 1000
        explorationRate = 1
        learningRate = 0.00025
        network_inputs = 20
        network_outputs = 2
        network_structure = [10, 10]
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
        network_outputs = 2
        network_structure = [10, 10]
        current_epoch = 0
        deepQ = DeepQ(network_inputs, network_outputs, learningRate)
        deepQ.initNetworks(network_structure)
        deepQ.loadWeights(weights_path)
        print ("Import sucess.")



    # ===== INITIAL PHASE =====
    num_teach_episode = 5
    num_step = 200
    print("Initial Phase Started.")
    for episode in range(num_teach_episode):
        observation = env.reset()
        for t in range(num_step):
            [image, laser] = observation
            hint_pos = getTargetPoints(image)
            if hint_pos == 0:
                hint_pos = getHintPoints(image)

            height, width, depth = image.shape
            image_action = get_image_action(width, hint_pos)
            laser_action = get_expert_action(laser)

            # a = []
            expert_action = select_action(laser)
            # a.append(expert_action)
            expert_action = deepQ.selectAction(laser, 0)
            # a.append(expert_action)
            # print(a)

            if expert_action == 1:
                action = 10 - image_action
            else:
                action = laser_action

            next_observation, reward, done, info = env.step(action)
            next_image, next_laser = next_observation
                
            deepQ.addMemory(laser, expert_action)
            observation = next_observation

            if (t >= num_step - 1):
                done = True

            if done:        
                print("Teaching Episode " + str(episode + 1) + " completed.")
                break
    deepQ.learn()

    # ===== TRAINING PHASE =====
    print("Traning Phase Started.")
    num_iteration = 20
    num_episode = 5
    num_step = 200

    for i in range(num_iteration):
        for episode in range(num_episode):
            observation = env.reset()

            for t in range(num_step):
                [image, laser] = observation
                hint_pos = getTargetPoints(image)
                if hint_pos == 0:
                    hint_pos = getHintPoints(image)

                height, width, depth = image.shape
                image_action = get_image_action(width, hint_pos)
                laser_action = get_expert_action(laser)

                expert_action = select_action(laser)
                robot_action = deepQ.selectAction(laser, explorationRate)
                robot_predict_action = deepQ.getAction(laser)
                print([expert_action, robot_predict_action, robot_action])

                if robot_action == 1:
                    action = 10 - image_action
                else:
                    action = laser_action

                next_observation, reward, done, info = env.step(action)
                next_image, next_laser = next_observation
                    
                deepQ.addMemory(laser, expert_action)
                observation = next_observation

                if (t >= num_step - 1):
                    print ("reached the end! :D")
                    done = True

                if done:         
                    print ("EP "+str(episode)+ " - {} timesteps".format(t+1)) + "  exploration rate: " + str(round(explorationRate, 2)) 
                    break

            explorationRate *= 0.99 #epsilon decay
            # explorationRate -= (2.0/epochs)
            explorationRate = max (0.05, explorationRate)
        deepQ.learn()
        print ("Saving model ...")  
        deepQ.saveModel(weights_path)
    env.close()


    # # Init nets
    # # laser_weights = "/home/lntk/Desktop/Project Robot/turtlebot_model/180_maze.h5"
    # # laser_params = "/home/lntk/Desktop/Project Robot/turtlebot_model/180_maze.json"

    # # with open(laser_params) as outfile:
    # #     d = json.load(outfile)
    # #     laser_learn_start = d.get('learnStart')
    # #     laser_learning_rate = d.get('learningRate')
    # #     laser_discount_factor = d.get('discountFactor')
    # #     laser_memory_size = d.get('memorySize')
    # #     laser_inputs = d.get('network_inputs')
    # #     laser_outputs = d.get('network_outputs')
    # #     laser_network_structure = d.get('network_structure')

    # # laser_net = LaserNet(laser_inputs, laser_outputs, laser_memory_size, laser_discount_factor, laser_learning_rate, laser_learn_start)
    # # laser_net.initNetworks(laser_network_structure)
    # # laser_net.loadWeights(laser_weights)

    # selection_weights = "/home/lntk/Desktop/Project Robot/turtlebot_model/maze_color.h5"
    # selection_params = "/home/lntk/Desktop/Project Robot/turtlebot_model/maze_color.json"

    # continue_execution = True

    # if not continue_execution:
    #     selection_update_target_network = 1000
    #     selection_current_episode = 0
    #     selection_minibatch_size = 64
    #     selection_exploration_rate = 1.0
    #     selection_learn_start = 64
    #     selection_learning_rate = 0.00025
    #     selection_discount_factor = 0.99
    #     selection_memory_size = 1000000
    #     selection_inputs = 20
    #     selection_outputs = 2
    #     selection_network_structure = [10, 10]

    #     selection_net = SelectionNet(selection_inputs, selection_outputs, selection_memory_size, selection_discount_factor, selection_learning_rate, selection_learn_start)
    #     selection_net.initNetworks(selection_network_structure)
    # else:
    #     with open(selection_params) as outfile:
    #         d = json.load(outfile)
    #         selection_exploration_rate = d.get('exploration_rate')
    #         selection_learn_start = d.get('learn_start')
    #         selection_learning_rate = d.get('learning_rate')
    #         selection_discount_factor = d.get('discount_factor')
    #         selection_memory_size = d.get('memory_size')
    #         selection_inputs = d.get('network_inputs')
    #         selection_outputs = d.get('network_outputs')
    #         selection_network_structure = d.get('network_structure')
    #         selection_update_target_network = d.get('update_target_network')
    #         selection_current_episode = d.get('current_episode')
    #         selection_minibatch_size = d.get('minibatch_size')

    #     selection_net = SelectionNet(selection_inputs, selection_outputs, selection_memory_size, selection_discount_factor, selection_learning_rate, selection_learn_start)
    #     selection_net.initNetworks(selection_network_structure)
    #     # selection_net.loadWeights(selection_weights)

    # env = gym.make('GazeboTurtlebotMazeColor-v0')
    # observation = env.reset()

    # total_episodes = 10000
    # total_steps = 1000
    # start_time = time.time()
    # step_counter = 0


    # for episode in range(total_episodes):
    #     done = False
    #     cumulated_reward = 0
    #     observation = env.reset()

    #     for t in range(total_steps):
    #         [image, laser] = observation
    #         hint_pos = getTargetPoints(image)
    #         if hint_pos == 0:
    #             hint_pos = getHintPoints(image)

    #         height, width, depth = image.shape
    #         image_action = get_image_action(width, hint_pos)
    #         laser_action = get_expert_action(laser)


    #         # qValues = selection_net.getQValues(laser)
    #         # rl_choice = selection_net.selectAction(qValues, 0)
    #         rl_choice = select_action(laser)



    #         if rl_choice == 1:
    #             action = 10 - image_action
    #             print("Follow image")
    #         else:
    #             action = laser_action
    #             print("Follow laser")

    #         next_observation, reward, done, info = env.step(action)
    #         next_image, next_laser = next_observation

    #         cumulated_reward += reward
    #         # selection_net.addMemory(laser, rl_choice, reward, next_laser, done)
            
    #         # if step_counter >= selection_learn_start:
    #         #     if step_counter <= selection_update_target_network:
    #         #         selection_net.learnOnMiniBatch(selection_minibatch_size, False)
    #         #     else:
    #         #         selection_net.learnOnMiniBatch(selection_minibatch_size, True)
            
    #         observation = next_observation

    #         if done:
    #             m, s = divmod(int(time.time() - start_time), 60)
    #             h, m = divmod(m, 60)            
    #             print ("EP "+str(episode)+" - {} timesteps".format(t+1)+" - Cumulated R: "+str(cumulated_reward)+"   Eps="+str(round(selection_exploration_rate, 2))+"     Time: %d:%02d:%02d" % (h, m, s))
    #             if (episode)%20==0:
    #                 print ("Saving model ...")  
    #                 selection_net.saveModel(selection_weights)
    #                 parameter_keys = ['episodes','steps','update_target_network','exploration_rate','minibatch_size','learn_start','learning_rate','discount_factor','memory_size','network_inputs','network_outputs','network_structure','current_episode']
    #                 parameter_values = [total_episodes, total_steps, selection_update_target_network, selection_exploration_rate, selection_minibatch_size, selection_learn_start, selection_learning_rate
    #                                     , selection_discount_factor, selection_memory_size, selection_inputs, selection_outputs, selection_network_structure, episode]
    #                 parameter_dictionary = dict(zip(parameter_keys, parameter_values))
    #                 with open(selection_params, 'w') as outfile:
    #                     json.dump(parameter_dictionary, outfile)
    #             break

    #         step_counter += 1
    #         if step_counter % selection_update_target_network == 0:
    #             selection_net.updateTargetNetwork()
    #             print ("updating target network")

    #     selection_exploration_rate *= 0.99
    #     selection_exploration_rate = max (0.05, selection_exploration_rate)

    env.close()
