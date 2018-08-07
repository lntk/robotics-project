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

from keras.models import Sequential, load_model
from keras.initializers import normal
from keras import optimizers
from keras.optimizers import RMSprop
from keras.layers import Conv2D, Flatten, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import SGD , Adam
import memory


class ImageNet:
    def __init__(self, outputs, memorySize, discountFactor, learningRate, learnStart):
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate

    def initNetworks(self):
        model = self.createModel()
        self.model = model

    def createModel(self):
        # Network structure must be directly changed here.
        model = Sequential()
        model.add(Conv2D(8, (2, 2), activation='relu', input_shape=(64, 64, 1)))
        model.add(Conv2D(8, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(network_outputs))
        #adam = Adam(lr=self.learningRate)
        #model.compile(loss='mse',optimizer=adam)
        model.compile(RMSprop(lr=self.learningRate), 'MSE')
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
        predicted = self.model.predict(state)
        return predicted[0]

    def getTargetQValues(self, state):
        predicted = self.targetModel.predict(state)
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
            X_batch = np.empty((1, 64, 64, 1), dtype = np.float64)
            Y_batch = np.empty((1,self.output_size), dtype = np.float64)
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
                X_batch = np.append(X_batch, state.copy(), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, newState.copy(), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, validation_split=0.2, batch_size = len(miniBatch), nb_epoch=1, verbose = 0)

    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())


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

def getPoints(image, red, green, blue):
    epsilon = 0.00001
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
                # weightY = abs(float(y - mid)) / mid
                # sumY += y * weightY
                sumY += y
                countX += 1
                # countY += weightY
                countY += 1

    if countX == 0 and countY == 0:
        return np.asarray([0, 0])

    centerX = float(sumX)/countX
    centerY = float(sumY)/countY
    # centerY = float(sumY)/(countY + epsilon)
    

    return np.asarray([centerX, centerY])

def getTargetPoints(image):
    return getPoints(image, 102, 20, 20)

def getHintPoints(image):
    return getPoints(image, 255, 120, 120)

def getImageAction(height, width, x, y):
    if x == 0 and y == 0:
        return 100
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

    laser_weights = "/home/lntk/Desktop/Project Robot/turtlebot_model/180_maze.h5"
    laser_params = "/home/lntk/Desktop/Project Robot/turtlebot_model/180_maze.json"

    with open(laser_params) as outfile:
        d = json.load(outfile)
        laser_learn_start = d.get('learnStart')
        laser_learning_rate = d.get('learningRate')
        laser_discount_factor = d.get('discountFactor')
        laser_memory_size = d.get('memorySize')
        laser_inputs = d.get('network_inputs')
        laser_outputs = d.get('network_outputs')
        laser_network_structure = d.get('network_structure')

    laser_net = LaserNet(laser_inputs, laser_outputs, laser_memory_size, laser_discount_factor, laser_learning_rate, laser_learn_start)
    laser_net.initNetworks(laser_network_structure)
    laser_net.loadWeights(laser_weights)

    image_weights = "/home/lntk/Desktop/Project Robot/turtlebot_model/maze_color.h5"
    image_params = "/home/lntk/Desktop/Project Robot/turtlebot_model/maze_color.json"

    img_rows, img_cols, img_channels = 64, 64, 1
    epochs = 100000
    steps = 1000
    continue_execution = False

    if not continue_execution:
        image_minibatch_size = 32
        image_learningRate = 1e-3#1e6
        image_discountFactor = 0.95
        image_network_outputs = 21
        image_memorySize = 100000
        image_learnStart = 64 # timesteps to observe before training
        image_explorationRate = 1
        image_current_epoch = 0
        image_stepCounter = 0

        image_net = ImageNet(image_network_outputs, image_memorySize, image_discountFactor, image_learningRate, image_learnStart)
        image_net.initNetworks()
        # env.monitor.start(outdir, force=True, seed=None)
    else:
        #Load weights, monitor info and parameter info.
        with open(image_params) as outfile:
            d = json.load(outfile)
            image_explorationRate = d.get('explorationRate')
            image_minibatch_size = d.get('minibatch_size')
            image_learnStart = d.get('learnStart')
            image_learningRate = d.get('learningRate')
            image_discountFactor = d.get('discountFactor')
            image_memorySize = d.get('memorySize')
            image_network_outputs = d.get('network_outputs')
            image_current_epoch = d.get('current_epoch')
            image_stepCounter = d.get('stepCounter')

        image_net = ImageNet(image_network_outputs, image_memorySize, image_discountFactor, image_learningRate, image_learnStart)
        image_net.initNetworks()
        image_net.loadWeights(image_weights)

    start_time = time.time()
    for epoch in range(image_current_epoch+1, epoch+1, 1):
        observation = en.reset()
        cumulated_reward = 0

        for t in range(steps):
            [image, laser] = observation
            qValues = image_net.getQValues(image)
            image_action = image_net.selectAction(qValues, image_explorationRate)
            laser_action = laser_net.selectAction(laser, 0)
            action = int((image_action + laser_action)/2)

            new_observation, reward, done, info = env.step(action)
            [new_laser, new_image] = new_observation
            image_net.addMemory(image, action, reward, new_image, done)
            observation = new_observation

            if image_stepCounter >= image_learnStart:
                image_net.learnOnMiniBatch(image_minibatch_size, False)

            if (t == steps-1):
                print ("reached the end")
                done = True

            if done:
                m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
                h, m = divmod(m, 60)
                print ("EP "+str(epoch)+" - {} steps".format(t+1)+" - CReward: "+str(round(cumulated_reward, 2))+"  Eps="+str(round(explorationRate, 2))+"  Time: %d:%02d:%02d" % (h, m, s))
                
                if (epoch)%50==0:
                    print ("Saving model ...")
                    image_net.saveModel('/home/lntk/Desktop/Project Robot/turtlebot_model/maze_color' + '.h5')
                    parameter_keys = ['explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_outputs','current_epoch','image_stepCounter']
                    parameter_values = [image_explorationRate, image_minibatch_size, image_learnStart, image_learningRate, image_discountFactor, image_memorySize, image_network_outputs, epoch, image_stepCounter]
                    parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                    with open('/home/lntk/Desktop/Project Robot/turtlebot_model/maze_color'+'.json', 'w') as outfile:
                        json.dump(parameter_dictionary, outfile)
                break

            image_stepCounter += 1
            image_explorationRate *= 0.99
            image_explorationRate = max (0.05, image_explorationRate)

    env.close()
