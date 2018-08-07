#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import random
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.layers import Input, Dense
from keras.models import Model
import memory
import cv2

class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s'))

    """
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        """
        Parameters:
            - inputs: input size
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
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

    def initRegularizedNetworks(self, hiddenLayers):
        model = self.createRegularizedModel(self.input_size, self.output_size, hiddenLayers, "LeakyReLU", self.learningRate)
        self.model = model

        targetModel = self.createRegularizedModel(self.input_size, self.output_size, hiddenLayers, "LeakyReLU", self.learningRate)
        self.targetModel = targetModel

    def createRegularizedModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        bias = True
        dropout = 0
        regularizationFactor = 0.01
        model = Sequential()
        if len(hiddenLayers) == 0:
            model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        else :
            if regularizationFactor > 0:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform', W_regularizer=l2(regularizationFactor),  bias=bias))
            else:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform', bias=bias))

            if (activationType == "LeakyReLU") :
                model.add(LeakyReLU(alpha=0.01))
            else :
                model.add(Activation(activationType))

            for index in range(1, len(hiddenLayers)):
                layerSize = hiddenLayers[index]
                if regularizationFactor > 0:
                    model.add(Dense(layerSize, init='lecun_uniform', W_regularizer=l2(regularizationFactor), bias=bias))
                else:
                    model.add(Dense(layerSize, init='lecun_uniform', bias=bias))
                if (activationType == "LeakyReLU") :
                    model.add(LeakyReLU(alpha=0.01))
                else :
                    model.add(Activation(activationType))
                if dropout > 0:
                    model.add(Dropout(dropout))
            model.add(Dense(self.output_size, init='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    # def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
    #     model = Sequential()
    #     if len(hiddenLayers) == 0:
    #         model.add(Dense(self.output_size, input_shape=(self.input_size,), init='lecun_uniform'))
    #         model.add(Activation("linear"))
    #     else :
    #         model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), init='lecun_uniform'))
    #         if (activationType == "LeakyReLU") :
    #             model.add(LeakyReLU(alpha=0.01))
    #         else :
    #             model.add(Activation(activationType))

    #         for index in range(1, len(hiddenLayers)):
    #             # print("adding layer "+str(index))
    #             layerSize = hiddenLayers[index]
    #             model.add(Dense(layerSize, init='lecun_uniform'))
    #             if (activationType == "LeakyReLU") :
    #                 model.add(LeakyReLU(alpha=0.01))
    #             else :
    #                 model.add(Activation(activationType))
    #         model.add(Dense(self.output_size, init='lecun_uniform'))
    #         model.add(Activation("linear"))
    #     optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
    #     model.compile(loss="mse", optimizer=optimizer)
    #     model.summary()
    #     return model

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        bias = True
        dropout = 0.5
        regularizationFactor = 0.01

        laser_input = Input(shape=(100,))
        hidden_1 = Dense(60, kernel_initializer='he_uniform', kernel_regularizer=l2(regularizationFactor), activation='relu', bias=bias)(laser_input)
        hidden_1 = BatchNormalization()(hidden_1)
        drop_1 = Dropout(dropout)(hidden_1)
        hidden_2 = Dense(21, kernel_initializer='he_uniform', kernel_regularizer=l2(regularizationFactor), activation='relu', bias=bias)(drop_1)
        hidden_2 = BatchNormalization()(hidden_2)
        drop_2 = Dropout(dropout)(hidden_2)

        point_input = Input(shape=(2,))
        merge_input = keras.layers.concatenate([drop_2, point_input])
        merge_input = BatchNormalization()(merge_input)

        output = Dense(21, kernel_initializer='lecun_uniform', activation='linear', bias=bias)(merge_input)

        model = Model(inputs=[laser_input, point_input], outputs=output)

        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print("layer ",i,": ",weights)
            i += 1


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
        [laser, points] = state
        rsLaser = laser.reshape(1, laser.shape[0])
        rsPoints = points.reshape(1, points.shape[0])
        rsState = [rsLaser, rsPoints]
        predicted = self.model.predict(rsState)
        return predicted[0]

    def getTargetQValues(self, state):
        [laser, points] = state
        rsLaser = laser.reshape(1, laser.shape[0])
        rsPoints = points.reshape(1, points.shape[0])
        rsState = [rsLaser, rsPoints]
        predicted = self.model.predict(rsState)
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

    def selectBoltzmannAction(self, qValues):
        actionDistribution = softmax(qValues, 0.5)
        action = np.random.choice(qValues.size, 1, p=actionDistribution)
        return action[0]
    
    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if (rand <= value):
                return i
            i += 1

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            # minibatch can be understood as a set of discovered states
            # here minibatch is a ... of dictionaries, each corresponding to a state
            miniBatch = self.memory.getMiniBatch(miniBatchSize)

            X_laser_batch = np.empty((0,100), dtype = np.float64)
            X_points_batch = np.empty((0,2), dtype = np.float64)
            X_batch = [X_laser_batch, X_points_batch]
            Y_batch = np.empty((0,self.output_size), dtype = np.float64)

            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                laser, points = state
                qValues = self.getQValues(state)
                newLaser, newPoints = newState
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_laser_batch = np.append(X_laser_batch, np.array([newLaser.copy()]), axis=0)
                X_points_batch = np.append(X_points_batch, np.array([newPoints.copy()]), axis=0)
                X_batch = [X_laser_batch, X_points_batch]
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_laser_batch = np.append(X_laser_batch, np.array([newLaser.copy()]), axis=0)
                    X_points_batch = np.append(X_points_batch, np.array([newPoints.copy()]), axis=0)
                    X_batch = [X_laser_batch, X_points_batch]
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), epochs=1, verbose = 0)

    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print(file)
        os.unlink(file)

def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist

def calculateReward(observation, action):
    bias = 2
    increment = 1
    reward = 0
    checkRange = 10
    minObservation = min(observation)
    for i in range(max(action * 5 - checkRange, 0), min(action * 5 + checkRange, 104)):
        if observation[i] > (minObservation + bias): 
            reward += increment
    return reward

def getPoints(image, red, green, blue):    
    height, width, depth = image.shape
    sumX = 0
    sumY = 0
    countX = 0
    countY = 0 
    for x in range(height):
        for y in range(width):
            if (image[x][y][0] == red) and (image[x][y][1] <= green) and (image[x][y][2] <= blue):
                sumX += x
                sumY += y
                countX += 1
                countY += 1

    if countX == 0 or countY == 0:
        return np.asarray([0, 0])

    centerX = float(sumX)/countX
    centerY = float(sumY)/countY

    return np.asarray([centerX/height, centerY/width])

def getTargetPoints(image):
    return getPoints(image, 102, 20, 20)

def getHintPoints(image):
    return getPoints(image, 255, 120, 120)

def normalize(array):
    epsilon = 1e-4
    max = np.amax(array)
    min = np.amin(array)
    return (array - min + epsilon)/(max - min + epsilon)

if __name__ == '__main__':

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    env = gym.make('GazeboTurtlebotMazeColor-v0')

    continue_execution = False

    weights_path = '/home/lntk/Desktop/turtle_mazecolor_camera_dqn.h5'
    # monitor_path = '/tmp/turtle_c2_dqn_ep200'
    params_json  = '/home/lntk/Desktop/turtle_mazecolor_camera_dqn.json'

    if not continue_execution:
        #Each time we take a sample and update our weights it is called a mini-batch.
        #Each time we run through the entire dataset, it's called an epoch.
        #PARAMETER LIST
        epochs = 1000
        steps = 10000
        updateTargetNetwork = 10000
        explorationRate = 1
        minibatch_size = 64
        learnStart = 64
        learningRate = 0.00025
        discountFactor = 0.99
        memorySize = 1000000
        network_inputs = 100
        network_outputs = 21
        network_structure = [200, 100, 50]
        current_epoch = 0

        deepQ = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)
        # env.monitor.start(outdir, force=True, seed=None)
    else:
        #Load weights, monitor info and parameter info.
        #ADD TRY CATCH for this else
        with open(params_json) as outfile:
            d = json.load(outfile)
            epochs = d.get('epochs')
            steps = d.get('steps')
            updateTargetNetwork = d.get('updateTargetNetwork')
            explorationRate = d.get('explorationRate')
            minibatch_size = d.get('minibatch_size')
            learnStart = d.get('learnStart')
            learningRate = d.get('learningRate')
            discountFactor = d.get('discountFactor')
            memorySize = d.get('memorySize')
            network_inputs = d.get('network_inputs')
            network_outputs = d.get('network_outputs')
            network_structure = d.get('network_structure')
            current_epoch = d.get('current_epoch')

        deepQ = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)
        deepQ.loadWeights(weights_path)
	print ("Import sucess.")

    stepCounter = 0
    highest_reward = 0

    start_time = time.time()

    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation = env.reset()
        cumulated_reward = 0

        for t in xrange(steps):
            # Get and preprocess observation
            [image, laser, position] = observation
        
            x, y = getTargetPoints(image)
            if x == 0 and y == 0:
                x, y = getHintPoints(image)
            points = np.asarray([x, y])

            # get Q-values
            qValues = deepQ.getQValues([laser, points])
            action = deepQ.selectAction(qValues, explorationRate)

            newObservation, reward, done, info = env.step(action)
            [newImage, newLaser, newPosition] = newObservation
            x, y = getTargetPoints(newImage)
            if x == 0 and y == 0:
                x, y = getHintPoints(newImage)
            newPoints = np.asarray([x, y])

            # total reward
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
                
            deepQ.addMemory([laser, points], action, reward, [newLaser, newPoints], done)

            if stepCounter >= learnStart:
                if stepCounter <= updateTargetNetwork:
                    deepQ.learnOnMiniBatch(minibatch_size, False)
                else :
                    deepQ.learnOnMiniBatch(minibatch_size, True)

            observation = newObservation

            if (t >= 2000):
                print ("reached the end! :D")
                done = True

            if done:
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)			
                print ("EP "+str(epoch)+" - {} timesteps".format(t+1)+" - Cumulated R: "+str(cumulated_reward)+"   Eps="+str(round(explorationRate, 2))+"     Time: %d:%02d:%02d" % (h, m, s))
                if (epoch)%20==0:
                    print ("Saving model ...")	
                    deepQ.saveModel('/home/lntk/Desktop/turtle_mazecolor_camera_dqn'+'.h5')
                    parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
                    parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
                    parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                    with open('/home/lntk/Desktop/turtle_mazecolor_camera_dqn'+'.json', 'w') as outfile:
                        json.dump(parameter_dictionary, outfile)
                break

            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network")

        explorationRate *= 0.995 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)

    # env.monitor.close()
    env.close()
