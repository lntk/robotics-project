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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
import memory

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
        self.image_input_size = [32, 32, 3]
        self.lazer_input_size = 100
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

    # added by lntk
    # this model takes both camera and laser input
    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        img_rows = 32
        img_cols = 32
        img_channels = 3
        img_input = Input(shape=(img_rows, img_cols, img_channels))
        layer1 = Conv2D(16, (2, 2), padding='same', activation='relu')(img_input)
        layer2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(layer1)
        layer3 = Conv2D(16, (2, 2), padding='same', activation='relu')(layer2)
        layer4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(layer3)
        layer5 = Flatten()(layer4)
        layer6 = Dense(512, activation='relu')(layer5)

        laser_input = Input(shape=(100,))
        merge_input = keras.layers.concatenate([layer6, laser_input])

        layer7 = Dense(300, activation='relu')(merge_input)
        layer8 = Dense(200, activation='relu')(layer7)
        layer9 = Dense(100, activation='relu')(layer8)
        output = Dense(21, activation='softmax')(layer9)

        model = Model(inputs=[img_input, laser_input], outputs=output)

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
    # state is a list consisting of 2 numpy array: image with shape (32, 32, 3) and laser with shape(100,)
    def getQValues(self, state):
        [image, laser] = state
        rsImage = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        rsLaser = laser.reshape(1, laser.shape[0])
        rsState = [rsImage, rsLaser]
        predicted = self.model.predict(rsState)
        return predicted[0]

    def getTargetQValues(self, state):
        #predicted = self.targetModel.predict(state.reshape(1,len(state)))
        [image, laser] = state
        rsImage = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        rsLaser = laser.reshape(1, laser.shape[0])
        rsState = [rsImage, rsLaser]
        predicted = self.targetModel.predict(rsState)

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

            [h, w, c] = self.image_input_size
            X_image_batch = np.empty((0,h,w,c), dtype = np.float64)
            X_laser_batch = np.empty((0,self.lazer_input_size), dtype = np.float64)
            X_batch = [X_image_batch, X_laser_batch]
            Y_batch = np.empty((0,self.output_size), dtype = np.float64)

            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                # for x in state:
                #     print(x.shape)
                [image, laser, position] = state
                qValues = self.getQValues([image, laser])
                newImage = newState[0]
                newLaser = newState[1]
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues([newImage, newLaser])
                else :
                    qValuesNewState = self.getQValues([newImage, newLaser])
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                # print(X_image_batch.shape)
                # print(newImage.copy().shape)
                X_image_batch = np.append(X_image_batch, np.array([newImage.copy()]), axis=0)
                # print(X_laser_batch.shape)
                # print(newLaser.copy().shape)
                X_laser_batch = np.append(X_laser_batch, np.array([newLaser.copy()]), axis=0)
                X_batch = [X_image_batch, X_laser_batch]
                Y_sample = qValues.copy()
                # print(type(Y_sample[action]))
                # print(type(targetValue))
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_image_batch = np.append(X_image_batch, np.array([newImage.copy()]), axis=0)
                    X_laser_batch = np.append(X_laser_batch, np.array([newLaser.copy()]), axis=0)
                    X_batch = [X_image_batch, X_laser_batch]
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

# ===== OBSTACLE AVOIDANCE =====
def expert_action(data):
    THRESHOLD = 4
    total_range = 200
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

    unit_angle = float(total_range)/20
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

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    env = gym.make('GazeboTurtlebotMazeColor-v0')

    # True if you want to continue training
    continue_execution = True

    weights_path = '/home/lntk/Desktop/turtle_mazecolor_dqn.h5'
    params_json  = '/home/lntk/Desktop/turtle_mazecolor_dqn.json'

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
        network_structure = [300, 200, 100]
        current_epoch = 0

        deepQ = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)
        # env.monitor.start(outdir, force=True, seed=None)
    else:
        #Load weights, monitor info and parameter info.
        #ADD TRY CATCH fro this else
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
            [image, laser, position] = observation
            qValues = deepQ.getQValues([image, laser])
            action = deepQ.selectAction(qValues, explorationRate)
            newObservation, reward, done, info = env.step(action)
            [newImage, newLaser, newPosition] = newObservation

            # tinh reward cong don
            # va cap nhat reward cao nhat
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward
                
            deepQ.addMemory(observation, action, reward, [newImage, newLaser], done)

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
                    deepQ.saveModel('/home/lntk/Desktop/turtle_mazecolor_dqn'+'.h5')
                    parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
                    parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
                    parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                    with open('/home/lntk/Desktop/turtle_mazecolor_dqn'+'.json', 'w') as outfile:
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
