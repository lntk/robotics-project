import numpy as np
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import sys
import os
import time
import random

class Qnetwork():
    def __init__(self, numstate, output):
        self.input_size = numstate
        self.output_size = output
        self.model = self.createModel(numstate, output)
        self.targetModel = self.createModel(numstate, output)

    def createModel(self, numstate, output):

        model = Sequential()
        model.add(Dense(300, input_shape = (numstate[1], ), init='lecun_uniform'))
        model.add(Activation("relu"))
        model.add(Dense(300, init='lecun_uniform'))
        model.add(Activation("relu"))
        model.add(Dense(output, init='lecun_uniform'))
        model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=0.00025, rho=0.9, epsilon=1e-06)
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
        predicted = self.model.predict(state[1].reshape(1,len(state[1])))
        return predicted[0]

    def getTargetQValues(self, state):
        #predicted = self.targetModel.predict(state.reshape(1,len(state)))
        predicted = self.targetModel.predict(state[1].reshape(1,len(state[1])))

        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())

    def learnOnMiniBatch(self, miniBatch, useTargetNetwork, config):
        # Do not learn until we've got self.learnStart samples
        # print self.input_size[1]
        X_batch = np.empty((0,self.input_size[1]), dtype = np.float64)
        Y_batch = np.empty((0,self.output_size), dtype = np.float64)
        for sample in miniBatch:
            isFinal = sample[3]
            state = sample[0]
            action = sample[1]
            reward = sample[2]
            newState = sample[4]

            qValues = self.getQValues(state)
            if useTargetNetwork:
                qValuesNewState = self.getTargetQValues(newState)
            else :
                qValuesNewState = self.getQValues(newState)
            if isFinal:
                targetValue = reward
            else:
                targetValue = reward + config.gamma * self.getMaxQ(qValuesNewState)
            X_batch = np.append(X_batch, np.array([state[1].copy()]), axis=0)
            Y_sample = qValues.copy()
            Y_sample[action] = targetValue
            Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
            if isFinal:
                X_batch = np.append(X_batch, np.array([newState[1].copy()]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
        self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), epochs=1, verbose = 0)

