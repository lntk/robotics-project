import random
import numpy as np
import sys
import pickle

from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import keras

def get_label(array):
    binary_vector = array >= 1.1
    i = 0
    length = len(array)
    component = []
    count = 0
    while i < length:
        if binary_vector[i]:
            break   
        i += 1
    while i < length:
        if binary_vector[i]:
            i += 1
            count += 1
        else:
            component.append(count)
            count = 0
            while i < length:
                if binary_vector[i]:
                    break
                i += 1
    if np.amax(component) > 14:
        return 1


    binary_vector = array >= 2.5
    i = 0
    length = len(array)
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

weight_path = '/home/lntk/Desktop/neural_net_test.h5'
data_path = '/home/lntk/Desktop/training_data.txt'

def load_data():
    with open(data_path, 'rb') as handle:
        return pickle.load(handle)

def save_data(data):
    print("Saving...")
    with open(data_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

model = Sequential()
model.add(Dense(40, input_dim=20, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(40, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.summary()
# model.set_weights(load_model(weight_path).get_weights())

# old_X_train, old_y_train = load_data()

X_train = np.empty((0,20), dtype = np.float64)
y_train = np.empty((0,1), dtype = np.float64)

for i in range(60000):
	x = np.random.uniform(low=0.06, high=3.0, size=20)
	y = get_label(x)
	x = x.reshape((1, 20))
	y = np.asarray([y])
	y = y.reshape((1, 1))
	X_train = np.append(X_train, x, axis=0)
	y_train = np.append(y_train, y, axis=0)

# X_train = np.append(old_X_train, X_train, axis=0)
# y_train = np.append(old_y_train, y_train, axis=0)
# save_data([X_train, y_train])


model.fit(X_train, y_train, batch_size=64, nb_epoch=100, verbose=1)
model.save(weight_path)


X_test = np.empty((0,20), dtype = np.float64)
y_test = np.empty((0,1), dtype = np.float64)

correct = 0
for i in range(5000):
	x = np.random.uniform(low=0.06, high=3.0, size=20)
	y = get_label(x)
	y_predict = model.predict(x.reshape(1,20))[0]
	if int(y_predict>0.5) == y:
		correct += 1

	# x = x.reshape((1, 20))
	# y = np.asarray([y])
	# y = y.reshape((1, 1))
	# X_test = np.append(X_test, x, axis=0)
	# y_test = np.append(y_test, y, axis=0)
print(float(correct)/1000)
# score = model.evaluate(X_test, y_test, batch_size=64, verbose=1)
# print(score)
