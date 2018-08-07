import gym
import gym_gazebo
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
from Qnetwork import Qnetwork
from ExperienceReplay import ExperienceReplay
from Utility import Utility
from Utility import Config

env = gym.make('GazeboTurtlebotMazeColor-v0')

observation = env.reset

#set parameter 
config = Config()
config.path = "./DQN_maze_target_v9"
if not os.path.exists(config.path):
    os.makedirs(config.path)
config.loadOldFile()
config.saveOldFile()
config.load_model = False
config.pre_train_step = 1000
config.epsilon_decay = 1.0/1000
config.gamma = 0.99

network = Qnetwork(env.num_state, env.num_action)
replay = ExperienceReplay(config.path)

utility = Utility(config.path + config.reward_file, config.path + config.step_file)



######load data#######
start_time = time.time()
if config.load_model == True:
    print('loading model....')
    if (os.path.isfile(config.path + "/model.h5")):
        network.loadWeights(config.path  + "/model.h5")
        utility.loadStep(config.oldStep)
        utility.loadReward(config.oldReward)
        utility.step_list = utility.step_list.tolist()
        utility.reward_list = utility.reward_list.tolist()
        config.episode = len(utility.step_list)
        config.epsilon -= config.epsilon_decay * config.episode
        print config.episode
        print "restore ok"



while True:
    observation = env.reset()

    observation = observation
    total_reward = 0
    total_random = 0
    replay_ep = ExperienceReplay(config.path)
    if (config.total_step >= config.pre_train_step):
        config.episode += 1
        config.ableSave = True
        
    for i in range(2000):
        # Get action
        if config.total_step < config.pre_train_step:
            action = np.random.randint(env.num_action)
            total_random += 1
        else:
            if (np.random.rand() < config.epsilon):
                action = np.random.randint(env.num_action)
                total_random += 1
            else:
                action = network.getMaxIndex(qvalue)


        # Get state                
        observation_old = observation
        observation, reward, done, _ = env.step(action)
        observation = observation
        if observation == []:
            break
        replay_ep.add(np.reshape([observation_old, action, reward, done, observation], [1, 5]))
        config.total_step += 1
        total_reward += reward
        if done or i == 1999:
            if (config.ableSave):
                utility.step_list.append(i + 1)
        if done:
            break
        

        # Training
        if config.total_step > config.batch_size:
            if config.total_step % 1 == 0:
                trainBatch = replay.sample(config.batch_size)
                if (config.total_step > config.update_target):
                    network.learnOnMiniBatch(trainBatch, True, config)
                else:
                    network.learnOnMiniBatch(trainBatch, False, config)
            if (config.total_step % config.update_target ==0):
                network.updateTargetNetwork()
                print ("updating target network")
                
    if config.total_step > config.pre_train_step:
        if config.epsilon > 0.05:
            config.epsilon -= config.epsilon_decay

    replay.add(replay_ep.buffer)


    #########Save data##########
    if (config.ableSave):
        utility.reward_list.append(total_reward)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
    if (config.ableSave):
        print ("EP: "+str(config.episode) + " - epsilon: "+str(round(config.epsilon,2))+"] - Random: "+ str(total_random)+"] - Reward: "+str(total_reward)+" Step: " + str(utility.step_list[-1]) + "     Time: %d:%02d:%02d" % (h, m, s))
        print ("Total step: " + str(config.total_step))
    if config.episode % config.save_ep == 0 and config.total_step >= config.pre_train_step:
        #print sess.run(W)
        network.saveModel(config.path + "/model.h5")
        print("Saved Model")
        utility.saveReward()
        utility.saveStep()


    
