import time
import os
import numpy as np

class Utility():
	def __init__(self, reward_file, step_file):
		self.reward_list = []
		self.step_list = []
		self.reward_file = reward_file
		self.step_file = step_file

	def updateTargetGraph(self, tfVars,tau):
		total_vars = len(tfVars)
		op_holder = []
		for idx,var in enumerate(tfVars[0:total_vars//2]):
			op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
		return op_holder

	def updateTarget(self, op_holder,sess):
		for op in op_holder:
			sess.run(op)

	def saveReward(self):
		np.save(self.reward_file, self.reward_list)

	def saveStep(self):
		np.save(self.step_file, self.step_list)
   
	def loadReward(self, file):
		try:
			print file+".npy"
			self.reward_list = np.load(file + ".npy")
			print self.reward_list
		except:
			pass
        
	def loadStep(self, file):
		try:
			self.step_list = np.load(file + ".npy")
			print self.step_list
		except:
			pass



class Config():
	"""docstring for Config"""
	def __init__(self):
		self.batch_size = 64
		self.gamma = 0.8
		self.epsilon = 1.0
		self.epsilon_decay = 1.0/3000
		self.load_model = False
		self.alpha = 0.2
		self.path = "./boltzmann1"
		self.step_file = '/step_list_' + str(time.time()) + '.txt'
		self.reward_file = '/reward_list_' + str(time.time()) + '.txt'
		self.oldFile = '/old.txt'
		self.oldReward = ''
		self.oldStep = ''
		self.pre_train_step = 5000
		self.tau = 0.5
		self.update_target = 10000
		self.save_ep = 20
		self.episode = 0
		self.total_step = 0

		self.ableSave = False
			
	def saveOldFile(self):
		with open(self.path + self.oldFile, 'w') as f:
			f.write(''.join([self.reward_file,'\n']))
			f.write(''.join([self.step_file,'\n']))

	def loadOldFile(self):
		try:
			with open(self.path + self.oldFile, 'r') as f:
				lines = []
				for line in f:
					lines.append(line)
				self.oldReward = self.path + lines[0].replace("\n", "")
				self.oldStep = self.path + lines[1].replace("\n", "")
		except:
			pass


