import numpy as np
import random

class ExperienceReplay():
    def __init__(self, path, buffer_size = 500000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.path = path + "/experience.txt"
 
    def add(self, experience):
        if len(self.buffer) + len(experience) > self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, min(size, len(self.buffer)))), [min(size, len(self.buffer)), 5])

    def save(self):
        np.save(self.path, self.buffer)
   
    def load(self):
        self.buffer = np.load(self.path + ".npy")
        
