import tensorflow as tf
import numpy as np


class ActorCritic():
	pass

class Actor():
	def __init__(self, learning_rate=0.1):
		self.num_state = 100
		self.num_hidden = 60
		self.num_action = 21

		self.learning_rate = tf.constant(learning_rate)
		self.state = tf.placeholder(tf.float32, (1,100), 'state')
		self.target = tf.placeholder(dtype=tf.float32, name='target')
		self.action = tf.placeholder(dtype=tf.int32, name='action')
		self.discount = tf.constant(0.9)

		self.hidden = tf.contrib.layers.fully_connected(
				inputs=self.state,
				num_outputs=self.num_hidden,
				activation_fn=tf.nn.relu,
				weights_initializer=tf.zeros_initializer)

		self.action_preferences = tf.contrib.layers.fully_connected(
				inputs=self.hidden,
				num_outputs=self.num_action,
				activation_fn=None,
				weights_initializer=tf.zeros_initializer)

		# tf.initialize_all_variables()

		self.action_probabilities = tf.squeeze(tf.nn.softmax(self.action_preferences))
		print(self.action_probabilities)
		self.chosen_action_probability = tf.gather(self.action_probabilities, self.action)
		self.loss = tf.log(self.chosen_action_probability) * self.target
		
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		grads_and_vars = self.optimizer.compute_gradients(self.loss, tf.trainable_variables())
		print(grads_and_vars)
		self.gradient, self.variable = zip(*grads_and_vars)
		self.new_variables_op = tf.add(self.variable, tf.multiply(self.learning_rate, tf.multiply(self.target, tf.multiply(self.discount, self.gradient))))
		self.update_op = self.variable.assign(self.new_variables_op)

	def update_gradient(self, state, target, action, sess=None):
		sess = tf.Session()
		feed_dict = {self.state : state, self.target : target, self.action : action}
		new_vars = sess.run(self.new_variables, feed_dict)
		sess.run(self.variable.assign(new_vars))
		print(self.variable.shape)


class Critic():
	pass



actor = Actor()

session = tf.Session()
session.run(tf.global_variables_initialization())
for i in range(10):
	state = [np.random.rand(100)]
	target = np.random.rand(1)[0]
	action = np.random.randint(21)
	feed_dict = {self.state : state, self.target : target, self.action : action}
	session.run(actor.new_variables_op,feed_dict)