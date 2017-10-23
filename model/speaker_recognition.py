from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import pdb


class SpeakerRecognition(object):

	def __init__(self, config, scope_name=None, device='gpu'):
		self.config = config
		self.scope = scope_name or "SpeakerRecogniton"
		self.create_placeholders()
		self.global_step = \
			tf.contrib.framework.get_or_create_global_step()
		self.metrics = {}
		if device == 'gpu':
			tf.device('/gpu:0')
		else:
			tf.device('/cpu:0')

		with tf.variable_scope(self.scope):
			self.build_model()
			self.compute_loss_and_metrics()
			self.compute_gradients_and_train_op()

	def create_placeholders(self):
		batch_size = self.config.batch_size
		time_dim = self.config.time_dim
		freq_dim = self.config.freq_dim
		num_speakers = self.config.num_speakers

		self.input = tf.placeholder(
			tf.float32, shape=[batch_size,freq_dim,time_dim,1], name="input")
		self.target = tf.placeholder(
			tf.int64, shape=[batch_size,num_speakers], name="target")
		self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")



	def build_model(self):
		config = self.config
		batch_size = config.batch_size
		num_speakers = config.num_speakers
		rand_uni_initializer = \
			tf.random_uniform_initializer(
				-self.config.init_scale, self.config.init_scale)
		
		layer_1 = self.input
		l1_padding  = tf.constant([[0,0],[1,1],[1,1],[0,0]])
		layer_1 = tf.pad(layer_1,l1_padding)
		layer_2 = tf.layers.conv2d(self.input,filters=96,kernel_size=7,strides=2,padding="valid",activation=tf.nn.relu,kernel_initializer=rand_uni_initializer,name="layer_2")
		layer_3 = tf.layers.max_pooling2d(layer_2,pool_size=3,strides=2,padding="valid",name="layer_3") 
		layer_3 = tf.pad(layer_3,l1_padding)
		layer_4 = tf.layers.conv2d(layer_3,filters=256,kernel_size=5,strides=2,padding="valid",activation=tf.nn.relu,kernel_initializer=rand_uni_initializer,name="layer_4")
		layer_5 = tf.layers.max_pooling2d(layer_4,pool_size=3,strides=2,padding="valid",name="layer_5")
		layer_6 = tf.layers.conv2d(layer_5,filters=256,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu,kernel_initializer=rand_uni_initializer, name="layer_6")
		layer_7 = tf.layers.conv2d(layer_6,filters=256,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu,kernel_initializer=rand_uni_initializer,name="layer_7")
		layer_8 = tf.layers.conv2d(layer_7,filters=256,kernel_size=3,strides=1,padding="same",activation=tf.nn.relu,kernel_initializer=rand_uni_initializer,name="layer_8")
		layer_9 = tf.layers.max_pooling2d(layer_8,pool_size=(5,3),strides=(3,2),padding="valid",name="layer_9")
		layer_10 = tf.layers.conv2d(layer_9,filters=4096,kernel_size=(9,1),strides=1,padding="valid",activation=tf.nn.relu,kernel_initializer=rand_uni_initializer,name="layer_10")
		layer_11 = tf.layers.average_pooling2d(layer_10,pool_size=(1,8),strides=1,padding="valid",activation=tf.nn.relu,name="layer_11")
		layer_12 = tf.layers.conv2d(layer_11,filters=1024,kernel_size=1,strides=1,padding="valid",activation=tf.nn.relu,name="layer_12")
		layer_13 = tf.layers.conv2d(layer_12,filters=num_speakers,kernel_size=1,strides=1,padding="valid",name="layer_13")
		self.metrics["final_result"] = tf.squeeze(layer_13)
	
	def compute_loss_and_metrics(self):
		entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.target,logits = self.metrics["final_result"])
		self.metrics["loss"] = tf.reduce_sum(entropy_loss)

	def compute_gradients_and_train_op(self):
		tvars = self.tvars = my_lib.get_scope_var(self.scope)
		my_lib.get_num_params(tvars)
		grads = tf.gradients(self.metrics["loss"], tvars)
		grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)

		self.metrics["grad_sum"] = tf.add_n([tf.reduce_sum(g) for g in grads])

		optimizer = tf.train.AdamOptimizer(learning_rate = self.config.learning_rate)
		self.train_op = optimizer.apply_gradients(
			zip(grads, tvars),
			global_step=self.global_step)

	def model_vars(self):
		return self.tvars

	def init_feed_dict(self):
		return {self.phase_train.name: True}


	def run_epoch(self, session,reader, is_training=False, verbose=False):
		start_time = time.time()
		epoch_metrics = {}
		keep_prob = 1.0
		fetches = {
			"loss": self.metrics["loss"],
			"grad_sum": self.metrics["grad_sum"]

		}
		if is_training:
			if verbose:
				print("\nTraining...")
			fetches["train_op"] = self.train_op
			keep_prob = self.config.keep_prob
		else:
			if verbose:
				print("\nEvaluating...")

		i, total_loss, grad_sum, total_words = 0, 0.0, 0.0, 0.0

		reader.start()

		i = 0
		batch = reader.next()
		while batch != None:
			i+=1
			session.run(self.inc)
			feed_dict = {}
			feed_dict[self.input.name] = batch["input"]
			feed_dict[self.target.name] = batch["target"]
			feed_dict[self.keep_prob.name] = keep_prob
			feed_dict[self.phase_train.name] = phase_train


			vals = session.run(fetches, feed_dict)
			total_loss += vals["loss"]
			grad_sum += vals["grad_sum"]
			percent_complete = i/batch["num_batches"]

			if verbose:
				print(
					"% Complete :", round(percent_complete, 0),
					"loss :", round((vals["loss"]/self.config.batch_size), 3), \
					"Gradient :", round(vals["grad_sum"],3))
			batch = reader.next()
			total_loss += vals["loss"]/config.batch_size
		return total_loss





