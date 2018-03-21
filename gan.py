import numpy as np
import os
import cv2

from utils import *

epsilon = 10e-8
beta1 = 0.9
beta2 = 0.999

class GAN(object):

	def __init__(self):
		self.batch_size = 64
		self.epochs = 500
		self.timestep = 1
		self.learning_rate = 0.0002
		

		self.img_path = "./images"
		if not os.path.exists(self.img_path):
			os.makedirs(self.img_path)

		# Xavier initialization is used to initialize the weights
		# https://theneuralperspective.com/2016/11/11/weights-initialization/
		# init generator weights
		self.g_W0 = np.random.randn(100,128).astype(np.float32) * np.sqrt(2.0/(100))
		self.g_b0 = np.zeros(128).astype(np.float32)

		self.g_W1 = np.random.randn(128,784).astype(np.float32) * np.sqrt(2.0/(128))
		self.g_b1 = np.zeros(784).astype(np.float32)

		# init discriminator weights 
		self.d_W0 = np.random.randn(784,128).astype(np.float32) * np.sqrt(2.0/(784))
		self.d_b0 = np.zeros(128).astype(np.float32)

		self.d_W1 = np.random.randn(128,1).astype(np.float32) * np.sqrt(2.0/(128))
		self.d_b1 = np.zeros(1).astype(np.float32)
		

		# Adam Optimizer Vars for the Discriminator
		# d_W0 and d_b0 
		self.d_w0_m, self.d_w0_v = 0.0, 0.0
		self.d_b0_m, self.d_b0_v = 0.0, 0.0
		# W1 and b1
		self.d_w1_m, self.d_w1_v = 0.0, 0.0
		self.d_b1_m, self.d_b1_v = 0.0, 0.0


		#Adam Optimizer Vars for the Generator
		# g_W0 and g_b0
		self.g_w0_m, self.g_w0_v = 0.0, 0.0
		self.g_b0_m, self.g_b0_v = 0.0, 0.0
		# g_W1 and g_b1
		self.g_w1_m, self.g_w1_v = 0.0, 0.0
		self.g_b1_m, self.g_b1_v = 0.0, 0.0

	def discriminator(self, img):
		#self.d_h{num}_l : hidden logit layer
		#self.d_h{num}_a : hidden activation layer

		self.d_input = np.reshape(img, (self.batch_size,-1))

		self.d_h0_l = self.d_input.dot(self.d_W0) + self.d_b0
		self.d_h0_l = instance_norm(self.d_h0_l)
		self.d_h0_a = lrelu(self.d_h0_l)

		self.d_h1_l = self.d_h0_a.dot(self.d_W1) + self.d_b1
		self.d_h1_a = sigmoid(self.d_h1_l)
		self.d_out = self.d_h1_a

		return self.d_h1_l, self.d_out

	def generator(self, z):
		#self.g_h{num}_l : hidden logit layer
		#self.g_h{num}_a : hidden activation layer
		
		self.z = np.reshape(z, (self.batch_size, -1))

		self.g_h0_l = self.z.dot(self.g_W0) + self.g_b0
		self.g_h0_l = instance_norm(self.g_h0_l)
		self.g_h0_a = lrelu(self.g_h0_l)

		self.g_h1_l = self.g_h0_a.dot(self.g_W1) + self.g_b1
		self.g_h1_a = tanh(self.g_h1_l)

		self.g_out = np.reshape(self.g_h1_a, (self.batch_size, 28, 28))
		
		return self.g_h1_l, self.g_out


	# generator backpropagation
	def backprop_gen(self, fake_logit, fake_output, fake_input, idx):
		# fake_logit : logit output from the discriminator D(G(z))
		# fake_output : sigmoid output from the discriminator D(G(z))
		# idx : element to backpropagate

		# flatten fake image input
		fake_input = np.reshape(fake_input, (self.batch_size,-1))

		# calculate the derivative of the loss term -log(D(G(z)))
		g_loss = np.reshape(fake_output, (self.batch_size, -1))
		g_loss = -1.0/(g_loss+ epsilon)

		# calculate the gradients from the end of the discriminator
		# we calculate them but won't update the discriminator weights
		loss_deriv = g_loss[idx]*sigmoid(fake_logit[idx], derivative=True)
		loss_deriv = loss_deriv.dot(self.d_W1.T)
		loss_deriv = loss_deriv*lrelu(self.d_h0_l[idx], derivative=True)		
		
		loss_deriv = loss_deriv.dot(self.d_W0.T)
		loss_deriv = loss_deriv*tanh(self.g_h1_l[idx], derivative=True)
		# Reached the end of the generator

		# Calculate generator gradients
		#######################################
		#		fake input gradients
		#		-log(D(G(z)))
		#######################################
		prev_layer = np.expand_dims(self.g_h0_a[idx], axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=0)
		grad_W1 = prev_layer.dot(loss_deriv_)
		grad_b1 = loss_deriv

		loss_deriv = loss_deriv.dot(self.g_W1.T)
		loss_deriv = loss_deriv*lrelu(self.g_h0_l[idx], derivative=True)
		prev_layer = np.expand_dims(self.z[idx], axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=0)
		grad_W0 = prev_layer.dot(loss_deriv_)
		grad_b0 = loss_deriv


		# update weights using Adam Optimizer
		# https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
		# g_W0
		self.g_w0_m = (beta1 * self.g_w0_m) + (1.0 - beta1) * grad_W0
		self.g_w0_v = (beta2 * self.g_w0_v) + (1.0 - beta2) * (grad_W0 ** 2)
		# g_b0
		self.g_b0_m = (beta1 * self.g_b0_m) + (1.0 - beta1) * grad_b0
		self.g_b0_v = (beta2 * self.g_b0_v) + (1.0 - beta2) * (grad_b0 ** 2)

		# g_W1
		self.g_w1_m = (beta1 * self.g_w1_m) + (1.0 - beta1) * grad_W1
		self.g_w1_v = (beta2 * self.g_w1_v) + (1.0 - beta2) * (grad_W1 ** 2)
		# g_b1
		self.g_b1_m = (beta1 * self.g_b1_m) + (1.0 - beta1) * grad_b1
		self.g_b1_v = (beta2 * self.g_b1_v) + (1.0 - beta2) * (grad_b1 ** 2)

		# corrected
		# g_W0
		g_w0_m_corrected = self.g_w0_m/(1-(beta1**self.timestep)+epsilon)
		g_w0_v_corrected = self.g_w0_v/(1-(beta2**self.timestep)+epsilon)
		# g_b0
		g_b0_m_corrected = self.g_b0_m/(1-(beta1**self.timestep)+epsilon)
		g_b0_v_corrected = self.g_b0_v/(1-(beta2**self.timestep)+epsilon)

		# g_W1
		g_w1_m_corrected = self.g_w1_m/(1-(beta1**self.timestep)+epsilon)
		g_w1_v_corrected = self.g_w1_v/(1-(beta2**self.timestep)+epsilon)
		# g_b1
		g_b1_m_corrected = self.g_b1_m/(1-(beta1**self.timestep)+epsilon)
		g_b1_v_corrected = self.g_b1_v/(1-(beta2**self.timestep)+epsilon)

		# make updates
		self.g_W0 = self.g_W0 - self.learning_rate*(g_w0_m_corrected/(np.sqrt(g_w0_v_corrected)+epsilon))
		self.g_b0 = self.g_b0 - self.learning_rate*(g_b0_m_corrected/(np.sqrt(g_b0_v_corrected)+epsilon))

		self.g_W1 = self.g_W1 - self.learning_rate*(g_w1_m_corrected/(np.sqrt(g_w1_v_corrected)+epsilon))
		self.g_b1 = self.g_b1 - self.learning_rate*(g_b1_m_corrected/(np.sqrt(g_b1_v_corrected)+epsilon))


	# discriminator backpropagation
	def backprop_dis(self, 
					 real_logit, real_output, real_input, 
					 fake_logit, fake_output, fake_input,
					 idx):
		# real_logit : real logit value before sigmoid activation function (real input)
		# real_output : Discriminator output in range 0~1 (real input)
		# real_input : real input image fed into the discriminator
		# fake_logit : fake logit value before sigmoid activation function (generated input)
		# fake_output : Discriminator output in range 0~1 (generated input) 
		# fake_input : fake input image fed into the discriminator

		real_input = np.reshape(real_input, (self.batch_size,-1))
		fake_input = np.reshape(fake_input, (self.batch_size,-1))

		# Discriminator loss = -np.mean(log(D(x)) - log(1-D(G(z)))) 
		# Calculate gradients of the loss
		d_real_loss = -1.0/(real_output + epsilon)
		d_fake_loss = -1.0/(fake_output - 1.0 + epsilon)

		# start from the error in the last layer
		#######################################
		#		real input gradients
		#		-log(D(x))
		#######################################
		loss_deriv = d_real_loss[idx]*sigmoid(real_logit[idx], derivative=True)
		prev_layer = np.expand_dims(self.d_h0_a[idx], axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=0)
		grad_real_W1 =  prev_layer.dot(loss_deriv_)
		grad_real_b1 = loss_deriv
			
		loss_deriv = loss_deriv.dot(self.d_W1.T)
		loss_deriv = loss_deriv*lrelu(self.d_h0_l[idx], derivative=True)
		prev_layer = np.expand_dims(real_input[idx], axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=0)
		grad_real_W0 = prev_layer.dot(loss_deriv_)
		grad_real_b0 = loss_deriv 
		
		#######################################
		#		fake input gradients
		#		-log(1 - D(G(z)))
		#######################################
		loss_deriv = d_fake_loss[idx]*sigmoid(fake_logit[idx], derivative=True)
		prev_layer = np.expand_dims(self.d_h0_a[idx], axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=0)
		grad_fake_W1 = prev_layer.dot(loss_deriv_)
		grad_fake_b1 = loss_deriv
			
		loss_deriv = loss_deriv.dot(self.d_W1.T)
		loss_deriv = loss_deriv*lrelu(self.d_h0_l[idx], derivative=True)
		prev_layer = np.expand_dims(fake_input[idx], axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=0)		
		grad_fake_W0 = prev_layer.dot(loss_deriv_)
		grad_fake_b0 = loss_deriv


		# combine two gradients (real + fake)
		grad_W1 = grad_real_W1 + grad_fake_W1
		grad_b1 = grad_real_b1 + grad_fake_b1

		grad_W0 = grad_real_W0 + grad_fake_W0
		grad_b0 = grad_real_b0 + grad_fake_b0


		# update weights using Adam Optimizer
		# https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
		# d_W0
		self.d_w0_m = (beta1 * self.d_w0_m) + (1.0 - beta1) * grad_W0
		self.d_w0_v = (beta2 * self.d_w0_v) + (1.0 - beta2) * (grad_W0 ** 2)
		# d_b0
		self.d_b0_m = (beta1 * self.d_b0_m) + (1.0 - beta1) * grad_b0
		self.d_b0_v = (beta2 * self.d_b0_v) + (1.0 - beta2) * (grad_b0 ** 2)

		# d_W1
		self.d_w1_m = (beta1 * self.d_w1_m) + (1.0 - beta1) * grad_W1
		self.d_w1_v = (beta2 * self.d_w1_v) + (1.0 - beta2) * (grad_W1 ** 2)
		# d_b1
		self.d_b1_m = (beta1 * self.d_b1_m) + (1.0 - beta1) * grad_b1
		self.d_b1_v = (beta2 * self.d_b1_v) + (1.0 - beta2) * (grad_b1 ** 2)

		# corrected
		# g_W0
		d_w0_m_corrected = self.d_w0_m/(1-(beta1**self.timestep)+epsilon)
		d_w0_v_corrected = self.d_w0_v/(1-(beta2**self.timestep)+epsilon)
		# g_b0
		d_b0_m_corrected = self.d_b0_m/(1-(beta1**self.timestep)+epsilon)
		d_b0_v_corrected = self.d_b0_v/(1-(beta2**self.timestep)+epsilon)

		# g_W1
		d_w1_m_corrected = self.d_w1_m/(1-(beta1**self.timestep)+epsilon)
		d_w1_v_corrected = self.d_w1_v/(1-(beta2**self.timestep)+epsilon)
		# g_b1
		d_b1_m_corrected = self.d_b1_m/(1-(beta1**self.timestep)+epsilon)
		d_b1_v_corrected = self.d_b1_v/(1-(beta2**self.timestep)+epsilon)

	
		# make updates
		self.d_W0 = self.d_W0 - self.learning_rate*(d_w0_m_corrected/(np.sqrt(d_w0_v_corrected)+epsilon))
		self.d_b0 = self.d_b0 - self.learning_rate*(d_b0_m_corrected/(np.sqrt(d_b0_v_corrected)+epsilon))

		self.d_W1 = self.d_W1 - self.learning_rate*(d_w1_m_corrected/(np.sqrt(d_w1_v_corrected)+epsilon))
		self.d_b1 = self.d_b1 - self.learning_rate*(d_b1_m_corrected/(np.sqrt(d_b1_v_corrected)+epsilon))


	def train(self):
		#we don't need labels.
		#just read images and shuffle
		trainX, _, train_size = mnist_reader()
		np.random.shuffle(trainX)

		#set batch indices
		batch_idx = train_size//self.batch_size
		for epoch in range(self.epochs):
			for idx in range(batch_idx):
				# prepare batch and input vector z
				train_batch = trainX[idx*self.batch_size:idx*self.batch_size + self.batch_size]

				#z = np.random.uniform(-1,1,[self.batch_size,100])
				z = np.random.randn(self.batch_size, 100).astype(np.float32) * np.sqrt(2.0/(self.batch_size))

				################################
				#		Forward Pass
				################################
				g_logits, fake_img = self.generator(z)

				d_real_logits, d_real_output = self.discriminator(train_batch)
				d_fake_logits, d_fake_output = self.discriminator(fake_img)

				# cross entropy loss using sigmoid output
				# add epsilon to avoid overflow
				# maximize Discriminator loss = -np.mean(log(D(x)) - log(1-D(G(z))))
				d_loss = -np.log(d_real_output+epsilon) - np.log(1 - d_fake_output+epsilon)
				
				# Generator loss 
				# ver1 : minimize log(1 - D(G(z)))
				# ver2 : maximize -log(D(G(z))) #this is better
				g_loss = -np.log(d_fake_output+epsilon)


				################################
				#		Backward Pass
				################################
				# for every result in the batch
				# calculate gradient and update the weights using Adam
				for index in range(0,self.batch_size):					
					# discriminator backward pass	
					# one for fake input, another for real input
					self.backprop_dis(d_real_logits, d_real_output, train_batch, d_fake_logits, d_fake_output, fake_img, index)
					
					# generator backward pass 
					self.backprop_gen(d_fake_logits, d_fake_output, fake_img, index)

					# train generator twice?
					# g_logits, fake_img = self.generator(z)
					# d_fake_logits, d_fake_output = self.discriminator(fake_img)
					# self.backprop_gen(d_fake_logits, d_fake_output, fake_img)

				#show res images as tile
				img_tile(np.array(fake_img), self.img_path, epoch, idx, "res", False)
				self.img = fake_img

				print "Epoch [%d] Step [%d] G Loss:%.4f D Loss:%.4f Real Ave.: %.4f Fake Ave.: %.4f"%(epoch, idx, np.mean(g_loss), np.mean(d_loss), np.mean(d_real_output), np.mean(d_fake_output))
				self.timestep += 1

			#save image result every epoch
			img_tile(np.array(self.img), self.img_path, epoch, idx, "res", True)


gan = GAN()
gan.train()	