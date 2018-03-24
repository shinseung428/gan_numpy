import numpy as np
import os
import cv2

from utils import *

epsilon = 10e-8
class GAN(object):
	def __init__(self, numbers):
		self.numbers = numbers

		self.epochs = 100
		self.batch_size = 64
		self.learning_rate = 0.001
		self.decay = 0.001

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

	def discriminator(self, img):
		#self.d_h{num}_l : hidden logit layer
		#self.d_h{num}_a : hidden activation layer

		self.d_input = np.reshape(img, (self.batch_size,-1))

		self.d_h0_l = self.d_input.dot(self.d_W0) + self.d_b0
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
		self.g_h0_a = lrelu(self.g_h0_l)

		self.g_h1_l = self.g_h0_a.dot(self.g_W1) + self.g_b1
		self.g_h1_a = tanh(self.g_h1_l)

		self.g_out = np.reshape(self.g_h1_a, (self.batch_size, 28, 28))

		return self.g_h1_l, self.g_out


	# generator backpropagation
	def backprop_gen(self,
					 fake_logit,
					 fake_output,
					 fake_input,
					 ):
		# fake_logit : logit output from the discriminator D(G(z))
		# fake_output : sigmoid output from the discriminator D(G(z))

		# flatten fake image input
		fake_input = np.reshape(fake_input, (self.batch_size,-1))

		# calculate the derivative of the loss term -log(D(G(z)))
		g_loss = np.reshape(fake_output, (self.batch_size, -1))
		g_loss = -1.0/(g_loss+ epsilon)

		# calculate the gradients from the end of the discriminator
		# we calculate them but won't update the discriminator weights
		loss_deriv = g_loss*sigmoid(fake_logit, derivative=True)
		loss_deriv = loss_deriv.dot(self.d_W1.T)

		loss_deriv = loss_deriv*lrelu(self.d_h0_l, derivative=True)
		loss_deriv = loss_deriv.dot(self.d_W0.T)
		# Reached the end of the generator

		# Calculate generator gradients
		#######################################
		#		fake input gradients
		#		-log(D(G(z)))
		#######################################
		loss_deriv = loss_deriv*tanh(self.g_h1_l, derivative=True)
		prev_layer = np.expand_dims(self.g_h0_a, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_W1 = np.matmul(prev_layer,loss_deriv_)
		grad_b1 = loss_deriv

		loss_deriv = loss_deriv.dot(self.g_W1.T)
		loss_deriv = loss_deriv*lrelu(self.g_h0_l, derivative=True)
		prev_layer = np.expand_dims(self.z, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_W0 = np.matmul(prev_layer,loss_deriv_)
		grad_b0 = loss_deriv

		# calculated all the gradients in the batch
		# now make updates
		for idx in range(self.batch_size):
			self.g_W0 = self.g_W0 - self.learning_rate*grad_W0[idx]
			self.g_b0 = self.g_b0 - self.learning_rate*grad_b0[idx]

			self.g_W1 = self.g_W1 - self.learning_rate*grad_W1[idx]
			self.g_b1 = self.g_b1 - self.learning_rate*grad_b1[idx]


	# discriminator backpropagation
	def backprop_dis(self,
					 real_logit, real_output, real_input,
					 fake_logit, fake_output, fake_input,
					 ):
		# real_logit : real logit value before sigmoid activation function (real input)
		# real_output : Discriminator output in range 0~1 (real input)
		# real_input : real input image fed into the discriminator
		# fake_logit : fake logit value before sigmoid activation function (generated input)
		# fake_output : Discriminator output in range 0~1 (generated input)
		# fake_input : fake input image fed into the discriminator

		real_input = np.reshape(real_input, (self.batch_size,-1))
		fake_input = np.reshape(fake_input, (self.batch_size,-1))

		# Discriminator loss = -np.mean(log(D(x)) + log(1-D(G(z))))
		# Calculate gradients of the loss
		d_real_loss = -1.0/(real_output + epsilon)
		d_fake_loss = -1.0/(fake_output - 1.0 + epsilon)

		# start from the error in the last layer
		#######################################
		#		real input gradients
		#		-log(D(x))
		#######################################
		loss_deriv = d_real_loss*sigmoid(real_logit, derivative=True)
		prev_layer = np.expand_dims(self.d_h0_a, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_real_W1 =  np.matmul(prev_layer,loss_deriv_)
		grad_real_b1 = loss_deriv

		loss_deriv = loss_deriv.dot(self.d_W1.T)
		loss_deriv = loss_deriv*lrelu(self.d_h0_l, derivative=True)
		prev_layer = np.expand_dims(real_input, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_real_W0 = np.matmul(prev_layer,loss_deriv_)
		grad_real_b0 = loss_deriv

		#######################################
		#		fake input gradients
		#		-log(1 - D(G(z)))
		#######################################
		loss_deriv = d_fake_loss*sigmoid(fake_logit, derivative=True)
		prev_layer = np.expand_dims(self.d_h0_a, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_fake_W1 = np.matmul(prev_layer,loss_deriv_)
		grad_fake_b1 = loss_deriv

		loss_deriv = loss_deriv.dot(self.d_W1.T)
		loss_deriv = loss_deriv*lrelu(self.d_h0_l, derivative=True)
		prev_layer = np.expand_dims(fake_input, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_fake_W0 = np.matmul(prev_layer,loss_deriv_)
		grad_fake_b0 = loss_deriv


		# combine two gradients (real + fake)
		grad_W1 = grad_real_W1 + grad_fake_W1
		grad_b1 = grad_real_b1 + grad_fake_b1

		grad_W0 = grad_real_W0 + grad_fake_W0
		grad_b0 = grad_real_b0 + grad_fake_b0

		# calculated all the gradients in the batch
		# now make updates
		for idx in range(self.batch_size):
			self.d_W0 = self.d_W0 - self.learning_rate*grad_W0[idx]
			self.d_b0 = self.d_b0 - self.learning_rate*grad_b0[idx]

			self.d_W1 = self.d_W1 - self.learning_rate*grad_W1[idx]
			self.d_b1 = self.d_b1 - self.learning_rate*grad_b1[idx]


	def train(self):
		#we don't need labels.
		#just read images and shuffle
		trainX, _, train_size = mnist_reader(self.numbers)
		np.random.shuffle(trainX)

		#set batch indices
		batch_idx = train_size//self.batch_size
		for epoch in range(self.epochs):
			for idx in range(batch_idx):
				# prepare batch and input vector z
				train_batch = trainX[idx*self.batch_size:idx*self.batch_size + self.batch_size]
				#ignore batch if there are insufficient elements
				if train_batch.shape[0] != self.batch_size:
					break

				z = np.random.uniform(-1,1,[self.batch_size,100])

				################################
				#		Forward Pass
				################################
				g_logits, fake_img = self.generator(z)

				d_real_logits, d_real_output = self.discriminator(train_batch)
				d_fake_logits, d_fake_output = self.discriminator(fake_img)

				# cross entropy loss using sigmoid output
				# add epsilon to avoid overflow
				# maximize Discriminator loss = -np.mean(log(D(x)) + log(1-D(G(z))))
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

				# discriminator backward pass
				# one for fake input, another for real input
				self.backprop_dis(d_real_logits, d_real_output, train_batch, d_fake_logits, d_fake_output, fake_img)

				# generator backward pass
				self.backprop_gen(d_fake_logits, d_fake_output, fake_img)
				# train generator twice?
				#g_logits, fake_img = self.generator(z)
				#d_fake_logits, d_fake_output = self.discriminator(fake_img)
				#self.backprop_gen(d_fake_logits, d_fake_output, fake_img)

				#show res images as tile
				#if you don't want to see the result at every step, comment line below
				img_tile(np.array(fake_img), self.img_path, epoch, idx, "res", False)
				self.img = fake_img


				print("Epoch [%d] Step [%d] G Loss:%.4f D Loss:%.4f Real Ave.: %.4f Fake Ave.: %.4f lr: %.4f"%(epoch, idx, np.mean(g_loss), np.mean(d_loss), np.mean(d_real_output), np.mean(d_fake_output), self.learning_rate))

			#update learning rate every epoch
			self.learning_rate = self.learning_rate * (1.0/(1.0 + self.decay*epoch))

			#save image result every epoch
			img_tile(np.array(self.img), self.img_path, epoch, idx, "res", True)




########################
#program starts here
#select numbers to generate
#recommended to put a single number to get faster result
numbers = [2]

gan = GAN(numbers)
gan.train()
