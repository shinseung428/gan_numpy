import numpy as np
import os
import cv2

from activations import *

epsilon = 10e-8
beta1 = 0.9
beta2 = 0.99

class GAN(object):

	def __init__(self):
		self.batch_size = 36
		self.epochs = 500
		self.learning_rate = 0.0001

		#save result every 5000 steps
		#batch of image is processed each step
		
		self.img_path = "./images"
		if not os.path.exists(self.img_path):
			os.makedirs(self.img_path)

		self.timestep = 0

		#init generator weights (xavier init)
		self.g_W0 = np.random.randn(100,256).astype(np.float32) * np.sqrt(2.0/(100))
		self.g_b0 = np.zeros(256).astype(np.float32)

		self.g_W1 = np.random.randn(256,512).astype(np.float32) * np.sqrt(2.0/(256))
		self.g_b1 = np.zeros(512).astype(np.float32)
		
		self.g_W2 = np.random.randn(512,28*28).astype(np.float32) * np.sqrt(2.0/(512))
		self.g_b2 = np.zeros(28*28).astype(np.float32)
		

		#init discriminator weights (xavier init)
		self.d_W0 = np.random.randn(28*28,512).astype(np.float32) * np.sqrt(2.0/(28*28))
		self.d_b0 = np.zeros(512).astype(np.float32)

		self.d_W1 = np.random.randn(512,256).astype(np.float32) * np.sqrt(2.0/(512))
		self.d_b1 = np.zeros(256).astype(np.float32)
		
		self.d_W2 = np.random.randn(256,1).astype(np.float32) * np.sqrt(2.0/(256))
		self.d_b2 = np.zeros(1).astype(np.float32)

		#Adam Optimizer Vars for the Discriminator
		self.d_w0_m, self.d_w0_v = 0.0, 0.0
		self.d_b0_m, self.d_b0_v = 0.0, 0.0

		self.d_w1_m, self.d_w1_v = 0.0, 0.0
		self.d_b1_m, self.d_b1_v = 0.0, 0.0

		self.d_w2_m, self.d_w2_v = 0.0, 0.0
		self.d_b2_m, self.d_b2_v = 0.0, 0.0
		

		#Adam Optimizer Vars for the Generator
		self.g_w0_m, self.g_w0_v = 0.0, 0.0
		self.g_b0_m, self.g_b0_v = 0.0, 0.0

		self.g_w1_m, self.g_w1_v = 0.0, 0.0
		self.g_b1_m, self.g_b1_v = 0.0, 0.0

		self.g_w2_m, self.g_w2_v = 0.0, 0.0
		self.g_b2_m, self.g_b2_v = 0.0, 0.0


	def discriminator(self, img):
		#self.d_h{num}_l : hidden logit layer
		#self.d_h{num}_a : hidden activation layer

		self.d_input = np.reshape(img, (self.batch_size,-1))

		self.d_h0_l = self.d_input.dot(self.d_W0) + self.d_b0
		self.d_h0_a = lrelu(self.d_h0_l)

		self.d_h1_l = self.d_h0_a.dot(self.d_W1) + self.d_b1
		self.d_h1_a = lrelu(self.d_h1_l)

		self.d_out = self.d_h1_a.dot(self.d_W2) + self.d_b2

		return self.d_out, sigmoid(self.d_out)

	def generator(self, z):
		#self.g_h{num}_l : hidden logit layer
		#self.g_h{num}_a : hidden activation layer
		
		self.z = np.reshape(z, (self.batch_size, -1))

		self.g_h0_l = self.z.dot(self.g_W0) + self.g_b0
		self.g_h0_a = lrelu(self.g_h0_l)

		self.g_h1_l = self.g_h0_a.dot(self.g_W1) + self.g_b1
		self.g_h1_a = lrelu(self.g_h1_l)
		
		self.g_h2_l = self.g_h1_a.dot(self.g_W2) + self.g_b2
		self.g_h2_a = tanh(self.g_h2_l)

		self.g_out = np.reshape(self.g_h2_a, (self.batch_size, 28, 28, 1))
		
		return self.g_h2_l, self.g_out

	# generator backpropagation
	def backprop_gen(self, fake_logit, fake_output, fake_input):
		#fake_logit : logit output from the discriminator D(G(z))
		#fake_sig_output : sigmoid output from the discriminator D(G(z))
		
		#flatten fake image input
		fake_input = np.reshape(fake_input, (self.batch_size,-1))

		#calculate the derivative of the loss term -log(D(G(z)))
		g_loss = np.reshape(fake_output, (self.batch_size, -1))
		g_loss = -1.0/(g_loss+ epsilon)

		# calculate the gradients from the end of the discriminator
		# we calculate them but won't update the discriminator weights
		#######################################
		#		fake input gradients
		#		-log(D(G(z)))
		#######################################
		loss_deriv = g_loss*sigmoid(fake_logit, derivative=True)
		loss_deriv = loss_deriv.dot(self.d_W2.T)
		loss_deriv = loss_deriv*lrelu(self.d_h1_l, derivative=True)
		
		loss_deriv = loss_deriv.dot(self.d_W1.T)
		loss_deriv = loss_deriv*lrelu(self.d_h0_l, derivative=True)		
		
		loss_deriv = loss_deriv.dot(self.d_W0.T)
		loss_deriv = loss_deriv*tanh(self.g_h2_l, derivative=True)
		#Reached the end of the generator

		#Calculate generator gradients
		prev_layer = np.expand_dims(self.g_h1_a, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_W2 = np.mean(np.matmul(prev_layer,loss_deriv_), axis=0)
		grad_b2 = np.mean(loss_deriv_, axis=0)	

		loss_deriv = loss_deriv.dot(self.g_W2.T)
		loss_deriv = loss_deriv*lrelu(self.g_h1_l, derivative=True)
		prev_layer = np.expand_dims(self.g_h0_a, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_W1 = np.mean(np.matmul(prev_layer,loss_deriv_), axis=0)
		grad_b1 = np.mean(loss_deriv_, axis=0)		

		loss_deriv = loss_deriv.dot(self.g_W1.T)
		loss_deriv = loss_deriv*lrelu(self.g_h0_l, derivative=True)
		prev_layer = np.expand_dims(self.z, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_W0 = np.mean(np.matmul(prev_layer,loss_deriv_), axis=0)
		grad_b0 = np.mean(loss_deriv_, axis=0)

		#update weights using Adam Optimizer
		#calculate first and second momentum for each w,b in layers
		#g_W0
		self.g_w0_m = (beta1 * self.g_w0_m) + (1.0 - beta1) * grad_W0
		self.g_w0_v = (beta2 * self.g_w0_v) + (1.0 - beta2) * (grad_W0 ** 2)
		#d_b0
		self.g_b0_m = (beta1 * self.g_b0_m) + (1.0 - beta1) * grad_b0
		self.g_b0_v = (beta2 * self.g_b0_v) + (1.0 - beta2) * (grad_b0 ** 2)

		#d_W1
		self.g_w1_m = (beta1 * self.g_w1_m) + (1.0 - beta1) * grad_W1
		self.g_w1_v = (beta2 * self.g_w1_v) + (1.0 - beta2) * (grad_W1 ** 2)
		#d_b1
		self.g_b1_m = (beta1 * self.g_b1_m) + (1.0 - beta1) * grad_b1
		self.g_b1_v = (beta2 * self.g_b1_v) + (1.0 - beta2) * (grad_b1 ** 2)

		#d_W2
		self.g_w2_m = (beta1 * self.g_w2_m) + (1.0 - beta1) * grad_W2
		self.g_w2_v = (beta2 * self.g_w2_v) + (1.0 - beta2) * (grad_W2 ** 2)
		#d_b0
		self.g_b2_m = (beta1 * self.g_b2_m) + (1.0 - beta1) * grad_b2
		self.g_b2_v = (beta2 * self.g_b2_v) + (1.0 - beta2) * (grad_b2 ** 2)


		# #corrected
		# #g_W0
		# g_w0_m_corrected = self.g_w0_m/(1-(beta1**self.timestep)+epsilon)
		# g_w0_v_corrected = self.g_w0_v/(1-(beta2**self.timestep)+epsilon)
		# #g_b0
		# g_b0_m_corrected = self.g_b0_m/(1-(beta1**self.timestep)+epsilon)
		# g_b0_v_corrected = self.g_b0_v/(1-(beta2**self.timestep)+epsilon)

		# #g_W0
		# g_w1_m_corrected = self.g_w1_m/(1-(beta1**self.timestep)+epsilon)
		# g_w1_v_corrected = self.g_w1_v/(1-(beta2**self.timestep)+epsilon)
		# #g_b0
		# g_b1_m_corrected = self.g_b1_m/(1-(beta1**self.timestep)+epsilon)
		# g_b1_v_corrected = self.g_b1_v/(1-(beta2**self.timestep)+epsilon)

		# #g_W2
		# g_w2_m_corrected = self.g_w2_m/(1-(beta1**self.timestep)+epsilon)
		# g_w2_v_corrected = self.g_w2_v/(1-(beta2**self.timestep)+epsilon)
		# #g_b2
		# g_b2_m_corrected = self.g_b2_m/(1-(beta1**self.timestep)+epsilon)
		# g_b2_v_corrected = self.g_b2_v/(1-(beta2**self.timestep)+epsilon)

		#make update 
		#d_W0 and d_b0
		self.g_W0 = self.g_W0 - self.learning_rate*(self.g_w0_m/(np.sqrt(self.g_w0_v)+epsilon))
		self.g_b0 = self.g_b0 - self.learning_rate*(self.g_b0_m/(np.sqrt(self.g_b0_v)+epsilon))

		self.g_W1 = self.g_W1 - self.learning_rate*(self.g_w1_m/(np.sqrt(self.g_w1_v)+epsilon))
		self.g_b1 = self.g_b1 - self.learning_rate*(self.g_b1_m/(np.sqrt(self.g_b1_v)+epsilon))

		self.g_W2 = self.g_W2 - self.learning_rate*(self.g_w2_m/(np.sqrt(self.g_w2_v)+epsilon))
		self.g_b2 = self.g_b2 - self.learning_rate*(self.g_b2_m/(np.sqrt(self.g_b2_v)+epsilon))

	# discriminator backpropagation
	def backprop_dis(self, 
					 real_logit, real_output, real_input, 
					 fake_logit, fake_output, fake_input):
		# real_logit : real logit value before sigmoid activation function (real input)
		# real_output : Discriminator output in range 0~1 (real input)
		# real_input : real input image fed into the discriminator
		# fake_logit : fake logit value before sigmoid activation function (generated input)
		# fake_output : Discriminator output in range 0~1 (generated input) 
		# fake_input : fake input image fed into the discriminator

		real_input = np.reshape(real_input, (self.batch_size,-1))
		fake_input = np.reshape(fake_input, (self.batch_size,-1))

		#loss = -np.mean(log(D(x)) - log(1-D(G(z)))) 
		#Calculate gradients of the loss(amount to move)
		d_real_loss = -1.0/(real_output + epsilon)
		d_fake_loss = -1.0/(fake_output - 1.0 + epsilon)

		#start from the error in the last layer
		#######################################
		#		real input gradients
		#		-log(D(x))
		#######################################
		
		loss_deriv = d_real_loss*sigmoid(real_logit, derivative=True)
		prev_layer = np.expand_dims(self.d_h1_a, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_real_W2 = np.mean(np.matmul(prev_layer,loss_deriv_), axis=0)
		grad_real_b2 = np.mean(loss_deriv_, axis=0) 

		loss_deriv = loss_deriv.dot(self.d_W2.T)
		loss_deriv = loss_deriv*lrelu(self.d_h1_l, derivative=True)
		prev_layer = np.expand_dims(self.d_h0_a, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_real_W1 = np.mean(np.matmul(prev_layer,loss_deriv_), axis=0)
		grad_real_b1 = np.mean(loss_deriv_, axis=0)
			
		loss_deriv = loss_deriv.dot(self.d_W1.T)
		loss_deriv = loss_deriv*lrelu(self.d_h0_l, derivative=True)
		prev_layer = np.expand_dims(real_input, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_real_W0 = np.mean(np.matmul(prev_layer,loss_deriv_), axis=0)
		grad_real_b0 = np.mean(loss_deriv_, axis=0)

		#######################################
		#		fake input gradients
		#		-log(1 - D(G(z)))
		#######################################
		
		loss_deriv = d_fake_loss*sigmoid(fake_logit, derivative=True)
		prev_layer = np.expand_dims(self.d_h1_a, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_fake_W2 = np.mean(np.matmul(prev_layer,loss_deriv_), axis=0)
		grad_fake_b2 = np.mean(loss_deriv_, axis=0) 

		loss_deriv = loss_deriv.dot(self.d_W2.T)
		loss_deriv = loss_deriv*lrelu(self.d_h1_l, derivative=True)
		prev_layer = np.expand_dims(self.d_h0_a, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_fake_W1 = np.mean(np.matmul(prev_layer,loss_deriv_), axis=0)
		grad_fake_b1 = np.mean(loss_deriv_, axis=0)
			
		loss_deriv = loss_deriv.dot(self.d_W1.T)
		loss_deriv = loss_deriv*lrelu(self.d_h0_l, derivative=True)
		prev_layer = np.expand_dims(fake_input, axis=-1)
		loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
		grad_fake_W0 = np.mean(np.matmul(prev_layer,loss_deriv_), axis=0)
		grad_fake_b0 = np.mean(loss_deriv_, axis=0)


		#combine two gradients
		grad_W2 = grad_real_W2 + grad_fake_W2
		grad_b2 = grad_real_b2 + grad_fake_b2

		grad_W1 = grad_real_W1 + grad_fake_W1
		grad_b1 = grad_real_b1 + grad_fake_b1

		grad_W0 = grad_real_W0 + grad_fake_W0
		grad_b0 = grad_real_b0 + grad_fake_b0

		#update weights using Adam Optimizer
		#d_W0
		self.d_w0_m = (beta1 * self.d_w0_m) + (1.0 - beta1) * grad_W0
		self.d_w0_v = (beta2 * self.d_w0_v) + (1.0 - beta2) * (grad_W0 ** 2)
		#d_b0
		self.d_b0_m = (beta1 * self.d_b0_m) + (1.0 - beta1) * grad_b0
		self.d_b0_v = (beta2 * self.d_b0_v) + (1.0 - beta2) * (grad_b0 ** 2)

		#d_W1
		self.d_w1_m = (beta1 * self.d_w1_m) + (1.0 - beta1) * grad_W1
		self.d_w1_v = (beta2 * self.d_w1_v) + (1.0 - beta2) * (grad_W1 ** 2)
		#d_b1
		self.d_b1_m = (beta1 * self.d_b1_m) + (1.0 - beta1) * grad_b1
		self.d_b1_v = (beta2 * self.d_b1_v) + (1.0 - beta2) * (grad_b1 ** 2)

		#d_W2
		self.d_w2_m = (beta1 * self.d_w2_m) + (1.0 - beta1) * grad_W2
		self.d_w2_v = (beta2 * self.d_w2_v) + (1.0 - beta2) * (grad_W2 ** 2)
		#d_b0
		self.d_b2_m = (beta1 * self.d_b2_m) + (1.0 - beta1) * grad_b2
		self.d_b2_v = (beta2 * self.d_b2_v) + (1.0 - beta2) * (grad_b2 ** 2)


		# #corrected
		# #g_W0
		# d_w0_m_corrected = self.d_w0_m/(1-(beta1**self.timestep)+epsilon)
		# d_w0_v_corrected = self.d_w0_v/(1-(beta2**self.timestep)+epsilon)
		# #g_b0
		# d_b0_m_corrected = self.d_b0_m/(1-(beta1**self.timestep)+epsilon)
		# d_b0_v_corrected = self.d_b0_v/(1-(beta2**self.timestep)+epsilon)

		# #g_W0
		# d_w1_m_corrected = self.d_w1_m/(1-(beta1**self.timestep)+epsilon)
		# d_w1_v_corrected = self.d_w1_v/(1-(beta2**self.timestep)+epsilon)
		# #g_b0
		# d_b1_m_corrected = self.d_b1_m/(1-(beta1**self.timestep)+epsilon)
		# d_b1_v_corrected = self.d_b1_v/(1-(beta2**self.timestep)+epsilon)

		# #g_W2
		# d_w2_m_corrected = self.d_w2_m/(1-(beta1**self.timestep)+epsilon)
		# d_w2_v_corrected = self.d_w2_v/(1-(beta2**self.timestep)+epsilon)
		# #g_b2
		# d_b2_m_corrected = self.d_b2_m/(1-(beta1**self.timestep)+epsilon)
		# d_b2_v_corrected = self.d_b2_v/(1-(beta2**self.timestep)+epsilon)

		#make updates
		#d_W0 and d_b0
		self.d_W0 = self.d_W0 - self.learning_rate*(self.d_w0_m/(np.sqrt(self.d_w0_v)+epsilon))
		self.d_b0 = self.d_b0 - self.learning_rate*(self.d_b0_m/(np.sqrt(self.d_b0_v)+epsilon))

		self.d_W1 = self.d_W1 - self.learning_rate*(self.d_w1_m/(np.sqrt(self.d_w1_v)+epsilon))
		self.d_b1 = self.d_b1 - self.learning_rate*(self.d_b1_m/(np.sqrt(self.d_b1_v)+epsilon))

		self.d_W2 = self.d_W2 - self.learning_rate*(self.d_w2_m/(np.sqrt(self.d_w2_v)+epsilon))
		self.d_b2 = self.d_b2 - self.learning_rate*(self.d_b2_m/(np.sqrt(self.d_b2_v)+epsilon))


	def train(self):
		
		#we don't need labels.
		#just read images
		trainX, _, train_size = mnist_reader()
		
		total_step = 0
		decay = 0.001
		
		#set batch indices
		batch_idx = train_size//self.batch_size
		for epoch in range(self.epochs):
			for idx in range(batch_idx):
				# prepare batch and input vector z
				train_batch = trainX[idx*self.batch_size:idx*self.batch_size + self.batch_size]
				
				#z = np.random.uniform(-1,1,[self.batch_size,100])
				z = np.random.randn(self.batch_size, 100).astype(np.float32) * np.sqrt(2.0/(self.batch_size))

				
				# Forward pass
				g_logits, fake_img = self.generator(z)

				d_real_logits, d_real_output = self.discriminator(train_batch)
				d_fake_logits, d_fake_output = self.discriminator(fake_img)

				# cross entropy loss using sigmoid output
				# add epsilon in log to avoid overflow
				# maximize Discriminator loss = -np.mean(log(D(x)) - log(1-D(G(z))))
				d_loss = -np.log(d_real_output+epsilon) - np.log(1 - d_fake_output+epsilon)
				
				# Generator loss 
				# ver1 : minimize log(1 - D(G(z)))
				# ver2 : maximize -log(D(G(z))) #this is better
				g_loss = -np.log(d_fake_output+epsilon)


				# Backward pass
				# train discriminator
				# one for fake input, another for real input
				self.backprop_dis(d_real_logits, d_real_output, train_batch, d_fake_logits, d_fake_output, fake_img)
				
				# #train generator 
				self.backprop_gen(d_fake_logits, d_fake_output, fake_img)

				# train generator twice?
				g_logits, fake_img = self.generator(z)
				d_fake_logits, d_fake_output = self.discriminator(fake_img)
				self.backprop_gen(d_fake_logits, d_fake_output, fake_img)


				
				img_tile(np.array(fake_img), self.img_path, epoch, idx, "res", False)
				self.img = fake_img

				print "Epoch [%d] Step [%d] G Loss:%.4f D Loss:%.4f Real Ave.: %.4f Fake Ave.: %.4f"%(epoch, idx, np.mean(g_loss), np.mean(d_loss), np.mean(d_real_output), np.mean(d_fake_output))
				total_step += 1
				self.timestep += 1

			self.learning_rate = self.learning_rate*(1/(1+decay*epoch))
			#save image result
			img_tile(np.array(self.img), self.img_path, epoch, idx, "res", True)


gan = GAN()
gan.train()	