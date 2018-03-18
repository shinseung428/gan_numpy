import numpy as np
import os
import cv2

from activations import *

epsilon = 10e-4
beta1 = 0.9
beta2 = 0.999

class GAN(object):

	def __init__(self):
		self.batch_size = 64
		self.epochs = 100
		self.learning_rate = 0.0002

		#save result every 5000 steps
		#batch of image is processed each step
		self.checkpoint = 1
		self.img_path = "./images"
		if not os.path.exists(self.img_path):
			os.makedirs(self.img_path)



		#init generator weights
		self.g_W0 = np.random.randn(100,250).astype(np.float32) * np.sqrt(2.0/(100))
		self.g_b0 = np.zeros(250).astype(np.float32)

		self.g_W1 = np.random.randn(250,450).astype(np.float32) * np.sqrt(2.0/(250))
		self.g_b1 = np.zeros(450).astype(np.float32)
		
		self.g_W2 = np.random.randn(450,28*28).astype(np.float32) * np.sqrt(2.0/(300))
		self.g_b2 = np.zeros(28*28).astype(np.float32)
		

		#init discriminator weights
		self.d_W0 = np.random.randn(28*28,300).astype(np.float32) * np.sqrt(2.0/(28*28))
		self.d_b0 = np.zeros(300).astype(np.float32)

		self.d_W1 = np.random.randn(300,150).astype(np.float32) * np.sqrt(2.0/(300))
		self.d_b1 = np.zeros(150).astype(np.float32)
		
		self.d_W2 = np.random.randn(150,1).astype(np.float32) * np.sqrt(2.0/(150))
		self.d_b2 = np.zeros(1).astype(np.float32)

		#Adam Optimizer Vars for the Discriminator
		self.v1_w,self.m1_w = 0.0, 0.0
		self.v1_b,self.m1_b = 0.0, 0.0
		self.v2_w,self.m2_w = 0.0, 0.0
		self.v2_b,self.m2_b = 0.0, 0.0
		self.v3_w,self.m3_w = 0.0, 0.0
		self.v3_b,self.m3_b = 0.0, 0.0

		#Adam Optimizer Vars for the Generator
		self.v4_w,self.m4_w = 0.0, 0.0
		self.v4_b,self.m4_b = 0.0, 0.0
		self.v5_w,self.m5_w = 0.0, 0.0
		self.v5_b,self.m5_b = 0.0, 0.0
		self.v6_w,self.m6_w = 0.0, 0.0
		self.v6_b,self.m6_b = 0.0, 0.0
		self.v7_w,self.m7_w = 0.0, 0.0
		self.v7_b,self.m7_b = 0.0, 0.0

	def discriminator(self, img):
		self.d_input = np.reshape(img, (self.batch_size,-1))
		
		# self.d_h0 = np.matmul(self.d_input, self.d_W0) + self.d_b0
		self.d_h0 = self.d_input.dot(self.d_W0) + self.d_b0
		self.d_h0 = relu(self.d_h0)

		# self.d_h1 = np.matmul(self.d_h0, self.d_W1) + self.d_b1
		self.d_h1 = self.d_h0.dot(self.d_W1) + self.d_b1
		self.d_h1 = relu(self.d_h1)

		# self.d_out = np.matmul(self.d_h1, self.d_W2) + self.d_b2
		self.d_out = self.d_h1.dot(self.d_W2) + self.d_b2

		return self.d_out, sigmoid(self.d_out)

	def generator(self, z):
		self.z = np.reshape(z, (self.batch_size, -1))

		# self.g_h0 = np.matmul(z, self.g_W0) + self.g_b0
		self.g_h0 = z.dot(self.g_W0) + self.g_b0
		self.g_h0 = relu(self.g_h0)

		# self.g_h1 = np.matmul(self.g_h0, self.g_W1) + self.g_b1
		self.g_h1 = self.g_h0.dot(self.g_W1) + self.g_b1
		self.g_h1 = relu(self.g_h1)
		
		# self.g_h2 = np.matmul(self.g_h1, self.g_W2) + self.g_b2
		self.g_h2 = self.g_h1.dot(self.g_W2) + self.g_b2
		self.g_out = tanh(self.g_h2)

		self.g_out = np.reshape(self.g_out, (self.batch_size, 28, 28))
		
		return self.g_h2, self.g_out

	# generator backpropagation
	def backprop_gen(self, fake_logit, fake_sig_output, fake_input):
		#logit : sigmoid output from the discriminator D(G(x))
		
		#flatten fake image input
		fake_input = np.reshape(fake_input, (self.batch_size,-1))
		
		#calculate the loss_derivative of the loss term -log(D(G(x)))
		d_loss = np.reshape(fake_sig_output, (self.batch_size, -1))
		d_loss = -1.0/(d_loss+epsilon)


		#calculate gradients from the end of the discriminator
		#we calculate them but won't update the discriminator weights
		#fake input gradients
		loss_deriv = d_loss*sigmoid(fake_logit, derivative=True)
		
		loss_deriv = loss_deriv.dot(self.d_W2.T)
		loss_deriv = loss_deriv*relu(self.d_h1, derivative=True)

		loss_deriv = loss_deriv.dot(self.d_W1.T)
		loss_deriv = loss_deriv*relu(self.d_h0, derivative=True)		
		
		loss_deriv = loss_deriv.dot(self.d_W0.T)
		#Reached the end of the generator

		#Calculate gradients
		loss_deriv = loss_deriv*tanh(self.g_h2, derivative=True)
		grad_W2 = self.g_h1.T.dot(loss_deriv)
		grad_b2 = np.mean(loss_deriv, axis=0)	

		loss_deriv = loss_deriv.dot(self.g_W2.T)
		loss_deriv = loss_deriv*relu(self.g_h1, derivative=True)
		grad_W1 = self.g_h0.T.dot(loss_deriv)
		grad_b1 = np.mean(loss_deriv, axis=0)		

		loss_deriv = loss_deriv.dot(self.g_W1.T)
		loss_deriv = loss_deriv*relu(self.g_h0, derivative=True)
		grad_W0 = self.z.T.dot(loss_deriv)
		grad_b0 = np.mean(loss_deriv, axis=0)

		#update weights Adam Optimizer
		#g_W0
		self.m4_w = (beta1 * self.m4_w) + (1.0 - beta1) * grad_W2
		self.v4_w = (beta2 * self.v4_w) + (1.0 - beta2) * (grad_W2 ** 2)
		#g_b0
		self.m4_b = (beta1 * self.m4_b) + (1.0 - beta1) * grad_b2
		self.v4_b = (beta2 * self.v4_b) + (1.0 - beta2) * (grad_b2 ** 2)

		#g_W1
		self.m5_w = (beta1 * self.m5_w) + (1.0 - beta1) * grad_W1
		self.v5_w = (beta2 * self.v5_w) + (1.0 - beta2) * (grad_W1 ** 2)
		#g_b1
		self.m5_b = (beta1 * self.m5_b) + (1.0 - beta1) * grad_b1
		self.v5_b = (beta2 * self.v5_b) + (1.0 - beta2) * (grad_b1 ** 2)

		#g_W2
		self.m6_w = (beta1 * self.m6_w) + (1.0 - beta1) * grad_W0
		self.v6_w = (beta2 * self.v6_w) + (1.0 - beta2) * (grad_W0 ** 2)
		#g_b2
		self.m6_b = (beta1 * self.m6_b) + (1.0 - beta1) * grad_b0
		self.v6_b = (beta2 * self.v6_b) + (1.0 - beta2) * (grad_b0 ** 2)
		
		#make update
		self.g_W2 = self.g_W2 - (self.learning_rate/(np.sqrt(self.v4_w / (1.0-beta2)) + epsilon))*(self.m4_w/(1.0-beta1))
		self.g_W1 = self.g_W1 - (self.learning_rate/(np.sqrt(self.v5_w / (1.0-beta2)) + epsilon))*(self.m5_w/(1.0-beta1))
		self.g_W0 = self.g_W0 - (self.learning_rate/(np.sqrt(self.v6_w / (1.0-beta2)) + epsilon))*(self.m6_w/(1.0-beta1))
		
		self.g_b2 = self.g_b2 - np.reshape((self.learning_rate/(np.sqrt(self.v4_b / (1.0-beta2)) + epsilon))*(self.m4_b/(1.0-beta1)), -1)
		self.g_b1 = self.g_b1 - np.reshape((self.learning_rate/(np.sqrt(self.v5_b / (1.0-beta2)) + epsilon))*(self.m5_b/(1.0-beta1)), -1)
		self.g_b0 = self.g_b0 - np.reshape((self.learning_rate/(np.sqrt(self.v6_b / (1.0-beta2)) + epsilon))*(self.m6_b/(1.0-beta1)), -1)

	# discriminator backpropagation
	def backprop_dis(self, real_logit, real_output, real_input, fake_logit, fake_output, fake_input):
		# real_logit : real logit value before sigmoid activation function (real input)
		# real_output : Discriminator output in range 0~1 (real input)
		# real_input : real input image fed into the discriminator
		# fake_logit : fake logit value before sigmoid activation function (generated input)
		# fake_output : Discriminator output in range 0~1 (generated input) 
		# fake_input : fake input image fed into the discriminator

		real_input = np.reshape(real_input, (self.batch_size,-1))
		fake_input = np.reshape(fake_input, (self.batch_size,-1))

		#Calculate gradients of the loss(amount to move)
		d_real_loss = -np.mean(1.0/(real_output+epsilon), axis=0)
		d_fake_loss = np.mean(1.0/(1.0-fake_output+epsilon), axis=0)

		#real input gradients
		loss_deriv = d_real_loss*sigmoid(real_logit, derivative=True)
		grad_real_W2 = self.d_h1.T.dot(loss_deriv)
		grad_real_b2 = np.mean(loss_deriv, axis=0) 

		loss_deriv = loss_deriv.dot(self.d_W2.T)
		loss_deriv = loss_deriv*relu(self.d_h1, derivative=True)
		grad_real_W1 = self.d_h0.T.dot(loss_deriv)
		grad_real_b1 = np.mean(loss_deriv, axis=0)
		
		

		loss_deriv = loss_deriv.dot(self.d_W1.T)
		loss_deriv = loss_deriv*relu(self.d_h0, derivative=True)
		grad_real_W0 = real_input.T.dot(loss_deriv)
		grad_real_b0 = np.mean(loss_deriv, axis=0)

		#fake input gradients
		loss_deriv = d_fake_loss*sigmoid(fake_logit, derivative=True)
		grad_fake_W2 = self.d_h1.T.dot(loss_deriv)
		grad_fake_b2 = np.mean(loss_deriv, axis=0)
		
		loss_deriv = loss_deriv.dot(self.d_W2.T)
		loss_deriv = loss_deriv*relu(self.d_h1, derivative=True)
		grad_fake_W1 = self.d_h0.T.dot(loss_deriv)
		grad_fake_b1 = np.mean(loss_deriv, axis=0)

		loss_deriv = loss_deriv.dot(self.d_W1.T)
		loss_deriv = loss_deriv*relu(self.d_h0, derivative=True)
		grad_fake_W0 = fake_input.T.dot(loss_deriv)
		grad_fake_b0 = np.mean(loss_deriv, axis=0)

		#combine two gradients
		grad_W2 = grad_real_W2 + grad_fake_W2
		grad_b2 = grad_real_b2 + grad_fake_b2

		grad_W1 = grad_real_W1 + grad_fake_W1
		grad_b1 = grad_real_b1 + grad_fake_b1

		grad_W0 = grad_real_W0 + grad_fake_W0
		grad_b0 = grad_real_b0 + grad_fake_b0

		#update weights using Adam Optimizer
		#d_W0
		self.m1_w = (beta1 * self.m1_w) + (1.0 - beta1) * grad_W2
		self.v1_w = (beta2 * self.v1_w) + (1.0 - beta2) * (grad_W2 ** 2)
		#d_b0
		self.m1_b = (beta1 * self.m1_b) + (1.0 - beta1) * grad_b2
		self.v1_b = (beta2 * self.v1_b) + (1.0 - beta2) * (grad_b2 ** 2)

		#d_W1
		self.m2_w = (beta1 * self.m2_w) + (1.0 - beta1) * grad_W1
		self.v2_w = (beta2 * self.v2_w) + (1.0 - beta2) * (grad_W1 ** 2)
		#d_b1
		self.m2_b = (beta1 * self.m2_b) + (1.0 - beta1) * grad_b1
		self.v2_b = (beta2 * self.v2_b) + (1.0 - beta2) * (grad_b1 ** 2)

		#d_W2
		self.m3_w = (beta1 * self.m3_w) + (1.0 - beta1) * grad_W0
		self.v3_w = (beta2 * self.v3_w) + (1.0 - beta2) * (grad_W0 ** 2)
		#d_b2
		self.m3_b = (beta1 * self.m3_b) + (1.0 - beta1) * grad_b0
		self.v3_b = (beta2 * self.v3_b) + (1.0 - beta2) * (grad_b0 ** 2)

		#make update 
		self.d_W2 = self.d_W2 - (self.learning_rate/(np.sqrt(self.v1_w / (1.0-beta2)) + epsilon))*(self.m1_w/(1.0-beta1))
		self.d_W1 = self.d_W1 - (self.learning_rate/(np.sqrt(self.v2_w / (1.0-beta2)) + epsilon))*(self.m2_w/(1.0-beta1))
		self.d_W0 = self.d_W0 - (self.learning_rate/(np.sqrt(self.v3_w / (1.0-beta2)) + epsilon))*(self.m3_w/(1.0-beta1))

		self.d_b2 = self.d_b2 - np.reshape((self.learning_rate/(np.sqrt(self.v1_b / (1.0-beta2)) + epsilon))*(self.m1_b/(1.0-beta1)), -1)
		self.d_b1 = self.d_b1 - np.reshape((self.learning_rate/(np.sqrt(self.v2_b / (1.0-beta2)) + epsilon))*(self.m2_b/(1.0-beta1)), -1)
		self.d_b0 = self.d_b0 - np.reshape((self.learning_rate/(np.sqrt(self.v3_b / (1.0-beta2)) + epsilon))*(self.m3_b/(1.0-beta1)), -1)

		return grad_fake_W0, grad_fake_b0

	def train(self):
		
		#we don't need labels.
		#just read images
		trainX, _, train_size = mnist_reader()
		
		total_step = 0
		#set batch indices
		batch_idx = train_size//self.batch_size
		for epoch in range(self.epochs):
			for idx in range(batch_idx):
				#prepare batch and input vector z
				train_batch = trainX[idx*self.batch_size:idx*self.batch_size + self.batch_size]
				z = np.random.uniform(-1,1,[self.batch_size,100])

				#process each element in the batch
				g_loss_sum, d_loss_sum = 0.0, 0.0
				
				#forward pass
				g_logits, fake_img = self.generator(z)

				d_real_logits, d_real_output = self.discriminator(train_batch)
				d_fake_logits, d_fake_output = self.discriminator(fake_img)


				#cross entropy loss using sigmoid output
				#add epsilon in log to avoid overflow
				#Discriminator loss = -log(D(x)) + log(1-D(G(x)))
				d_loss = -np.log(d_real_output+epsilon) + np.log(1 - d_fake_output+epsilon)
				
				#Generator loss = -log(D(G(x)))
				g_loss = -np.log(d_fake_output+epsilon)


				# #train discriminator
				# #one for fake input, another for real input
				self.backprop_dis(d_real_logits, d_real_output, train_batch, d_fake_logits, d_fake_output, fake_img)
									
				# #train generator twice
				self.backprop_gen(d_fake_logits, d_fake_output, fake_img)
				g_logits, fake_img = self.generator(z)
				d_fake_logits, d_fake_output = self.discriminator(fake_img)
				self.backprop_gen(d_fake_logits, d_fake_output, fake_img)

				

				if total_step%self.checkpoint == 0:
					img_tile(np.array(fake_img), self.img_path, epoch, idx)

				print "Epoch [%d] Step [%d] G Loss:%.4f D Loss:%.4f"%(epoch, idx, np.mean(g_loss), np.mean(d_loss))
				total_step += 1



gan = GAN()
gan.train()	