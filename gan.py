import numpy as np
import cv2

from activations import *

epsilon = 10e-4
beta1 = 0.9
beta2 = 0.999

class GAN(object):

	def __init__(self):
		self.batch_size = 64
		self.epochs = 25
		self.learning_rate = 0.0002


		#init generator weights
		self.g_W0 = np.random.uniform(-1,1,(100,150))
		self.g_b0 = np.random.uniform(-1,1,(150))

		self.g_W1 = np.random.uniform(-1,1,(150,300))
		self.g_b1 = np.random.uniform(-1,1,(300))
		
		self.g_W2 = np.random.uniform(-1,1,(300,28*28))
		self.g_b2 = np.random.uniform(-1,1,(28*28))
		

		#init discriminator weights
		self.d_W0 = np.random.uniform(-1,1,(28*28,300))
		self.d_b0 = np.random.uniform(-1,1,(300))

		self.d_W1 = np.random.uniform(-1,1,(300,150))
		self.d_b1 = np.random.uniform(-1,1,(150))
		
		self.d_W2 = np.random.uniform(-1,1,(150,1))
		self.d_b2 = np.random.uniform(-1,1,(1))


		#Adam Optimizer Var
		self.v1,self.m1 = 0,0
		self.v2,self.m2 = 0,0
		self.v3,self.m3 = 0,0

		self.v4,self.m4 = 0,0
		self.v5,self.m5 = 0,0
		self.v6,self.m6 = 0,0

	def discriminator(self, img):
		self.d_input = np.reshape(img, (self.batch_size, -1))

		self.d_h0 = np.matmul(self.d_input, self.d_W0)# + self.d_b0
		self.d_h0 = sigmoid(self.d_h0)

		self.d_h1 = np.matmul(self.d_h0, self.d_W1)# + self.d_b1
		self.d_h1 = sigmoid(self.d_h1)

		self.d_out = np.matmul(self.d_h1, self.d_W2)# + self.d_b2

		return self.d_out, sigmoid(self.d_out)

	def generator(self, z):

		self.g_h0 = np.matmul(z, self.g_W0) + self.g_b0
		self.g_h0 = relu(self.g_h0)

		self.g_h1 = np.matmul(self.g_h0, self.g_W1)# + self.g_b1
		self.g_h1 = relu(self.g_h1)

		self.g_h2 = np.matmul(self.g_h1, self.g_W2)# + self.g_b2
		self.g_out = sigmoid(self.g_h2)

		self.g_out = np.reshape(self.g_h2, (self.batch_size, 28, 28))
		
		return self.g_h2, self.g_out

	# generator backpropagation
	def backprop_gen(self, logit, output):
		output = np.reshape(output, (self.batch_size, -1))
		output = -1.0/output

		logit = np.tile(logit, [1, output.shape[-1]])

		#Calculate gradients
		err = output*relu(logit, derivative=True)
		grad_W2 = np.matmul(self.g_h1.T, err)	

		err = np.matmul(err, self.g_W2.T)
		err = err*relu(self.g_h1,derivative=True)
		grad_W1 = np.matmul(self.g_h0.T, err)
		

		err = np.matmul(err, self.g_W1.T)
		err = err*sigmoid(self.g_h0,derivative=True)
		grad_W0 = np.matmul(self.z.T, err)

		#update weights (Adam)
		self.m4 = beta1 * self.m4 + (1 - beta1) * grad_W2
		self.v4 = beta2 * self.v4 + (1 - beta2) * grad_W2 ** 2

		self.m5 = beta1 * self.m5 + (1 - beta1) * grad_W1
		self.v5 = beta2 * self.v5 + (1 - beta2) * grad_W1 ** 2

		self.m6 = beta1 * self.m6 + (1 - beta1) * grad_W0
		self.v6 = beta2 * self.v6 + (1 - beta2) * grad_W0 ** 2

		
		self.g_W2 -= (self.learning_rate/(np.sqrt(self.v4 / (1-beta2)) + epsilon))*(self.m4/(1-beta1))
		self.g_W1 -= (self.learning_rate/(np.sqrt(self.v5 / (1-beta2)) + epsilon))*(self.m5/(1-beta1))
		self.g_W0 -= (self.learning_rate/(np.sqrt(self.v6 / (1-beta2)) + epsilon))*(self.m6/(1-beta1))
		
	# discriminator backpropagation 
	def backprop_dis(self, real_logit, real_output, fake_logit, fake_output):
		
		#Calculate gradients
		real_output = -1.0/real_output
		fake_output = 1.0/(1.0-fake_output+epsilon)

		#real input gradients
		err = real_output*sigmoid(real_logit, derivative=True)
		grad_real_W2 = np.matmul(self.d_h1.T, err)
		
		err = np.matmul(err, self.d_W2.T)
		err = err*relu(self.d_h1,derivative=True)
		grad_real_W1 = np.matmul(self.d_h0.T, err)
		
		err = np.matmul(err, self.d_W1.T)
		err = err*relu(self.d_h0,derivative=True)
		grad_real_W0 = np.matmul(self.d_input.T, err)
		
		#fake input gradients
		err = fake_output*sigmoid(fake_logit, derivative=True)
		grad_fake_W2 = np.matmul(self.d_h1.T, err)
		
		err = np.matmul(err, self.d_W2.T)
		err = err*relu(self.d_h1,derivative=True)
		grad_fake_W1 = np.matmul(self.d_h0.T, err)
		
		err = np.matmul(err, self.d_W1.T)
		err = err*relu(self.d_h0,derivative=True)
		grad_fake_W0 = np.matmul(self.d_input.T, err)
		
		#combine two gradients
		grad_W2 = grad_real_W2 + grad_fake_W2
		grad_W1 = grad_real_W1 + grad_fake_W1
		grad_W0 = grad_real_W0 + grad_fake_W0

		#update weights (Adam)
		self.m1 = beta1 * self.m1 + (1 - beta1) * grad_W2
		self.v1 = beta2 * self.v1 + (1 - beta2) * grad_W2 ** 2

		self.m2 = beta1 * self.m2 + (1 - beta1) * grad_W1
		self.v2 = beta2 * self.v2 + (1 - beta2) * grad_W1 ** 2

		self.m3 = beta1 * self.m3 + (1 - beta1) * grad_W0
		self.v3 = beta2 * self.v3 + (1 - beta2) * grad_W0 ** 2

		self.d_W2 -= (self.learning_rate/(np.sqrt(self.v1 / (1-beta2)) + epsilon))*(self.m1/(1-beta1))
		self.d_W1 -= (self.learning_rate/(np.sqrt(self.v2 / (1-beta2)) + epsilon))*(self.m2/(1-beta1))
		self.d_W0 -= (self.learning_rate/(np.sqrt(self.v3 / (1-beta2)) + epsilon))*(self.m3/(1-beta1))

	def train(self):
		
		#we don't need labels.
		#just read images
		trainX, _, train_size = mnist_reader()
		
		batch_idx = train_size//self.batch_size
		for epoch in range(self.epochs):
			for idx in range(batch_idx):
				#prepare batch and input vector z
				train_batch = trainX[idx*self.batch_size:idx*self.batch_size + self.batch_size]
				self.z = np.random.uniform(0,1,[self.batch_size,100])

				#forward pass
				g_logits, fake_img = self.generator(self.z)
				d_real_logits, d_real_output = self.discriminator(train_batch)
				d_fake_logits, d_fake_output = self.discriminator(fake_img)

				#calculate loss
				real_label = np.ones((self.batch_size, 1), dtype=np.float32)
				fake_label = np.zeros((self.batch_size, 1), dtype=np.float32)

				#cross entropy loss using sigmoid output
				#add epsilon in log to avoid overflow
				d_loss = -np.log(d_real_output+epsilon) + np.log(1 - d_fake_output+epsilon)

				g_loss = -np.log(d_fake_output+epsilon)

				#train discriminator
				#one for fake input, another for real input
				self.backprop_dis(d_fake_logits, d_fake_output, d_real_logits, d_real_output)
				
				#train generator twice
				self.backprop_gen(d_fake_logits, fake_img)
				self.backprop_gen(d_fake_logits, fake_img)

				img_tile(fake_img)

				print "Epoch [%d] Step [%d] G Loss:%.4f D Loss:%.4f"%(epoch, idx, np.sum(g_loss)/self.batch_size, np.sum(d_loss)/self.batch_size)


gan = GAN()
gan.train()	