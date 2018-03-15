import numpy as np

from activations import *

class GAN(object):

	def __init__(self):
		self.batch_size = 32
		self.epochs = 25
		self.learning_rate = 0.001


		#init generator weights
		self.g_W0 = np.random.random((100,150)) * 2 - 1
		self.g_b0 = np.random.random((150)) * 2 - 1

		self.g_W1 = np.random.random((150,300)) * 2 - 1
		self.g_b1 = np.random.random((300)) * 2 - 1
		
		self.g_W2 = np.random.random((300,28*28)) * 2 - 1
		self.g_b2 = np.random.random((28*28)) * 2 - 1
		

		#init discriminator weights
		self.d_W0 = np.random.random((28*28,300)) * 2 - 1
		self.d_b0 = np.random.random((300)) * 2 - 1

		self.d_W1 = np.random.random((300,150)) * 2 - 1
		self.d_b1 = np.random.random((150)) * 2 - 1
		
		self.d_W2 = np.random.random((150,1)) * 2 - 1
		self.d_b2 = np.random.random((1)) * 2 - 1

	def discriminator(self, img):
		flattened = np.reshape(img, (self.batch_size, -1))

		self.h0 = np.matmul(flattened, self.d_W0) + self.d_b0
		# self.h0 = relu(self.h0)

		self.h1 = np.matmul(self.h0, self.d_W1) + self.d_b1
		# self.h1 = relu(self.h1)

		self.h2 = np.matmul(self.h1, self.d_W2) + self.d_b2

		return self.h2, sigmoid(self.h2)

	def generator(self, z):

		self.h0 = np.matmul(z, self.g_W0) + self.g_b0
		# self.h0 = relu(self.h0)

		self.h1 = np.matmul(self.h0, self.g_W1) + self.g_b1
		# self.h1 = relu(self.h1)

		self.h2 = np.matmul(self.h1, self.g_W2) + self.g_b2
		self.h2 = tanh(self.h2)

		self.out = np.reshape(self.h2, (self.batch_size, 28, 28))
		
		return self.out


	def train(self):
		#we don't need labels
		trainX, _, train_size = mnist_reader()
		
		batch_idx = train_size//self.batch_size
		for idx in range(batch_idx):
			train_batch = trainX[idx*self.batch_size:idx*self.batch_size + self.batch_size]

			z = np.random.random([self.batch_size,100]) * 2 - 1

			#forward pass
			fake_img = self.generator(z)
			real_logits, real_output = self.discriminator(train_batch)
			fake_logits, fake_output = self.discriminator(fake_img)

			#calculate loss
			real_label = np.ones((self.batch_size, 1))
			fake_label = np.zeros((self.batch_size, 1))

			#cross entropy loss using sigmoid output
			real_loss = -(real_label*np.log(real_output) + (1-real_label)*np.log(1-real_output))
			fake_loss = -(fake_label*np.log(fake_output) + (1-fake_label)*np.log(1-fake_output))

			print real_output
			
			# print "Real:", np.sum(real_loss)/self.batch_size
			# print "Fake:", np.sum(fake_loss)/self.batch_size
			
			#train discriminator


			#train generator



gan = GAN()
gan.train()	