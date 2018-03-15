import numpy as np


def mnist_reader():
	def one_hot(label, output_dim):
		one_hot = np.zeros((len(label), output_dim))
		
		for idx in range(0,len(label)):
			one_hot[idx, label[idx]] = 1
		
		return one_hot


	#Training Data
	f = open('./data/train-images.idx3-ubyte')
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32) / 127.5 - 1


	f = open('./data/train-labels.idx1-ubyte')
	loaded = np.fromfile(file=f, dtype=np.uint8)
	trainY = loaded[8:].reshape((60000)).astype(np.int32)
	trainY = one_hot(trainY, 10)


	# #Test Data
	# f = open('./data/t10k-images.idx3-ubyte')
	# loaded = np.fromfile(file=f, dtype=np.uint8)
	# testX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32) / / 127.5 - 1

	# f = open('./data/t10k-labels.idx1-ubyte')
	# loaded = np.fromfile(file=f, dtype=np.uint8)
	# testY = loaded[8:].reshape((10000)).astype(np.int32)
	# testY = one_hot(testY, 10)

	return trainX, trainY, len(trainX) #, testX, testY, len(testX)

def sigmoid(input, derivative=False):
	# for batch in input:
	res = 1/(1+np.exp(-input))

	if derivative:
		return res*(1-res)

	return res

def relu(input, derivative=False):
	res = []
	for batch in input:
		batch[batch<=0] = 0
		res.append(batch)
	
	return np.array(res)


def tanh(input, derivative=False):
	res = np.tanh(input)
	return res


