import numpy as np

def sigmoid(input, derivative=False):
	res = []
	for batch in input:
		# res.append(1/(1+np.exp(-batch)))
		tmp = 1/(1+np.exp(-batch))
		if derivative:
			res.append(tmp*(1-tmp))
		else:
			res.append(tmp)

	res = np.array(res)

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

