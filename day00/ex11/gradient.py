import numpy as np

def linear_mse(x, y, theta):
	ret = 0
	for i in range(0, x.shape[0]):
		ret += (np.dot(theta, x[i]) - y[i]) ** 2
	return ret / x.shape[0]

def gradient(x, y, theta):
	ret = np.zeros((x.shape[1],1))
	for j in range(0, x.shape[1]):
		val = 0
		for i in range(0, x.shape[0]):
			val += (x[i].dot(theta) - y[i]) / x.shape[0] * x[i][j]
		ret[j][0] = val
	return ret



X = np.array([
[ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]])

Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3,0.5,-6])
print(gradient(X, Y, Z))