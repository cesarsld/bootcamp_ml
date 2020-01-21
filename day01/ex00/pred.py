import numpy as np

def predict_(theta, X):
	pred = np.zeros((X.shape[0], 1))
	for n in range(0, X.shape[0]):
		for j in range(0, X.shape[1]):
			pred[n][0] = theta[0][0]
			for k in range(1, theta.shape[0]):
				pred[n][0] += X[n][k - 1] * theta[k][0]
	return pred
		
X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
print(predict_(theta1, X1))

X3 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,80.]])
theta3 = np.array([[0.05], [1.], [1.], [1.]])
print(predict_(theta3, X3))