import numpy as np

def predict_(theta, X):
	new = np.full((len(X), 1), 1)
	X = np.hstack((new, X))
	pred = X.dot(theta)
	return pred

def vec_linear_mse(x, y, theta):
	y_hat_y = predict_(theta, x) - y
	print(predict_(theta, x))
	print(y_hat_y)
	return np.dot(y_hat_y, y_hat_y.T) / len(x)


def cost_elem_old(theta, X, Y):
	pred = predict_(theta, X)
	ret = np.zeros((X.shape[0], 1))
	for pred_y, y, i in zip(pred, Y, range(0, X.shape[0])):
		ret[i] = (pred_y - y) ** 2
	return ret / (2 * X.shape[0])

def cost_elem_(theta, X, Y):
	pred = predict_(theta, X)
	ret = (pred - Y) ** 2
	return ret / (2 * X.shape[0])

def cost_(theta, X, Y):
	costs = cost_elem_(theta, X, Y)
	return costs.sum()

X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
Y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
print(cost_elem_(theta1, X1, Y1))

print(cost_(theta1, X1, Y1))

X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
Y2 = np.array([[19.], [42.], [67.], [93.]])
print(cost_elem_(theta2, X2, Y2))

print(cost_(theta2, X2, Y2))
