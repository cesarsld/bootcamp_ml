import numpy as np

class MylinearRegression():
	def __init__(self, theta):
		super().__init__()
		self.theta = theta

	def predict_(self, X):
		new = np.full((len(X), 1), 1)
		X = np.hstack((new, X))
		pred = X.dot(self.theta)
		return pred

	def cost_elem_(self, X, Y):
		pred = self.predict_(X)
		ret = (pred - Y) ** 2
		return ret / (2 * X.shape[0])

	def cost_(self, X, Y):
		costs = self.cost_elem_(X, Y)
		return costs.sum()

	def __vec_gradient(self, x, y):
		gradient = (x.dot(self.theta) - y) / len(x)
		gradient = x.T.dot(gradient)
		return gradient

	def fit_(self, X, Y, alpha, n_cycle):
		new = np.full((len(X), 1), 1)
		X = np.hstack((new, X))
		for n in range(n_cycle):
			self.theta = self.theta - alpha * self.__vec_gradient(X, Y)
		return self.theta

	def mse_(self, x, y):
		if type(x).__module__ != 'numpy' or x.size == 0:
			return None
		if type(y).__module__ != 'numpy' or y.size == 0:
			return None
		if x.shape[0] != y.shape[0]:
			return None
		return np.dot((self.predict_(x) - y).T, self.predict_(x) - y)[0][0] / y.shape[0]

# X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89.,144.]])
# Y = np.array([[23.], [48.], [218.]])
# mylr = MylinearRegression([[1.], [1.], [1.], [1.], [1]])
# print(mylr.predict_(X))

# print(mylr.cost_elem_(X,Y))

# print(mylr.cost_(X,Y))

# mylr.fit_(X, Y, alpha = 1.6e-4, n_cycle=200000)
# print(mylr.theta)

# print(mylr.predict_(X))

# print(mylr.cost_elem_(X,Y))
# print(mylr.cost_(X,Y))
