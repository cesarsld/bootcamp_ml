import numpy as np

def predict_(theta, X):
	new = np.full((len(X), 1), 1)
	X = np.hstack((new, X))
	pred = X.dot(theta)
	return pred
		
# X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
# theta1 = np.array([[2.], [4.]])
# print(predict_(theta1, X1))

# X3 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,80.]])
# theta3 = np.array([[0.05], [1.], [1.], [1.]])
# print(predict_(theta3, X3))