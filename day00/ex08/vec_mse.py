import numpy as np

def sum(x, f):
	"""Computes the sum of a non-empty numpy.ndarray onto wich a function is
applied element-wise, using a for-loop.
    Args:
     x: has to be an numpy.ndarray, a vector.
     f: has to be a function, a function to apply element-wise to the
vector.
    Returns:
     The sum as a float.
     None if x is an empty numpy.ndarray or if f is not a valid function.
    Raises:
     This function should not raise any Exception.
"""
	if type(x).__module__ != 'numpy' or x.size == 0:
		return None
	if not hasattr(f, '__call__'):
		return None
	ret = 0
	for y in np.nditer(x):
		ret += f(y)
	return ret


def vec_mse(y, y_hat):
	if type(y_hat).__module__ != 'numpy' or y_hat.size == 0:
		return None
	if type(y).__module__ != 'numpy' or y.size == 0:
		return None
	if y_hat.shape[0] != y.shape[0]:
		return None
	return np.dot(y_hat - y, y_hat - y) / y.shape[0]

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(vec_mse(X, Y))