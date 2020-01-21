import numpy as np

def dot(x, y):
	if type(x).__module__ != 'numpy' or x.size == 0:
		return None
	if type(y).__module__ != 'numpy' or y.size == 0:
		return None
	if x.size != y.size:
		return None
	ret = 0
	for v1, v2 in zip(x, y):
		ret += v1 * v2
	return ret

def mat_vec_prod(x, y):
	if type(x).__module__ != 'numpy' or x.size == 0:
		return None
	if type(y).__module__ != 'numpy' or y.size == 0:
		return None
	if len(x.shape) == 2 and len(y.shape) == 2 and y.shape[1] == 1 and x.shape[1] == y.shape[0]:
		ret = np.zeros((x.shape[0],1))
		for arr, index in zip(x, range(0, x.shape[0])):
			ret[index] = dot(arr, y)
		return ret
	return None

W = np.array([
    [ -8, 8, -6, 14, 14, -9, -4],
    [ 2, -11, -2, -11, 14, -2, 14],
    [-13, -2, -5, 3, -8, -4, 13],
    [ 2, 13, -14, -15, -14, -15, 13],
    [ 2, -1, 12, 3, -7, -3, -6]])
X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((7,1))
print(X)
print(mat_vec_prod(W, X))