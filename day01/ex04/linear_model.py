import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MylinearRegression as MyLR

raw = pd.read_csv("are_blue_pills_magics.csv")
X = np.array(raw['Micrograms']).reshape(-1, 1)
Y = np.array(raw['Score']).reshape(-1, 1)
myLR1 = MyLR([[89.0], [-8.]])
myLR2 = MyLR([[105.], [-45.]])

Y_model1 = myLR1.predict_(X)
Y_model2 = myLR2.predict_(X)

Y1 = myLR1.predict_(X)

def predict_(theta, X):
	new = np.full((len(X), 1), 1)
	X = np.hstack((new, X))
	pred = X.dot(theta)
	return pred

def cost_elem_(theta, X, Y):
	pred = predict_(theta, X)
	ret = (pred - Y) ** 2
	return ret / (2 * X.shape[0])

def cost_(theta, X, Y):
	costs = cost_elem_(theta, X, Y)
	return costs.sum()

print(myLR1.mse_(X, Y))
myLR1.fit_(X, Y, 0.001, 20000)
print(myLR1.mse_(X, Y))

theta = myLR1.theta
t0 = theta[0][0]

the = np.linspace(-14, -4, 200)
y = cost_(np.array([[t0], [the]]), X, Y)
plt.plot(the, y, '-r', label='J(0)')
# plt.plot(X, Y, 'ro')
# x = np.linspace(0,7,100)
# y = myLR1.theta[0][0] + x * myLR1.theta[1][0]
# print(myLR1.theta)
# plt.plot(x, y, '-r', label='y=theta0 + theta1 * x')
plt.show()