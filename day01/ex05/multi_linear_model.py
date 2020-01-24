import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MylinearRegression as MyLR

raw = pd.read_csv("spacecraft_data.csv")
age = np.array(raw['Age']).reshape(-1, 1)
X = np.array(raw[['Age','Thrust_power','Terameters']])
Y = np.array(raw['Sell_price']).reshape(-1, 1)
myLR1 = MyLR([[1.0],[1.0],[1.0],[1.0]])
print(myLR1.mse_(X, Y))

# Y_model1 = myLR1.predict_(X)
myLR1.fit_(X, Y, 0.00001, 1000000)

print(myLR1.mse_(X, Y))
plt.plot(age, Y, 'ro')
plt.plot(age, myLR1.predict_(X), 'bo')
# x = np.linspace(0,200,1000)
# y = myLR1.theta[0][0] + x * myLR1.theta[1][0]
# print(myLR1.theta)
# plt.plot(x, y, '-r', label='y=theta0 + theta_age * x')
plt.show()