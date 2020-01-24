import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MylinearRegression as MyLR


raw = pd.read_csv("spacecraft_data.csv")
X = np.array(raw[['Age','Thrust_power','Terameters']])
X1 = np.array(raw[['Age','Thrust_power','Terameters']])
Y = np.array(raw['Sell_price']).reshape(-1, 1)
Y1 = np.array(raw['Sell_price']).reshape(-1, 1)
myLR_ne = MyLR([[1.], [1.], [1.], [1.]])
myLR_lgd = MyLR([[1.], [1.], [1.], [1.]])
myLR_lgd.fit_(X1,Y1, alpha = 5e-5, n_cycle = 100000)
myLR_ne.normalequation_(X,Y)

print(myLR_ne.mse_(X,Y))
print(myLR_lgd.mse_(X1, Y1))