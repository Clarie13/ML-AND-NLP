import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("/kaggle/input/datasets/neha3919/houseprices/house_prices.csv")
df

%matplotlib inline
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df.area, df.price, color='red', marker='+')

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

reg.predict([[5000]])

reg.coef_ #this  is your value of m in formula y= m*x + b
reg.intercept_ #this is the value of b


d= pd.read_csv("areas.csv")
p = reg.predict(d)
d['prices'] = p
d.to_csv("prediction.csv", index=False)