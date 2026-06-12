import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("house_prices_multivariate.csv")
df

#data preprocesing
import math
median_bedrooms = math.floor(df.bedrooms.median())
median_bedrooms

df.bedrooms = df.bedrooms.fillna(median_bedrooms)
df

#regression
reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms', 'age']], df.price)

reg.predict([[3000,4,15]])
reg.coef_
reg.intercept_
