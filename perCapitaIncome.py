import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('percapita.csv')
df

%matplotlib inline
# %matplotlib inline will make your plot outputs appear and be stored within the notebook. This is a Line magics, another we have is Cell magics

plt.xlabel('year')                                            # Give labels to you x-axis
plt.ylabel('perCapitaIncome')                                 # Give labels to you y-axis
plt.scatter(df.year, df.capita, color='orange', marker='^')   # A scatter plot of y vs. x with varying marker size and/or color.

reg = linear_model.LinearRegression()  # Create an object for linear regression.
reg.fit(df[['year']], df.capita)
# LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares 
# between the observed targets in the dataset, and the targets predicted by the linear approximation.

plt.plot(df.year, reg.predict(df[['year']]), color='blue')
# Plot y versus x as lines and/or markers.

reg.predict([[2020]])     # Predict the perCapitaIncome for the year 2020

reg.coef_                 # Print coef_ : m in our case: y = mx + b.

reg.intercept_            # Print intercept_ : b in our case: y = mx + b.

