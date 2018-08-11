# Part 1: Import libratries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Part 2: Import dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values # Independent Data
y = dataset.iloc[:, 1].values #Dependent Data

# Part 5: Splitting into Testing set & Training set
from sklearn.model_selection import train_test_split
#x_train, x_test = train_test_split(x, test_size=0.3)
#y_train, y_test = train_test_split(y, test_size=0.3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

"""
# Part 6: Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""

# Fitting linear regression model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

x1 = pd.DataFrame(x)
y1 = pd.DataFrame(y)

#predicting the test set result
y_pred = regressor.predict(x_test)

#visualising the training/test set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('experience (years)')
plt.ylabel('salary ($)')
plt.show()

plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, y_pred, color='blue')
plt.title('Salary vs Experience (test set)')

