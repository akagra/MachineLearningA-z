# simple linear regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) """



# Fitting simple linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


# Predicting the test set result
Y_pred = regressor.predict(X_test)

# visualizing training set results
plt.scatter(X_train,Y_train,color = "red")
plt.plot(X_train,regressor.predict(X_train),color = "blue")
plt.title("Salary vs experience(training set)")
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()

# visluazing test set results
plt.scatter(X_test,Y_test,color = "red")
plt.plot(X_train,regressor.predict(X_train),color = "blue")
plt.title("Salary vs experience(test set)")
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()