# Xgboost

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
import tensorflow as tf
import keras   


   
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1= LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2= LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categories= [1])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Xgboost to training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)


#Predicting the test set result
y_pred = classifier.predict(X_test)



# Importing the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train,y = y_train,cv =10)
accuracies.mean()
accuracies.std()
