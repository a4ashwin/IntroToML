from pandas import DataFrame
from pandas import read_csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from random import random
from random import randint
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN

#Loading Iris Data
url = "C:\Stuff\KU Study\EECS 690 Intro to Machine Learning\Assignments\AshwinRathore_Assignment5\imbalanced iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
dataset=dataset.drop([0])
#Part 1
array=dataset.values
X=array[:,0:4]
y=array[:,4]
adasyn = ADASYN()
X_resampled, y_resampled = adasyn.fit_resample(X, y)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_resampled, y_resampled, test_size=0.5, random_state=1)
model= MLPClassifier(max_iter=800)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
model.fit(X_validation,Y_validation)
predictions=np.append(predictions,model.predict(X_train))
Y_validation=np.append(Y_validation,Y_train)
print("Using ADASYN: ")
print("Accuracy:", accuracy_score(Y_validation, predictions))
print("Confusion matrix:")
print(confusion_matrix(Y_validation, predictions))
print()