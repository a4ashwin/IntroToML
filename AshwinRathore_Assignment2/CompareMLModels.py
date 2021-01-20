from pandas import DataFrame
from pandas import read_csv
from pandas import concat
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.datasets import load_iris

#Loading Iris Data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Creating training and test set
array=dataset.values
X=array[:,0:4]
y=array[:,4]

#calculating overall accuracy
models = []
models.append(('Naive Baysian', GaussianNB()))
#models.append(('Linear Regression', LinearRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))


# evaluate each model in turn
results = []
names = []
for name, model in models:
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.5, random_state=1)
    print('%s' % (name))
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    predictions = model.predict(X_validation)
    model.fit(X_validation,Y_validation)
    predictions=np.append(predictions,model.predict(X_train))
    Y_validation=np.append(Y_validation,Y_train)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    
#evaluating Simple linear regression
model=LinearRegression()
y[y=='Iris-setosa']=1.
y[y=='Iris-versicolor']=2.
y[y=='Iris-virginica']=3.
y=y.astype(float)
X=X.astype(float)
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.5, random_state=1)
print("Linear Regression:")
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
model.fit(X_validation,Y_validation)
predictions=np.append(predictions,model.predict(X_train))
Y_validation=np.append(Y_validation,Y_train)
print(accuracy_score(Y_validation, predictions.round()))
print(confusion_matrix(Y_validation, predictions.round()))

#Linear regression with polynomial degree of 2
polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(X)
X_train, X_validation, Y_train, Y_validation = train_test_split(x_poly, y, test_size=0.5, random_state=1)
print("Linear Regression: with polynomial of degree 2")
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
model.fit(X_validation,Y_validation)
predictions=np.append(predictions,model.predict(X_train))
Y_validation=np.append(Y_validation,Y_train)
for i in range(len(predictions)):
    if predictions[i] > 3.0:
        predictions[i] = 3.0
print(accuracy_score(Y_validation, predictions.round()))
print(confusion_matrix(Y_validation, predictions.round()))

# Linear regression with polynomial degree of 3
polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(X)
X_train, X_validation, Y_train, Y_validation = train_test_split(x_poly, y, test_size=0.5, random_state=1)
print("Linear Regression: with polynomial of degree 3")
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
model.fit(X_validation,Y_validation)
predictions=np.append(predictions,model.predict(X_train))
Y_validation=np.append(Y_validation,Y_train)
for i in range(len(predictions)):
    if predictions[i] > 3.0:
        predictions[i] = 3.0
print(accuracy_score(Y_validation, predictions.round()))
print(confusion_matrix(Y_validation, predictions.round()))

