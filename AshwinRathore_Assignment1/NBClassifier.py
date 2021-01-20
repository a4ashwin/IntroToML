from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Loading Iris Data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Creating training and test set
array=dataset.values
X=array[:,0:4]
y=array[:,4]
X_training, X_validation, Y_training, Y_validation = train_test_split(X, y, test_size=0.5, random_state=1)

#calculating overall accuracy
kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
cv_results = cross_val_score(GaussianNB(), X_training, Y_training, cv=kfold, scoring='accuracy')
print('%s: %f (%f)' % ('NB Accuracy = ', cv_results.mean(), cv_results.std()))

#Applying Naive Bayesian Model
model=GaussianNB()
model.fit(X_training,Y_training)
predictions=model.predict(X_validation)

# Printing Presicion, Confusion matrix and F1
print('%s: %f' % ('Precision =' , accuracy_score(Y_validation,predictions)))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

