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

#Loading Iris Data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#Part 1
# Creating training and test set
array=dataset.values
X=array[:,0:4]
y=array[:,4]
model=tree.DecisionTreeClassifier()
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.5, random_state=1)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
model.fit(X_validation,Y_validation)
predictions=np.append(predictions,model.predict(X_train))
Y_validation=np.append(Y_validation,Y_train)
print("Part 1:")
print("Accuracy:", accuracy_score(Y_validation, predictions))
print("Confusion matrix:")
print(confusion_matrix(Y_validation, predictions))
print("List of features used:",names[:4])


#Part 2
print()
print()
print("Part 2: ")
M = mean(X.T, axis=1)
# center columns by subtracting column means
C = X - M
# calculate covariance matrix of centered matrix
V = cov(C.astype(float), rowvar=False)
values, vectors = eig(V)
print("Values: ",values)
print("Vectors: ",vectors)
ar = []
g=0
pov=0.0
sum=0.0
#generating PoV value
for value in values:
    sum+=value
    ar.append(g)
    pov=sum/np.sum(values)
    if pov>0.9:
        break
    g+=1
print("PoV: ",pov)
P = vectors.T.dot(C.T)
#Creating transformed dataset
X1= P.T[:,int(ar[0]):(int(ar[len(ar)-1])+1)]
X_train, X_validation, Y_train, Y_validation = train_test_split(X1, y, test_size=0.5, random_state=1)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
model.fit(X_validation,Y_validation)
predictions=np.append(predictions,model.predict(X_train))
Y_validation=np.append(Y_validation,Y_train)
print()
print("Accuracy:", accuracy_score(Y_validation, predictions))
print("Confusion matrix:")
print(confusion_matrix(Y_validation, predictions))
cols=[]
for str in ar:
    cols.append(names[int(str)])
#print("List of features used:",names[int(ar[0]):(int(ar[len(ar)-1])+1)])
print("List of features used:", cols)

print()
print()
print("Part 3:")
#Part 3: 
#creating dataframe with 8 features
dataset2 = np.concatenate((X, P.T), axis=1)
dataframe = pd.DataFrame(data=dataset2, columns=['sepal-length', 'sepal-width', 'petal-length', 'petal-width','lambda_1', 'lambda_2', 'lambda_3', 'lambda_4'])
dataset3=dataframe.copy()
print(dataframe.shape)
dataset4 = dataset3.copy()
dataset_reset = dataset3.copy()
random_features=1
reset_index=1
reset_accuracy=0.0
current_accuracy=0.0
p=0
#starting 100 iterations
for i in range(1, 101):
    p+=1
    m=randint(0, 1)
    #adding 1 feature to dataset
    if m==1:
        if np.size(dataset4, 1) < 7:
            while(True):
                q=randint(0, 7)
                if dataframe.columns[q] not in dataset4:
                    dataset4[dataframe.columns[q]]= dataframe[dataframe.columns[q]].copy()
                    break            
    #Deleting feature
    else:
        if np.size(dataset4, 1) > 1:
            q=randint(0, np.size(dataset4, 1)-1)
            dataset4.drop(dataset4.columns[q], axis=1, inplace=True)
    print()
    print("Iteration: " , i)
    print("value of i: ", p)
    print("Features used", dataset4.columns)
    X_train, X_validation, Y_train, Y_validation = train_test_split(dataset4, y, test_size=0.5, random_state=1)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    model.fit(X_validation,Y_validation)
    predictions=np.append(predictions,model.predict(X_train))
    Y_validation=np.append(Y_validation,Y_train)
    print('%s: %f' % ('Accuracy = ', accuracy_score(Y_validation, predictions)))
    #Updating accuracy score
    #veriying status
    if accuracy_score(Y_validation, predictions) > current_accuracy:
        current_accuracy=accuracy_score(Y_validation, predictions)
        dataset3=dataset4.copy()
        dataset_reset=dataset4.copy()
        reset_accuracy=current_accuracy
        print("Status: Improved")
    #Deciding to Accept or Discard
    else:
        reset_index+=1
        pr_accept = 2.718**(-i*(current_accuracy-accuracy_score(Y_validation, predictions))/current_accuracy)
        print('%s: %f' % ('Pr_accept = ', pr_accept))
        q=random()
        print('%s: %f' % ('Randon Uniform = ', q))
        if q>=pr_accept:
            print("Status: Discarded")
            continue
        else:
            current_accuracy=accuracy_score(Y_validation, predictions)
            dataset3=dataset4.copy()
            print("Status: Accepted")
    #Restarting the iterations
    if reset_index > 9:
        i=i-10
        p=0
        reset_index=0
        dataset4=dataset_reset.copy()
        dataset3=dataset_reset.copy()
        current_accuracy=reset_accuracy
        print("Status: Restart")

#Calculating final accuracy and Confusion matrix
X_train, X_validation, Y_train, Y_validation = train_test_split(dataset3, y, test_size=0.5, random_state=1)
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
model.fit(X_validation,Y_validation)
predictions=np.append(predictions,model.predict(X_train))
Y_validation=np.append(Y_validation,Y_train)
print()
print("Part 3:")
print("Accuracy:", accuracy_score(Y_validation, predictions))
print("Confusion matrix:")
print(confusion_matrix(Y_validation, predictions))
print("Features used", dataset3.columns)



#Part 4:
print()
print()
print("Part 4:")
Y_validation_final=[]
predictions_final=[]
features_final=[]
predictions
dataset2 = np.concatenate((X, P.T), axis=1)
dataframe = pd.DataFrame(data=dataset2, columns=['sepal-length', 'sepal-width', 'petal-length', 'petal-width','lambda_1', 'lambda_2', 'lambda_3', 'lambda_4'])
column_names_init={}
column_names_all={}
accuracy=[0.0, 0.1, 0.0, 0.0, 0.0]
#initializing the 5 original sets
column_names_init[1] = ['lambda_1', 'sepal-length', 'sepal-width', 'petal-length', 'petal-width']
column_names_init[2] = ['lambda_1', 'lambda_2', 'sepal-width', 'petal-length', 'petal-width']
column_names_init[3] = ['lambda_1', 'lambda_2', 'lambda_3', 'sepal-width', 'petal-length']
column_names_init[4] = ['lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'sepal-width']
column_names_init[5] = ['lambda_1', 'lambda_2', 'lambda_3', 'lambda_4', 'sepal-length']
column_names_all=column_names_init.copy()

#starting the loop of 100 iterations
for k in range(1, 101):
    if accuracy[0]==1.0:
        break
    v=6
    column_names_all=column_names_init.copy()
    #generating 20 sets by doing interection and union on every pair of original set
    for t in range(1,6):
        for u in range (t+1,6):
            column_names_all[v]=list(set(column_names_init.get(t)).union(set(column_names_init.get(u))))
            v=v+1
            column_names_all[v]=list(set(column_names_init.get(t)).intersection(set(column_names_init.get(u))))
            v=v+1
    
    #mutation of 25 sets
    v=26
    for t in range(1,26):
       
        n=column_names_all[t].copy()
        u=randint(1, 3)
        #adding a feature
        if u==1:
            if len(n) < 8:
                while(True):
                    q=randint(0, 7)
                    if dataframe.columns[q] not in n:
                        n.append(dataframe.columns[q])
                        break
        #deleting a feature
        if u==2:
            if len(n) > 1:
                q=randint(0,(len(n)-1))
                n.pop(q)
        #replacing a feature
        if u==3:
            temp=""
            if len(n) < 8 and len(n) > 0:
                while(True):
                    q=randint(0, 7)
                    if dataframe.columns[q] not in n:
                        temp = dataframe.columns[q]
                        break
                q=randint(0,(len(n)-1))
                n.pop(q)
                n.insert(q, temp)
        column_names_all[v]=n
        v=v+1
            
    # we have first 50 columns ready, we perform the evaluation of every set    
    for t in range(1,51):
        temp_df=dataframe[column_names_all.get(t)].copy()
        X_train, X_validation, Y_train, Y_validation = train_test_split(temp_df, y, test_size=0.5, random_state=1)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        model.fit(X_validation,Y_validation)
        predictions=np.append(predictions,model.predict(X_train))
        Y_validation=np.append(Y_validation,Y_train)
        #if accuracy is highest till now
        if accuracy_score(Y_validation, predictions) > accuracy[0]:
            accuracy[4] = accuracy[3]
            accuracy[3] = accuracy[2]
            accuracy[2] = accuracy[1]
            accuracy[1] = accuracy[0]
            accuracy[0] = accuracy_score(Y_validation, predictions)
            column_names_init[5]=column_names_init[4].copy()
            column_names_init[4]=column_names_init[3].copy()
            column_names_init[3]=column_names_init[2].copy()
            column_names_init[2]=column_names_init[1].copy()
            column_names_init[1]=column_names_all.get(t)
            Y_validation_final=Y_validation
            predictions_final=predictions
            features_final=column_names_all.get(t)    
            if accuracy[0]==1.0:
                break
            continue
        #if accuracy is second best till now
        elif accuracy_score(Y_validation, predictions) > accuracy[1]:
            accuracy[4] = accuracy[3]
            accuracy[3] = accuracy[2]
            accuracy[2] = accuracy[1]
            accuracy[1] = accuracy_score(Y_validation, predictions)
            column_names_init[5]=column_names_init[4].copy()
            column_names_init[4]=column_names_init[3].copy()
            column_names_init[3]=column_names_init[2].copy()
            column_names_init[2]=column_names_all.get(t)
            continue
        #if accuracy is third best till now
        elif accuracy_score(Y_validation, predictions) > accuracy[2]:
            accuracy[4] = accuracy[3]
            accuracy[3] = accuracy[2]
            accuracy[2] = accuracy_score(Y_validation, predictions)
            column_names_init[5]=column_names_init[4].copy()
            column_names_init[4]=column_names_init[3].copy()
            column_names_init[3]=column_names_all.get(t)
            continue
        #if accuracy is forth best till now
        elif accuracy_score(Y_validation, predictions) > accuracy[3]:
            accuracy[4] = accuracy[3]
            accuracy[3] = accuracy_score(Y_validation, predictions)
            column_names_init[5]=column_names_init[4].copy()
            column_names_init[4]=column_names_all.get(t)
            continue
        #if accuracy is fifth best till now
        elif accuracy_score(Y_validation, predictions) > accuracy[4]:
            accuracy[4] = accuracy_score(Y_validation, predictions)
            column_names_init[5]=column_names_all.get(t)
            continue
    
    print()
    print()
    print("Generation: ", k)
    print("Features: ", column_names_init.get(1))
    print("Accuracy score 1:", accuracy[0])
    print("Features: ", column_names_init.get(2))
    print("Accuracy score 2:", accuracy[1])
    print("Features: ", column_names_init.get(3))
    print("Accuracy score 3:", accuracy[2])
    print("Features: ", column_names_init.get(4))
    print("Accuracy score 4:", accuracy[3])
    print("Features: ", column_names_init.get(5))
    print("Accuracy score 5:", accuracy[4])

#printing the best accuracy and features
print()
print("Part 4: final results")
print("Accuracy:", accuracy_score(Y_validation_final, predictions_final))
print("Confusion matrix:")
print(confusion_matrix(Y_validation_final, predictions_final))
print("Features used", features_final)




