# David Sjöberg 
# Boosting using Gradient Classifier and the 10 most important features
# 
#Removes warning
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
import sklearn.metrics as skm
print('STARTING CALCULATIONS')
start = time.perf_counter()

#Imports the csv file for training
traindata = pd.read_csv('train.csv')

#Import csv file for training
testdata = pd.read_csv('test.csv')

#Wanted classification
ylabel = ['Lead']
#The n number of most important features, in order of most important to least
allfeatures =  ['Number words female','Number of female actors','Age Lead','Difference in words lead and co-lead','Age Co-Lead','Number of male actors','Number words male','Mean Age Female','Mean Age Male','Number of words lead','Total words','Gross','Year']

#The 9 features giving the highest accuracy
importantfeatures = ['Number words female','Number of female actors','Age Lead','Difference in words lead and co-lead','Age Co-Lead','Number of male actors','Number words male','Mean Age Female','Mean Age Male']

#Number of folds
n_folds = 10
kf = KFold(n_splits=n_folds)

#X and Y from the csv
X = traindata[importantfeatures].values
Y = traindata[ylabel].values

#Loads final testdata
X_data = testdata[importantfeatures].values

#Choosen classifier
clf = GradientBoostingClassifier(n_estimators=100)

scores = []
#Performs k-fold cross-validation
for train_index, test_index in kf.split(X):

    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

        
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = skm.accuracy_score(y_test, y_pred)
    scores.append(score)

crossval = cross_val_score(clf, X, Y, cv=kf, scoring ='f1_macro')
mean_CV = np.mean(crossval)

meanmissclassification = 1 - np.mean(scores)

y_pred_test = clf.predict(X_data)

end = time.perf_counter()

print(f'DONE')
print(f'Time taken: ', round(end-start, 3))
print(f'Accuracy was : ', round(np.mean(scores),3))
print(f'The estimated E_new was: ', round(meanmissclassification, 3))
print(f'Cross-val f1 score was: {round(mean_CV, 3)} with a std of {crossval.std()}')

print(f'Preditiction for testfile:')
print(pd.value_counts(y_pred_test))