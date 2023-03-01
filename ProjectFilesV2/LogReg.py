# Boosting using LogReg and the 12 most important features
# 
#Removes warning
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import sklearn.metrics as skm
print('STARTING CALCULATIONS')
start = time.perf_counter()

#Imports the csv file for training
traindata = pd.read_csv('train.csv')

#Wanted classification
ylabel = ['Lead']
#The n number of most important features, in order of most important to least
allfeatures =  ['Number words female','Number of female actors','Age Lead','Difference in words lead and co-lead','Age Co-Lead','Number of male actors','Number words male','Mean Age Female','Mean Age Male','Number of words lead','Total words','Gross','Year']

#The 12 features giving the highest accuracy
importantfeatures = ['Number words female','Number of female actors','Age Lead','Difference in words lead and co-lead','Age Co-Lead','Number of male actors','Number words male','Mean Age Female','Mean Age Male','Number of words lead','Total words','Gross']

#Number of folds
n_folds = 10
kf = KFold(n_splits=n_folds)

#X and Y from the csv
X = traindata[importantfeatures].values
Y = np.where(traindata[ylabel].values == 'Female', 1, 0)

#Choosen classifier
clf = LogisticRegression(penalty='l1', C=100, solver = "liblinear")

#Performs k-fold cross-validation
for train_index, test_index in kf.split(X):

    # Split the data into training and testing sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

        
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
accuracy = cross_val_score(clf, X, Y, cv=kf)
f1 = cross_val_score(clf, X, Y, cv=kf, scoring ='f1_macro')
mean_f1 = np.mean(f1)

meanmissclassification = 1 - np.mean(accuracy)

I_acc = [np.mean(accuracy) - (1.96*accuracy.std()/np.sqrt(len(accuracy))), 
         np.mean(accuracy) + (1.96*accuracy.std()/np.sqrt(len(accuracy))) ]

I_f1 = [np.mean(f1) - (1.96*f1.std()/np.sqrt(len(f1))), 
         np.mean(f1) + (1.96*f1.std()/np.sqrt(len(f1))) ]

end = time.perf_counter()

print(f'DONE')
print(f'Time taken: ', round(end-start, 3))
print(f'Accuracy on training was: {round(np.mean(accuracy),3)} with 95% confidence interval: ({I_acc[0]:.3f},{I_acc[1]:.3f})')
print(f'The estimated E_new was: ', round(meanmissclassification, 3))
print(f'Cross-val f1 score was: {round(mean_f1, 3)} with with 95% confidence interval: ({I_f1[0]:.3f},{I_f1[1]:.3f})')



