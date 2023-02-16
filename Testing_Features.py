#Boosting algorithm testfile for finding optimal parameters
#Remove warings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
import sklearn.metrics as skm
import random

def boostingparameters(parameters,ylabel, algorithm,estimators, indata, i):
    start_time = time.perf_counter()
    data = pd.read_csv(indata)
    features = parameters
    randomvars = random.sample(features, (random.randint(1,len(features))))

    X = data[randomvars].values
    y = data[ylabel].values

    n_folds = 10
    kf = KFold(n_splits=n_folds)

    clf = algorithm(n_estimators=estimators)

    accuracies = []
    for train_i, test_i in kf.split(X):
        X_train, X_test = X[train_i], X[test_i]
        y_train, y_test = y[train_i], y[test_i]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        ascore = skm.accuracy_score(y_test, y_pred)
        accuracies.append(ascore)
    amean = np.mean(accuracies)
    stop_time = time.perf_counter()
    print(f'Time for iteration {i}: ', stop_time-start_time)
    return [amean,randomvars]



data = pd.read_csv('train.csv')
Features = ['Number words female','Total words','Number of words lead','Difference in words lead and co-lead','Number of male actors','Year','Number of female actors','Number words male','Gross','Mean Age Male','Mean Age Female','Age Lead','Age Co-Lead']
Ylabel = ['Lead']
parameters = []
def firstn(list, i):
    for n in range(i):
        print(list[n])
for i in range(15):
    ret = boostingparameters(Features, Ylabel, GradientBoostingClassifier, 100, 'train.csv', i)
    parameters.append(ret)
print('In order of highest to lowest accuracy for a random combination of parameters; ')
firstn(sorted(parameters,reverse=True),5)

print(f'With a max accuracy of: ', max(parameters))
