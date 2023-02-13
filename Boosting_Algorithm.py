#Boosting code David Sjöberg

#Removes warning
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
import sklearn.metrics as skm
import random
seed = random.seed(5)
data = pd.read_csv('train.csv')

Features = ['Number words female','Total words','Number of words lead','Difference in words lead and co-lead','Number of male actors','Year','Number of female actors','Number words male','Gross','Mean Age Male','Mean Age Female','Age Lead','Age Co-Lead']
Randomsample = random.sample(Features, (random.randint(1,len(Features))))
Ylabel = ['Lead'] 
nEWFEAT = [ 'Difference in words lead and co-lead', 'Number words male', 'Age Co-Lead', 'Number words female', 'Number of male actors', 'Age Lead', 'Mean Age Male', 'Number of words lead', 'Mean Age Female', 'Total words', 'Number of female actors']
X = data[nEWFEAT].values
y = data[Ylabel].values

n_folds = 10

kf = KFold(n_splits=n_folds)

Amean_acc = []
Gmean_acc = []
atimes = []
gtimes = []

test_range = range(1,2)
for i in test_range:
    clf = AdaBoostClassifier(n_estimators=100)
    gclf = GradientBoostingClassifier(n_estimators=100)
    ascores = []
    gscores = []
    

    for train_index, test_index in kf.split(X):
        # Split the data into training and testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train the Gradient and AdaBoost classifier on the training set
        astart = time.perf_counter()
        
        clf.fit(X_train, y_train)

        # Predict the labels for the test set
        Ay_pred = clf.predict(X_test)
        

        # Compute the accuracy score for this fold
        Ascore = skm.accuracy_score(y_test, Ay_pred)
        aend = time.perf_counter()

        gstart = time.perf_counter()
        gclf.fit(X_train, y_train)
        Gy_pred = gclf.predict(X_test)
        Gscore = skm.accuracy_score(y_test, Gy_pred)
        gend = time.perf_counter()


        ascores.append(Ascore)
        gscores.append(Gscore)
        atimes.append(aend-astart)
        gtimes.append(gend-gstart)
    # Compute the mean accuracy score across all folds
    Amean_score = np.mean(ascores)
    Gmean_score = np.mean(gscores)

    Amean_acc.append(Amean_score)
    Gmean_acc.append(Gmean_score)



print(f'Max accuracy of AdaBoost was; ', max(Amean_acc))
print(f'Max accuracy of Gradient boosting was; ',max(Gmean_acc))

print(f'Time for AdaBoost: ', np.mean(atimes))
print(f'Time for GradientBooost: ', np.mean(gtimes))

"""plt.figure(1)
plt.scatter(test_range, Amean_acc)

plt.figure(2)
plt.scatter(test_range, Gmean_acc)

plt.show()"""