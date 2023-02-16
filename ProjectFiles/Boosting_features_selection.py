
#Removes warning
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold , cross_val_score

data = pd.read_csv('train.csv')

Ylabel = ['Lead'] 
#Features in order of importance
testfeat = ['Number words female','Number of female actors','Age Lead','Difference in words lead and co-lead','Age Co-Lead','Number of male actors','Number words male','Mean Age Female','Mean Age Male','Number of words lead','Total words','Gross','Year']

index = []

n_folds = 10
kf = KFold(n_splits=n_folds)


mean_acc = []
times = []

test_range = range(13)
for i in test_range:
    X = data[testfeat].values
    y = data[Ylabel].values
    
    #Used GradientBoostingClassifier this time
    #For other classifier, change it here
    clf = GradientBoostingClassifier(n_estimators=100)
    
    
    
    start = time.perf_counter()
    for train_index, test_index in kf.split(X):
        # Split the data into training and testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    end = time.perf_counter()
    times.append(end-start)
    mean_score = np.mean(cross_val_score(clf, X, y, cv=kf))
    mean_acc.append(mean_score)

    index.append(len(testfeat))
    testfeat.pop()
    



print(f'Time for each GradientBooost: {round(np.mean(times), 3)} seconds')
print(f'Total time', (round(np.sum(times),3)))
print(f'Max Accuracy was : {round(max(mean_acc),3)}' )

plt.figure(1)
plt.scatter(test_range, mean_acc)
plt.ylabel('Accuracy')
plt.title('Accuracy = f(Number of features)')
plt.xlabel('Number of features')
plt.xticks(range(13),index, rotation=90)
plt.savefig('accgradient.png', dpi=1000)
plt.show()

