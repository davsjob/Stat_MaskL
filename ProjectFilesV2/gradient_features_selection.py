
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold , cross_val_score

data = pd.read_csv('train.csv')

Ylabel = ['Lead'] 
#Features in order of importance
testfeat = ['Number words female','Number of female actors','Age Lead','Difference in words lead and co-lead','Age Co-Lead','Number of male actors','Number words male','Mean Age Female','Mean Age Male','Number of words lead','Total words','Gross','Year']

index = []

#numbers of folds for k-fold cross val
n_folds = 10
kf = KFold(n_splits=n_folds)


mean_acc = []
times = []

test_range = range(13)
for i in test_range:
    X = data[testfeat].values
    # Sets binary where Female == 1 
    y = np.where(data[Ylabel].values == 'Female', 1, 0)
    
    #Used GradientBoostingClassifier with 100 estimators
    clf = GradientBoostingClassifier(n_estimators=100)
    
    
    
    start = time.perf_counter()
    for train_index, test_index in kf.split(X):
        # Split the data into training and testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # trains classifier on training data-fold
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    end = time.perf_counter()
    times.append(end-start)
    
    # appends cross evaluation score
    mean_score = np.mean(cross_val_score(clf, X, y, cv=kf))
    mean_acc.append(mean_score)

    #inserts the numbers of features for x label on the plot
    index.append(len(testfeat))

    #removes a feature
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

