
#Removes warning
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold , cross_val_score
import sklearn.metrics as skm

data = pd.read_csv('train.csv')

Ylabel = ['Lead'] 
#Ordered Features in order of importance
testfeat = ['Number words female','Number of female actors','Age Lead','Difference in words lead and co-lead','Age Co-Lead','Number of male actors','Number words male','Mean Age Female','Mean Age Male','Number of words lead','Total words','Gross','Year']

index = []

n_folds = 10
kf = KFold(n_splits=n_folds)


Gmean_acc = []
atimes = []
gtimes = []

test_range = range(13)
for i in test_range:
    X = data[testfeat].values
    y = data[Ylabel].values
    
    #Used GradientBoostingClassifier
    gclf = GradientBoostingClassifier(n_estimators=100)
    
    
    
    start = time.perf_counter()
    for train_index, test_index in kf.split(X):
        # Split the data into training and testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        
        gclf.fit(X_train, y_train)
        y_pred = gclf.predict(X_test)
        


        #ascores.append(Ascore)
        
        #atimes.append(aend-astart)
        
    # Compute the mean accuracy score across all folds
    #Amean_score = np.mean(ascores)
    end = time.perf_counter()
    gtimes.append(end-start)
    Gmean_score = np.mean(cross_val_score(gclf, X, y, cv=kf))


    #Amean_acc.append(Amean_score)
    Gmean_acc.append(Gmean_score)

    index.append(len(testfeat))
    testfeat.pop()
    



#print(f'Time for AdaBoost: ', np.mean(atimes))
print(f'Time for each GradientBooost: {round(np.mean(gtimes), 3)} seconds')
print(f'Total time', (round(np.sum(gtimes),3)))
print(f'Max Accuracy was : {round(max(Gmean_acc),3)}' )
#plt.figure(1)
#plt.scatter(test_range, Amean_acc)

plt.figure(1)
plt.scatter(test_range, Gmean_acc)
plt.ylabel('Accuracy')
plt.title('Accuracy = f(Number of features)')
plt.xlabel('Number of features')
plt.xticks(range(13),index, rotation=90)
plt.savefig('accgradient.png', dpi=1000)
plt.show()

