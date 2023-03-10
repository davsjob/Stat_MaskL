import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

#File computes and sorts the importance of different features on the dataset using RandomForestClassifier
start_time = time.perf_counter()
data = pd.read_csv('train.csv')
features = ['Number words female','Total words','Number of words lead','Difference in words lead and co-lead','Number of male actors','Year','Number of female actors','Number words male','Gross','Mean Age Male','Mean Age Female','Age Lead','Age Co-Lead']
ylabel = ['Lead']

#Takes values of the data and ignores label
X = data[features].values
y = np.where(data[ylabel].values == 'Female', 1, 0)


#Number of folds for kf cross validation
n_folds = 10
kf = KFold(n_splits=n_folds)

#Classifier choice, default value of 100 estimators
clf = RandomForestClassifier(n_estimators=100)

#Creates a list of importances
importances = []

#Does it for each fold
for train_i, test_i in kf.split(X):
    #Creates training and test
    X_train, X_test = X[train_i], X[test_i]
    y_train, y_test = y[train_i], y[test_i]

    #Fits the classification to training
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    #Appends importances values to importances
    importances.append(clf.feature_importances_)
    
#Calculates standard variation of importances
std = np.std(importances, axis = 0)

#Calculates mean value of the importances over all folds
avg_feature_importances = np.mean(importances, axis=0)

#Creates indexes by sorting largest to smallest
srt_indx = list(reversed(np.argsort(avg_feature_importances)))

#Assigns Feature name : Feature importance
val_to_feature = dict([(i,j) for i, j in zip(features, avg_feature_importances)])

#Assigns std to each feature
val_to_feature = dict(sorted(val_to_feature.items(), key=lambda item: item[1],reverse=True))

#Calculates time
elapsed_time = time.perf_counter() - start_time

#Printing statements
print(f'Time for computations: {elapsed_time}, seconds')
print()
print('In order of most to least important feature, with corresponding score: ')
print("\n".join("{}\t{}".format(k, v) for k, v in val_to_feature.items()))


# Plot the feature importances as a bar plot
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.figure(dpi=1000)
plt.title('Importance of individual features')
plt.bar(val_to_feature.keys(), val_to_feature.values(), yerr=std[srt_indx])
plt.xticks(range(13), range(1,14))
plt.ylabel('Avg feature importance')
plt.xlabel('Feature number')
plt.savefig('avgf_importance.png')
plt.show()

