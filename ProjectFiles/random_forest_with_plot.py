import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("train.csv")

# Split the data into features (X) and target (y)
testfeat = ['Number words female','Number of female actors','Age Lead','Difference in words lead and co-lead','Age Co-Lead','Number of male actors','Number words male','Mean Age Female','Mean Age Male','Number of words lead','Total words','Gross','Year']
testfeat1 = testfeat[:10]
new_test = data[testfeat1].values
y = data["Lead"] #what we want to get out of the dataset

# Initialize the k-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Define a range of values for n_estimators with a 20 increment
num_trees = range(1, 501, 20)

# Train random forest classifiers with different numbers of trees
accuracies = []
for n in num_trees:
    clf = RandomForestClassifier(n_estimators=n)
    scores = cross_val_score(clf, new_test, y, cv=kfold)
    accuracy = scores.mean()
    accuracies.append(accuracy)


plt.scatter(num_trees, accuracies)
plt.title("Random Forest Accuracy vs. Number of Trees")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.show()
