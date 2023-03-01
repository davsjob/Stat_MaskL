import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("train.csv")

# Split the data into features (X) and target (y)
testfeat = ['Number words female','Number of female actors','Age Lead','Difference in words lead and co-lead','Age Co-Lead','Number of male actors','Number words male','Mean Age Female','Mean Age Male','Number of words lead','Total words','Gross','Year']
testfeat1 = testfeat[:10]
new_test = data[testfeat1].values
y = (data["Lead"] == "Female").astype(int) # 1 for Female and 0 for Male

# Initialize the k-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Train a random forest classifier using k-fold cross-validation
clf = RandomForestClassifier(n_estimators=200) #uses 200 trees
scores = cross_val_score(clf, new_test, y, cv=kfold)

# Print the average accuracy and F1 score
accuracy = scores.mean()
std = scores.std()
ci_acc = 1.96 * (std / (len(scores) ** 0.5))

f1_scores = cross_val_score(clf, new_test, y, cv=kfold, scoring='f1_macro')
f1_mean = f1_scores.mean()
f1_std = f1_scores.std()
ci_f1 = 1.96 * (f1_std / (len(f1_scores) ** 0.5))

print("Accuracy: {:.2f}% ({:.2f}-{:.2f})".format(accuracy * 100, (accuracy - ci_acc) * 100, (accuracy + ci_acc) * 100))
print("F1 score: {:.2f} ({:.2f}-{:.2f})".format(f1_mean, (f1_mean - ci_f1), (f1_mean + ci_f1)))



