'''
Script that computes the maximum accuracy obtained when varying number of input features as well as neighbors 
for the k-NN method.
'''
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Load training data
data = pd.read_csv('train.csv')

# Define features and target
features = ['Number words female', 'Number of female actors', 'Age Lead', 'Difference in words lead and co-lead', 'Age Co-Lead', 'Number of male actors', 'Number words male', 'Mean Age Female', 'Mean Age Male', 'Number of words lead', 'Total words', 'Gross', 'Year'] # Feature importance order
label = 'Lead'

max_accuracies = []
num_features = []

# Vary number of features by always removing the least important feature
for i in range(len(features), 0, -1):
    subset = features[:i]
    num_features.append(i)

    # Extract feature and target arrays
    X = data[subset].values
    y = data[label].values

    # Normalize the features in matrix X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Vary number of neighbors
    accuracies = []
    for k in range(1, 51):
        # Create the k-NN classifier
        knn = KNeighborsClassifier(n_neighbors=k)

        # Conduct k-fold cross-validation
        kf = KFold(n_splits=10)
        scores = cross_val_score(knn, X, y, cv=kf)

        # Calculate average accuracy
        accuracy = np.mean(scores)
        accuracies.append(accuracy)

    # Append only the maximum accuracy obtained when varying number of neighbors 
    max_accuracy = max(accuracies)
    max_accuracies.append(max_accuracy)


# Plot results
plt.scatter(num_features, max_accuracies)
plt.title('Maximum accuracy = f(Number of features)')
plt.xlabel('Number of Features')
plt.ylabel('Maximum Accuracy')
plt.xticks(num_features)
plt.show()





