'''
Script that plots how the average missclassification error varies with k, the number of neighbors. The best k-NN
classifier later computes the the average missclassification error, the accuracy and the F1-score. 
'''

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load training data
data = pd.read_csv('train.csv')

# Define features and label
features = ['Number words female', 'Number of female actors', 'Age Lead', 'Difference in words lead and co-lead', 'Age Co-Lead', 'Number of male actors', 'Number words male', 'Mean Age Female', 'Mean Age Male', 'Number of words lead', 'Total words', 'Gross', 'Year'] # Feature importance order
features = features[:5] # Only to get the 9 most important features
label = 'Lead'

# Extract feature and target 
X = data[features].values
y = (data[label] == 'Female').astype(int)  # Convert label to binary array, where 'Female' is positive (1) and 'Male' is negative (0)


# Normalize the features in matrix X
scaler = StandardScaler()
X = scaler.fit_transform(X) # Results in normalized version of X where mean of each feature is 0 and the std is 1

# Vary number of neighbors
average_errors = []
accuracies = []
for i in range(1, 51): 

    # Create the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=i)

    # Calculate average misclassification error and accuracy using 10-fold cross-validation
    scores = cross_val_score(knn, X, y, cv=10)
    average_error = 1 - np.mean(scores)
    average_errors.append(average_error)
    accuracy = np.mean(scores)
    accuracies.append(accuracy)
    f1 = np.mean(cross_val_score(knn, X, y, cv=10, scoring='f1_macro'))

    print(f"k = {i}, Average misclassification error: {average_error:.3f}, Accuracy: {accuracy:.3f}, F1-score: {f1:.3f}")

# Plot results
plt.scatter([i for i in range(1, 51)], average_errors)
plt.title('Average E_new = f(k)')
plt.xlabel('k')
plt.ylabel('Average E_new')
plt.xticks([i for i in range(1, 51)])
plt.show()

# Create k-NN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=17)

# Calculate average misclassification error, accuracy and f1-score using 10-fold cross-validation
scores = cross_val_score(knn, X, y, cv=10)
average_error = 1 - np.mean(scores)
accuracy = np.mean(scores)
f1 = cross_val_score(knn, X, y, cv=10, scoring='f1_macro')
f1_mean = np.mean(f1)

# Calculate 95% confidence intervals for accuracy and F1-score
l = 1.96
I_accuracy = [accuracy - (l *(scores.std()/np.sqrt(len(scores)))), accuracy + (l*(scores.std()/np.sqrt(len(scores))))]
I_f1 = [f1_mean - (l*(f1.std()/np.sqrt(len(f1)))), f1_mean + (l*(f1.std()/np.sqrt(len(f1))))]

# Results 
print(f"k = 5, Average misclassification error: {average_error:.3f}, Accuracy: {accuracy:.3f}, 95% CI for Accuracy: [{I_accuracy[0]:.3f}, {I_accuracy[1]:.3f}], F1-score: {f1_mean:.3f}, 95% CI for F1-score: [{I_f1[0]:.3f}, {I_f1[1]:.3f}]")

