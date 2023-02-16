import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

"""Has gender balance in speaking roles changed over time (i.e. years)?"""

# Genderyear = pd.read_csv('train.csv')

# Year = Genderyear['Year']
# Male = Genderyear['Number of male actors']
# Female = Genderyear['Number of female actors']

# plt.bar(Year, Male, label='Number of male actors')  

# plt.bar(Year, Female, label='Number of female actors')


# plt.title("Gender balance vs years")
# plt.xlabel("Years")
# plt.ylabel("Number of actors")
# plt.legend(loc='upper left')
# # Show the plot
# plt.show()





"""Här börjar logistic regression koden"""

# Load the data into a pandas DataFrame
df = pd.read_csv('train.csv')

# Define the number of folds for the k-fold cross validation
k = 10

# Split the data into k folds
kf = KFold(n_splits=k)

# Initialize a list to store the accuracy scores for each fold

all_ave_acc = []
max_acc_scores = []
numb_parameters = []

parameters = ['Number words female', 'Number of female actors', 'Age Lead', 'Difference in words lead and co-lead', 'Age Co-Lead', 'Number of male actors', 'Number words male', 'Mean Age Female', 'Mean Age Male', 'Number of words lead', 'Total words', 'Gross']
#parameters = ['Age Lead', 'Number words female', 'Age Co-Lead', 'Number of male actors', 'Total words', 'Difference in words lead and co-lead', 'Number of female actors']
ylabel = ['Lead']

for i in range(len(parameters), 0, -1):

    subsets = parameters[:i]
    numb_parameters.append(i)

    x = df[subsets].values
    y = df[ylabel].values.ravel()

    accuracy_scores = []
    # Loop over the folds
    for train_index, test_index in kf.split(df):
        # Split the data into training and testing sets
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train a logistic regression model
        logreg = LogisticRegression(penalty='l1', C=100, solver = "liblinear")
        logreg.fit(X_train, y_train)
        
        # Predict loan approval using the test data
        y_pred = logreg.predict(X_test)
        
        # Calculate the accuracy score for this fold
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores.append(accuracy)
    

    average_accuracy = np.mean(accuracy_scores)
    all_ave_acc.append(average_accuracy)


    max_acc_score = max(all_ave_acc)
    max_acc_scores.append(max_acc_score)

    print("Max accuracy: ", max_acc_scores)

# Plot results
plt.scatter(numb_parameters, max_acc_scores)
plt.title('Maximum accuracy = f(Number of parameters)')
plt.xlabel('Number of parameters')
plt.ylabel('Maximum Accuracy')
plt.xticks(numb_parameters)
plt.show()

