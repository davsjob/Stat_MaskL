'''
Data Analysis Task

'''

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Question 1

'''
Script that creates a new column indicating the gender of the co-lead based on the gender of the lead, groups the 
data by lead and co-lead gender and calculates the sum of the number of words spoken by each gender, calculates 
the total number of words spoken by each gender, and finally creates a bar chart showing the percentage 
of words spoken by each gender.

'''

# Load data 
data = pd.read_csv('train.csv')

# Create a new column indicating the gender of the co-lead based on the gender of the lead
data.loc[data['Lead'] == 'Female', 'Co-Lead Gender'] = 'Male'
data.loc[data['Lead'] == 'Male', 'Co-Lead Gender'] = 'Female'

# Group the data by lead and co-lead gender and calculate the sum of the number of words spoken by each gender
gender_words = data.groupby(['Lead', 'Co-Lead Gender'])[['Number words female', 'Number words male']].sum()

# Calculate the total number of words spoken by each gender
total_words = gender_words.sum()

# Calculate the percentage of words spoken by each gender
percent_female = 100 * total_words['Number words female'] / sum(total_words)
percent_male = 100 * total_words['Number words male'] / sum(total_words)

# Create a bar chart showing the percentage of words spoken by each gender
plt.bar(['Female', 'Male'], [percent_female, percent_male])
plt.title('Percentage of words spoken by gender')
plt.ylabel('Percentage')
plt.show()

# Question 2

# load the data into a pandas DataFrame
df = pd.read_csv('train.csv')

# create separate columns for male and female words spoken by all actors
df['Number words male all'] = df.apply(lambda row: row['Number words male'] + row['Number of words lead'] if row['Lead'] == 'Male' else row['Number words male'], axis=1)
df['Number words female all'] = df.apply(lambda row: row['Number words female'] + row['Number of words lead'] if row['Lead'] == 'Female' else row['Number words female'], axis=1)

# group the data by year and calculate the total number of words spoken by male and female actors
df_yearly = df.groupby('Year')[['Number words male all', 'Number words female all']].sum()

# calculate the total number of words spoken by all actors each year
df_yearly['Total words'] = df_yearly['Number words male all'] + df_yearly['Number words female all']

# calculate the percentage of words spoken by male and female actors each year
df_yearly['Male %'] = df_yearly['Number words male all'] / df_yearly['Total words'] * 100
df_yearly['Female %'] = df_yearly['Number words female all'] / df_yearly['Total words'] * 100

# create a bar chart
plt.bar(df_yearly.index, df_yearly['Male %'], label='Male')
plt.bar(df_yearly.index, df_yearly['Female %'], bottom=df_yearly['Male %'], label='Female')

# set the chart title, axes labels, and legend
plt.title('Gender Balance in Speaking Roles Over Time')
plt.xlabel('Year')
plt.ylabel('Percentage of Total Words')
plt.legend()

# show the chart
plt.show()

# Question 3

'''
Script that computes the average gross for films with female and male lead respectively

'''

# Load data
data = pd.read_csv('train.csv')

# Compute average gross for female and male leads
female_gross_avg = np.mean(data.loc[data['Lead']=='Female', 'Gross'])
male_gross_avg = np.mean(data.loc[data['Lead']=='Male', 'Gross'])

# Print results
print(f"Average gross for films with female lead: {female_gross_avg:.2f}")
print(f"Average gross for films with male lead: {male_gross_avg:.2f}")


