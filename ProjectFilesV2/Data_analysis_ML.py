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

# Calculate total number of words spoken by each gender in the dataset
total_male_words = data['Number words male'].sum() + data.loc[data['Lead']=='Male', 'Number of words lead'].sum()
total_female_words = data['Number words female'].sum() + data.loc[data['Lead']=='Female', 'Number of words lead'].sum()

# Calculate percentage of words spoken by each gender
male_percentage = total_male_words / (total_male_words + total_female_words) * 100
female_percentage = total_female_words / (total_male_words + total_female_words) * 100

print(f'Male percentage: {male_percentage}')
print(f'Female percentage: {female_percentage}')

# Create bar plot

fig, ax = plt.subplots()
ax.bar(['Male', 'Female'], [male_percentage, female_percentage])
ax.set_ylabel('Percentage of Words Spoken')
ax.set_title('Percentage of Words Spoken by Gender')
plt.show()

# Question 2

# load the data into a pandas DataFrame
df = pd.read_csv('train.csv')

df['Number words male all'] = df.apply(lambda row: row['Number words male'] + row['Number of words lead'] if row['Lead'] == 'Male' else row['Number words male'], axis=1)
df['Number words female all'] = df.apply(lambda row: row['Number words female'] + row['Number of words lead'] if row['Lead'] == 'Female' else row['Number words female'], axis=1)


df_yearly = df.groupby('Year')[['Number words male all', 'Number words female all']].sum()

# calculate the total number of words spoken by all actors each year
df_yearly['Total words'] = df_yearly['Number words male all'] + df_yearly['Number words female all']

# Calculate the percentage of words spoken by male and female actors each year
df_yearly['Male %'] = df_yearly['Number words male all'] / df_yearly['Total words'] * 100
df_yearly['Female %'] = df_yearly['Number words female all'] / df_yearly['Total words'] * 100


plt.bar(df_yearly.index, df_yearly['Male %'], label='Male')
plt.bar(df_yearly.index, df_yearly['Female %'], bottom=df_yearly['Male %'], label='Female')

plt.title('Gender Balance in Speaking Roles Over Time')
plt.xlabel('Year')
plt.ylabel('Percentage of Total Words')
plt.legend()


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


