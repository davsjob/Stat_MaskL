#Plotting for Data Analysis Tasks
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('train.csv')

#Plots number of words spoken by gender over the years
plt.figure(1, dpi=1000)
plt.bar(data["Year"], data["Number words male"], label='Male')
plt.bar(data["Year"], data["Number words female"], label='Female')
plt.legend(loc='upper left')
plt.title("Number of words spoken by gender over years")
plt.ylabel("Number of words")
plt.xlabel("Year")
plt.savefig("nr_words_gender_year.png")

#Plots number of actors per gender over the years
plt.figure(2, dpi=1000)
plt.bar(data['Year'], data['Number of male actors'], label='Number of male actors')  
plt.bar(data['Year'], data['Number of female actors'], label='Number of female actors')
plt.title("Gender balance vs years")
plt.xlabel("Years")
plt.ylabel("Number of actors")
plt.legend(loc='upper left')
plt.savefig('genderbalance_years.png')

plt.figure(3, dpi=1000)

col1 = data["Number words female"]
col2 = data["Number words male"]

data["more female words"] = col1 > col2
data["more male words"] = col2 > col1

df_female = data[data["more female words"]]
df_male = data[data["more male words"]]

# Plot the scatter plots
plt.scatter(df_male["Number words male"], df_male["Gross"], label='More Male Words', s=5)

plt.scatter(df_female["Number words female"], df_female["Gross"], label='More Female Words', s=5)

# Add a title and labels to the x and y axes
plt.title("Words by Gender")
plt.xlabel("Number of Words")
plt.ylabel("Gross")

# Add a legend
plt.legend(loc='upper left')
plt.savefig('words_by_gender_and_gross.png')