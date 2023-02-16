import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')

# 
col1 = df["Number words female"]
col2 = df["Number words male"]

df["more female words"] = col1 > col2
df["more male words"] = col2 > col1

df_female = df[df["more female words"]]
df_male = df[df["more male words"]]

# Plot the scatter plots
plt.scatter(df_male["Number words male"], df_male["Gross"], label='More Male Words', s=5)

plt.scatter(df_female["Number words female"], df_female["Gross"], label='More Female Words', s=5)

# Add a title and labels to the x and y axes
plt.title("Words by Gender")
plt.xlabel("Number of Words")
plt.ylabel("Gross")

# Add a legend
plt.legend(loc='upper left')

# Show the plot
plt.show()

