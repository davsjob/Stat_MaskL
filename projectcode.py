#Projectfile David S
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cols = ["Number words female", "Total words", "Year"]
csv = pd.read_csv('train.csv', usecols=cols)
csv["Number words male"] = csv["Total words"] - csv["Number words female"]

plt.figure(dpi=1200)
plt.bar(csv["Year"], csv["Number words male"], label='Male')
plt.bar(csv["Year"], csv["Number words female"], label='Female')


plt.legend()
plt.title("Number of words spoken by gender over years")
plt.ylabel("Number of words")
plt.xlabel("Year")
plt.savefig("nr_words_gender_year.png")

