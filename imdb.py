import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(sns.get_dataset_names())

df=sns.load_dataset("IMDB.csv")
print(df.head(10))

print(df.shape)

print(df.tail())

print(df.isnull().sum())

print(df.describe())

print(df.dtypes)

print(df.info())

print(df.describe(include="all"))

print(df.corr(numeric_only=True))

sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()

df.select_dtypes(include=[np.number].hist(figsize=(15,10)))
plt.show()

df.select_dtypes(include=[np.number]).plot(kind="box",subplots=True,layout=(2,2),figsize=(15,10))
plt.show()

print(df.Certificate.value_counts())

print(df.Director.value_counts())

print(df.Genre.value_counts())

sns.count_plot(data=df,x="Overview")
plt.show()

sns.count_plot(data=df,x="Star1")
plt.show()

sns.count_plot(data=df,x="Genre",hue="Star2")
plt.show()

sns.count_plot(data=df,x="Released_Year",hue="Series_Title")
plt.show()

sns.count_plot(data=df,x="Runtime",hue="Poster_Link")
plt.show()

sns.pairplot(data=df,hue="IMDB_Rating")
plt.show()