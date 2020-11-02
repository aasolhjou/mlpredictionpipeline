#import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import _pickle as pickle

#read in data
feature_df = pd.read_csv('datasets/main_df.csv', index_col = 0)

#price vs. location barplot
plt.figure(figsize=(10,8))
sns.barplot(x = "location", y = "price", data = feature_df)
plt.savefig('visualizations/price_by_location.png')
plt.close()

#price vs. year barplot
plt.figure(figsize=(10,8))
sns.barplot(x = "year", y = "price", data = feature_df)
plt.savefig('visualizations/price_by_year.png')
plt.close()

#price vs. km scatterplot
plt.figure(figsize=(10,8))
sns.scatterplot(x = "km", y = "price", data = feature_df, hue = 'kmpl')
plt.savefig('visualizations/price_by_km.png')
plt.close()

#price vs. fuel boxplot
plt.figure(figsize=(10,8))
sns.boxplot(x = "fuel", y = "price", data = feature_df)
plt.savefig('visualizations/price_by_fuel.png')
plt.close()

#price vs. gear boxplot
plt.figure(figsize=(10,8))
sns.boxplot(x = "gears", y = "price", data = feature_df)
plt.savefig('visualizations/price_by_gears.png')
plt.close()

#price vs. owners barplot
plt.figure(figsize=(10,8))
sns.barplot(x = "owners", y = "price", data = feature_df)
plt.savefig('visualizations/price_by_owners.png')
plt.close()

#price vs. kmpl scatterplot
plt.figure(figsize=(10,8))
sns.scatterplot(x = "kmpl", y = "price", data = feature_df, hue = 'km')
plt.savefig('visualizations/price_by_kmpl.png')
plt.close()

#price vs. engine_size regplot
plt.figure(figsize=(10,8))
sns.regplot(x = "engine_size", y = "price", data = feature_df)
plt.savefig('visualizations/price_by_engine_size.png')
plt.close()

#price vs. power regplot
plt.figure(figsize=(10,8))
sns.regplot(x = "power", y = "price", data = feature_df)
plt.savefig('visualizations/price_by_power.png')
plt.close()

#price vs. seats barplot
plt.figure(figsize=(10,8))
sns.barplot(x = "seats", y = "price", data = feature_df)
plt.savefig('visualizations/price_by_seats.png')
plt.close()

#price distribution plot
x=feature_df.price
plt.figure(figsize=(10,8))
sns.distplot(x)
plt.savefig('visualizations/price_distribution_plot.png')
plt.close()

print("Top 5 Features:")

#correlation matrix
corr = feature_df.corr().iloc[[-1],:-1]

plt.figure(figsize=(8,1))
sns.heatmap(corr, annot=False, linewidths=.1, cmap="coolwarm")
plt.xticks()
plt.yticks(rotation=0)
plt.savefig('visualizations/top_features_heatmap.png')
plt.close()

corr = feature_df.corr().iloc[[-1],:-1]

top5_features = corr.transpose().apply(abs).sort_values(by='price', ascending=False)[:5]

top5_features_df = feature_df[top5_features.index].join(feature_df.price)

#top 5 features
top5_features.plot(kind='bar')

print(top5_features)
