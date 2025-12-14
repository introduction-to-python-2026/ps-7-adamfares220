import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/day.csv')

df.sample(10)

df.describe()

df.dtypes

features = ["temp", "hum", "windspeed", "cnt"]

df[features].hist(bins=20, figsize=(10, 8))
plt.savefig("histograms.png")
plt.close()

df['hum'].hist(bins=20, figsize=(10, 8))
plt.savefig("hum.png")
plt.close()

df['windspeed'].hist(bins=20, figsize=(10, 8))
plt.savefig("windspeed.png")
plt.close()

df["cnt"].hist(bins=20, figsize=(10, 8))
plt.savefig("cnt.png")
plt.close()

plt.figure()
plt.scatter(df["hum"], df["cnt"], alpha=0.5)
plt.xlabel("humidity")
plt.ylabel("Bike Rentals Count")
plt.title("humidity vs Bike Rentals")
plt.savefig("scatter_hum_cnt.png")
plt.close()

plt.figure()
plt.scatter(df["windspeed"], df["cnt"], alpha=0.5)
plt.xlabel("windspeed")
plt.ylabel("Bike Rentals Count")
plt.title("windspeed vs Bike Rentals")
plt.savefig("scatter_windspeed_cnt.png")
plt.close()

plt.figure()
plt.scatter(df["windspeed"], df["cnt"], alpha=0.5)
plt.xlabel("windspeed")
plt.ylabel("Bike Rentals Count")
plt.title("windspeed vs Bike Rentals")
plt.savefig("scatter_windspeed_cnt.png")
plt.close()

plt.figure()
plt.scatter(df["temp"], df["cnt"], alpha=0.5)
plt.xlabel("Temperature")
plt.ylabel("Bike Rentals Count")
plt.title("Temperature vs Bike Rentals")
plt.savefig("scatter_temp_cnt.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.regplot( x="temp", y="cnt", data=df, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"} )
plt.title("Correlation between Temperature and Bike Rentals")
plt.tight_layout()
plt.savefig("correlation.png")
plt.close()

corr =df[features].corr()
print(corr)
