# Week7Assignment
import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset from sklearn
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]  # Add target column

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Check data types and missing values
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# Drop or fill missing values (if any)
df = df.dropna()  # If missing values exist
print("\nBasic Statistics:")
print(df.describe())

print("\nMean Sepal Length by Species:")
print(df.groupby('species')['sepal length (cm)'].mean())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
df['sepal length (cm)'].plot(kind='line', color='blue')
plt.title("Sepal Length Trend (Simulated)")
plt.xlabel("Index (Sample)")
plt.ylabel("Sepal Length (cm)")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='sepal length (cm)', data=df, palette='viridis')
plt.title("Average Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Mean Sepal Length (cm)")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df['petal length (cm)'], bins=20, kde=True, color='green')
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

try:
    df = pd.read_csv("nonexistent_file.csv")
except FileNotFoundError:
    print("Error: File not found. Using Iris dataset instead.")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)












