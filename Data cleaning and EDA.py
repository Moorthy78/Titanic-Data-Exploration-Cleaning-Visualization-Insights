#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "Untitled Folder 1/titanic.csv"  
df = pd.read_csv("titanic.csv")

print("🔹 First 5 Rows:\n", df.head())
print("\n🔹 Data Info:\n")
print(df.info())
print("\n🔹 Missing Values:\n", df.isnull().sum())


df['Age'].fillna(df['Age'].median(), inplace=True)


df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.drop(columns=['Cabin'], inplace=True)


plt.figure(figsize=(6, 4))
sns.barplot(x="Sex", y="Survived", data=df, palette="coolwarm")
plt.title("Survival Rate by Gender")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True, color='blue')
plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# 🔹 (C) Survival Rate by Passenger Class
plt.figure(figsize=(6, 4))
sns.barplot(x="Pclass", y="Survived", data=df, palette="coolwarm")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(x="Pclass", y="Fare", data=df, palette="coolwarm")
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.yscale("log")  
plt.show()


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1️⃣ Load the Titanic Dataset ---
file_path = "titanic.csv"  # Update if needed
df = pd.read_csv(file_path)

# --- 2️⃣ Basic Data Overview ---
print("🔹 First 5 Rows:\n", df.head())
print("\n🔹 Data Info:\n")
print(df.info())
print("\n🔹 Missing Values:\n", df.isnull().sum())

# --- 3️⃣ Handle Missing Values ---
df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill Age with median
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Fill Embarked with mode
df.drop(columns=['Cabin'], inplace=True)  # Drop Cabin (too many missing values)

# --- 4️⃣ Convert Categorical Data ---
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Convert 'Sex' to numeric

# --- 5️⃣ Exploratory Data Analysis (EDA) ---

# 🔹 (A) Survival Rate by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x="Sex", y="Survived", data=df, palette="coolwarm")
plt.xticks([0, 1], ['Male', 'Female'])
plt.title("Survival Rate by Gender")
plt.show()

# 🔹 (B) Age Distribution of Passengers
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True, color='blue')
plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# 🔹 (C) Survival Rate by Passenger Class
plt.figure(figsize=(6, 4))
sns.barplot(x="Pclass", y="Survived", data=df, palette="coolwarm")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

# 🔹 (D) Correlation Heatmap (Fixed)
plt.figure(figsize=(8, 6))

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 🔹 (E) Box Plot: Fare Distribution by Class
plt.figure(figsize=(8, 5))
sns.boxplot(x="Pclass", y="Fare", data=df, palette="coolwarm")
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.yscale("log")  # Log scale for better visualization
plt.show()


# In[ ]:




