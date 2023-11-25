# Project Title
 ## Fraud Detection for Credit Cards
 
# Project Description
In this project, we aimed to develop an effective fraud detection system for credit card transactions. Two machine learning models, Logistic Regression and Random Forest, were employed to achieve this goal.

# Table of Contents
Features

Getting Started

Installation


Data Overview 

Exploratory Data Analysis

Model

Evaluation

Contributing

License

# Features
Real-time transaction monitoring
Anomaly detection algorithms
Interactive visualization of fraudulent activities

# Getting Started
Provide instructions on how to get started with your project. Include information on system requirements, dependencies, and any prerequisites. For example:

Prerequisites

Python 3.6 or higher

Jupyter Notebook

Required Python packages (list them, e.g., pandas, scikit-learn)

# Installation
```bash
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```
It has all the necessary libraries 

## Data Overview 

The dataset consists of 250,000 instances and 31 features. The features include numerical variables.

Time:

      Data Type: Numerical
      Description: The time elapsed since the first transaction in seconds.

  V1-V28:
  
      Data Type: Numerical
      Description:  Anonymous feature resulting from a PCA transaction. They are numerical variable derived from the original data to protect user identities.
      
   Amount:
   
      Data Type: Numerical
      Description: The transaction amounts.

Class:

     Data Type: Numerical
     Description: The target variables indicating whether the transaction is fraud (1) or not (0).
  

 ## Checking for Null Values in the DataFrame
```bash
df.isnull().sum()
```
The dataset is free of missing values, providing a clean and complete foundation for analysis. 

# Exploratory Data Analysis

To understand the distribution of fraudulent and non-fraudulent transactions in the dataset, a bar plot is created using Python's Seaborn library. The code snippet for visualization is as follows

```bash
# Visualize the class distribution (fraudulent vs. non-fraudulent transactions)
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.show()
```
In your data analysis, it's essential to understand the relationships between different variables. One effective way to visualize these relationships is by creating a correlation matrix heatmap.
```bash
# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```
It selects numerical features from the DataFrame, excluding the 'Class' column.
For each numerical feature, a box plot is created to visualize the distribution of that feature with respect to the 'Class' variable.
```bash
# Box plots for numerical features by class
num_features = df.columns[:-1]  # Exclude the 'Class' column
for feature in num_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Class', y=feature, data=df)
    plt.title(f'Box Plot of {feature} by Class')
    plt.show()
```
Visualizing Transaction Amounts for Fraud Detection
This code snippet utilizes the seaborn library to visualize the distribution of transaction amounts for both fraudulent and non-fraudulent transactions in a credit card fraud detection project. The goal is to provide insights into the patterns of transaction amounts for different classes.

```bash
 #Set the aesthetic style of the plots
sns.set_style('whitegrid')

#Plot the distribution of transaction amounts for fraudulent and non-fraudulent transactions
plt.figure(figsize=(10,6))
sns.histplot(df[df['Class'] == 0]['Amount'], bins=100, kde=True, color='blue', label='Non-Fraudulent')
sns.histplot(df[df['Class'] == 1]['Amount'], bins=100, kde=True, color='red', label='Fraudulent')
plt.title('Distribution of Transaction Amounts for Fraudulent and Non-Fraudulent Transactions')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.xscale('log')
plt.legend()
plt.show()
```






