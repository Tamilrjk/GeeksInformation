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



[image](https://github.com/Tamilrjk/GeeksInformation/blob/main/Task3/Image/download.png) 









In your data analysis, it's essential to understand the relationships between different variables. One effective way to visualize these relationships is by creating a correlation matrix heatmap.
```bash
# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```


[image](https://github.com/Tamilrjk/GeeksInformation/blob/main/Task3/Image/download%20(1).png)





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

[image](https://github.com/Tamilrjk/GeeksInformation/blob/main/Task3/Image/download%20(2).png)




# Machine Learning Model

## Data splitting
The dataset is split into training and testing sets to facilitate model training and evaluation.
In the section, we explore the application of machine learning model to address our research question. we employed Logistic Regression and Random Forest due to their suitability for our binary classification problem.

Logistic Regression:

   Logistic regression is a linear model widely used for binary classification task. It calculates the probability of an belong to particular class and is particular useful when the relationship between the feature and target variable.

```bash
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
​
print(accuracy),
print(precision),
print(recall), 
print(f1)
```
Random Forest:

Random Forest is an ensemble learning method that combines the predictions of multiple decision trees. First random forest was instantiated after the model was trained using the training dataset (X_train and y_train)   and the model was used to make predictions on the test data.
```bash
# Train a Random Forest classifier on the balanced data
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
# Make predictions on the test set
y_pred_ra = rf_model.predict(X_test)
# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred_ra)
precision = precision_score(y_test, y_pred_ra)
recall = recall_score(y_test, y_pred_ra)
f1 = f1_score(y_test, y_pred_ra)

print(accuracy),
print(precision),
print(recall), 
print(f1)
```
# Evaluation
To assess the preformation of the Logistic regression and Random Forest model, we employed a range of evaluation metrics, including accuracy, Precision, recall and F1 score. 

             Model	Accuracy	  Precision	   Recall	    F1 score
Logistic Regression	97%      97%          96%        97%

Random Forest	99%          99%            100%         99%

## Roc Curve

[image](https://github.com/Tamilrjk/GeeksInformation/blob/main/Task3/Image/download%20(3).png)

The ROC curve shows the performance of a binary classifier at different threshold values. The true positive rate (TPR) is the proportion of positive cases that are correctly identified by the classifier, and the false positive rate (FPR) is the proportion of negative cases that are incorrectly identified as positive by the classifier. The ROC curve is plotted with the TPR on the y-axis and the FPR on the x-axis. A perfect classifier would have a TPR of 1 and an FPR of 0, meaning that it would correctly identify all positive cases and none of the negative cases.


# License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

