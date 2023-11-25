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




