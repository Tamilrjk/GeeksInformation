# Employee Turnover Project

## Overview
This project analyzes employee turnover in a company to identify trends and patterns.By analyzing historical employee data, encompassing job satisfaction, salary, work environment, and performance metrics, the model aims to identify employees at risk of leaving the organization.

## What is employee turnover?
Employee turnover refers to the rate at which employees leave a company and are replaced by new employees. It is a critical metric for organizations, as it can impact productivity, morale, and overall business success. Employee turnover is usually expressed as a percentage and is calculated by dividing the number of employees who leave the company by the average number of employees during a specific period.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data OverView](#configuration)
- [Exploration Data Analysis](#examples)
- [Machine Learning Model](#troubleshooting)
- [Accurary](#Accurary)

  # Installation


  # Usage
This command will install the following libraries:
pandas

numpy

seaborn

matplotlib

scikit-learn

import the necessary libraries at the beginning of your code
```bash
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
```
# Data Overview
In your Python script or Jupyter Notebook, use the following code to read the employee turnover data from a CSV file into a Pandas DataFrame
```bash
# Read the CSV file into a DataFrame
df = pd.read_csv("Employee turnover prediction.csv")
```
Now, df is your DataFrame containing the data from the CSV file. You can proceed with data exploration, analysis, and any other tasks related to your project.

This code display the first few rows of the DataFrame
```bash
# Display the first few rows of the DataFrame
print(df.head())
# Display the last few rows of the dataFrame
print(df.tail())
```
## Dataset
1. **satisfaction_level:**
   - Represents the employee's satisfaction level, ranging from 0 to 1.
  
2. **last_evaluation:**
   - Represents the employee's performance evaluation score.

3. **number_project:**
   - Indicates the number of projects the employee has worked on.

4. **average_monthly_hours:**
   - Represents the average number of hours the employee works per month.

5. **time_spend_company:**
   - Indicates the number of years the employee has spent in the company.
  
6. **Work_accident:**
   - Binary variable indicating whether the employee has had a work accident (1 for Yes, 0 for No).
  
7. **left:**
   - Binary variable indicating whether the employee has left the company (1 for Yes, 0 for No).

8. **promotion_last_5years:**
   - Binary variable indicating whether the employee has been promoted in the last 5 years (1 for Yes, 0 for No).

9. **Department:**
   - Represents the department in which the employee works.

10. **salary:**
    - Represents the employee's salary level (low, medium, high).
   
  In the above example, each column is described along with an explanation of its role in the dataset
  
Now, you can describe the shape of your dataset using the df.shape attribute:
```bash
User
print(df.shape)
```
This will print the number of rows and columns in your dataset.

The dataset used in this project has 14,999 rows and 10 columns. Each row represents a data point, and each column represents a feature or attribute related to employee turnover prediction.

To get a quick overview of the numeric features in your dataset, you can use the describe() method in Pandas
```bash
# Display descriptive statistics
print(df.describe())
```
This will output a summary of the central tendency, dispersion, and shape of the distribution of your numeric columns.

you can use the df.info() method to get an overview of the DataFrame's structure.
```bash
# Display information about the DataFrame
df.info()
```
This includes information on the number of non-null values, data types, and memory usage.

use the following code to check for null values in your DataFrame
```bash
# Check for null values
null_values = df.isnull().any()
```
This code snippet will print the columns that contain null values in your DataFrame
# Exploration Data Analysis(EDA)

In your Python script or Jupyter Notebook, use the following code to create a set of subplots visualizing key employee metrics
```bash

# Set the style and grid for seaborn
sns.set(style="whitegrid")

# Create a figure with subplots
plt.figure(figsize=(12, 6))

# Subplot 1: Satisfaction Level Distribution
plt.subplot(2, 2, 1)
sns.histplot(df['satisfaction_level'], kde=True)
plt.title('Satisfaction Level Distribution')

# Subplot 2: Last Evaluation Distribution
plt.subplot(2, 2, 2)
sns.histplot(df['last_evaluation'], kde=True)
plt.title('Last Evaluation Distribution')

# Subplot 3: Average Monthly Hours Distribution
plt.subplot(2, 2, 3)
sns.histplot(df['average_montly_hours'], kde=True)
plt.title('Average Monthly Hours Distribution')

# Subplot 4: Time Spend in the Company Distribution
plt.subplot(2, 2, 4)
sns.histplot(df['time_spend_company'], kde=True)
plt.title('Time Spend in the Company Distribution')

# Display the subplots
plt.show()
```
This code generates a 2x2 grid of histograms using seaborn, visualizing the distribution of key employee metrics such as satisfaction level, last evaluation, average monthly hours, and time spent in the company.


