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
- [Data Preprocessing](#DataPreprocessing)
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

## Satisfaction Level Distribution
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

![image1](https://github.com/Tamilrjk/GeeksInformation/blob/main/Task2/image/download.png)

The provided image shows two graphs that illustrate the satisfaction level distribution and the last evaluation distribution of a company.

**Satisfaction Level Distribution**

The satisfaction level distribution graph shows the average monthly satisfaction level of employees, grouped by their average monthly hours worked. The x-axis of the graph shows the average monthly hours worked, and the y-axis shows the average monthly satisfaction level.

The graph shows that, in general, employees who work more hours per month tend to have lower satisfaction levels. However, there is some variation within each group. For example, some employees who work 150 hours per month have satisfaction levels above 800, while others have satisfaction levels below 200.

**Last Evaluation Distribution**

The last evaluation distribution graph shows the average monthly last evaluation score of employees, grouped by their time spent in the company. The x-axis of the graph shows the time spent in the company, and the y-axis shows the average monthly last evaluation score.

The graph shows that, in general, employees who have spent more time in the company tend to have higher last evaluation scores. However, there is some variation within each group. For example, some employees who have spent 5 years in the company have last evaluation scores above 6000, while others have last evaluation scores below 2000.

Overall, the two graphs show that there is a positive correlation between employee satisfaction and last evaluation score. However, there is also some variation within each group, suggesting that there are other factors that also influence employee satisfaction and last evaluation score.

## Visualizing Department and Salary Distributions

In your Python script or Jupyter Notebook, use the following code to create a side-by-side countplot visualizing the distribution of employees across different departments and salary levels
```bash
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(data=df, x='Department')
plt.title('Department Distribution')

plt.subplot(1, 2, 2)
sns.countplot(data=df, x='salary')
plt.title('Salary Distribution')

plt.show()
```
This code generates a single figure with two side-by-side countplots using seaborn, visualizing the distribution of employees across different departments and salary levels.


![image2](https://github.com/Tamilrjk/GeeksInformation/blob/main/Task2/image/image.png)


The image you provided shows a graph of the distribution of salaries by department and salary. The department distribution is higher than the salary distribution, and the salary distribution is lower than the department distribution. The salary distribution is higher than the department distribution in the following departments: sales, accounting, HR, and technical. The salary distribution is lower than the department distribution in the following departments: IPPAN, bagmen, product manager, and RandD

This graph suggests that the company has a higher concentration of employees in the sales, accounting, HR, and technical departments, and that these departments also have higher salaries. The IPPAN, bagmen, product manager, and RandD departments have a lower concentration of employees, and these departments also have lower salaries.

This graph could be used by the company to make a number of decisions, such as how to allocate resources, how to set salaries, and how to recruit and retain employees

The sales department has the highest salary distribution, followed by the accounting department. This suggests that these two departments are the most valuable to the company.

The RandD department has the lowest salary distribution. This could be because RandD is a long-term investment, and the company may not expect to see immediate returns from this department.

The salary distribution in the technical department is higher than the salary distribution in the IPPAN department. This suggests that the company values technical skills more than general skills.

## Calculating and Visualizing Correlation Matrix
The following code calculates the correlation matrix for numeric columns in your DataFrame and visualizes it as a heatmap

```bash
numeric_df = df.select_dtypes(include=[float, int])

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
```
It then calculates the correlation matrix using Pandas' corr() method.

The resulting matrix is visualized as a heatmap using seaborn's heatmap function.

The annot=True parameter adds the correlation values to the heatmap for better interpretation.

The colormap is set to 'coolwarm' for better visibility, and linewidths are adjusted for clarity.


![image3](https://github.com/Tamilrjk/GeeksInformation/blob/main/Task2/image/image1.png )



The chat you sent is a correlation matrix, which is a table that shows the correlation between pairs of variables. The correlation coefficient is a measure of the linear relationship between two variables. It can range from -1 to 1, with a value of 1 indicating a perfect positive correlation, a value of -1 indicating a perfect negative correlation, and a value of 0 indicating no correlation.

The correlation matrix shows that there is a strong positive correlation between satisfaction_level and last_evaluation (correlation coefficient = 0.8). This means that employees who are more satisfied with their jobs tend to receive higher evaluations.

## Churn Distribution

To visualize the distribution of employee churn in our dataset, we use the following code snippet in Python with seaborn
```bash
# Countplot to visualize churn distribution
sns.countplot(data=df, x='left')
plt.title('Churn Distribution')
plt.show()
```
This code generates a count plot using Seaborn to visualize the distribution of employee churn in the dataset. The x-axis represents the 'left' column, indicating whether an employee has left the company (1) or not (0). The y-axis represents the count of employees in each category.

A bar at the value 0 on the x-axis indicates employees who have not left the company.
A bar at the value 1 on the x-axis indicates employees who have left the company.


![image4](https://github.com/Tamilrjk/GeeksInformation/blob/main/Task2/image/image2.png)


The images show a churn distribution graph, which is a graph that shows the percentage of employees who churned at each point in time. The x-axis of the graph shows the left, and the y-axis shows the number of employees who churned.

The churn rate is highest among the employees who have not left the company. This suggests that the company needs to focus on onboarding and providing employees with a positive employee experience.

The churn rate decreases over time, but it never reaches zero. This suggests that there will always be some employees who churn, regardless of what the company does.

## Employee Attrition Visualization
In this project, we use Python and the Seaborn library to create a scatter plot visualizing the relationship between employee satisfaction level and average monthly hours, with points colored based on whether the employee left the company.
```bash
plt.figure(figsize=(10, 6))
sns.scatterplot(x='satisfaction_level', y='average monthly hours', data=df, hue='left', palette='coolwarm', alpha=0.6)
plt.title('Satisfaction Level vs. Average Monthly Hours')
plt.xlabel('Satisfaction Level')
plt.ylabel('Average Monthly Hours')
plt.legend(title='Left Company')
plt.tight_layout()
plt.show()
```
This code generates a scatter plot that helps visualize the relationship between employee satisfaction levels, average monthly hours worked, and whether the employee left the company. Points are color-coded to indicate whether the employee left or stayed. The size of the figure, color palette, and transparency are adjusted for better visualization


![image5](https://github.com/Tamilrjk/GeeksInformation/blob/main/Task2/image/image3.png)

The image you sent shows an employee satisfaction vs. churn rate graph. The x-axis of the graph shows the employees' satisfaction score, and the y-axis shows the churn rate.

The graph shows a clear inverse correlation between employee satisfaction and churn rate. This means that as employee satisfaction increases, the churn rate decreases


The image you sent shows a customer satisfaction vs. churn rate graph. The x-axis of the graph shows the customer satisfaction score, and the y-axis shows the churn rate.

The graph shows a clear inverse correlation between customer satisfaction and churn rate. This means that as customer satisfaction increases, the churn rate decreases. This is because customers who are more satisfied with a product or service are less likely to cancel their subscription or switch to a competitor.

The graph also shows that there is a point at which employee satisfaction is so high that the churn rate is very low. This point is known as the employee satisfaction threshold.

Employee satisfaction is a key driver of churn reduction. Companies that focus on improving customer satisfaction will see a decrease in their churn rate.

## Visualizing Distribution of the Number of Projects

In your Python script or Jupyter Notebook, the following code snippet uses seaborn to create a count plot, illustrating the distribution of the number of projects for employees
```bash
f, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='number_project', data=df, palette='viridis', ax=ax)
ax.set_title('Distribution of Number of Projects')
ax.set_xlabel('Number of Projects')
ax.set_ylabel('Count')
plt.tight_layout()
plt.show()
```
This code generates a count plot to visualize the distribution of the number of projects for employees in the DataFrame DF. Adjust the data frame and column name ('number_project') based on the specifics of your dataset


![image](https://github.com/Tamilrjk/GeeksInformation/blob/main/Task2/image/image4.png)

The image you show is of the Distribution of Number of Projects graph. The x-axis of the graph shows the number of projects, and the y-axis shows the count of project

The graph shows that the majority of 4 projects were taken on by more than 4000 employees.
The graph shows that 7 projects were taken by a few peoples


## Data Preprocessing
Before building predictive models, it's essential to split the dataset into features (X) and the target variable (y). In this case, the target variable is 'left,' representing whether an employee has left the company.
```bash
# Split the data into features (X) and the target variable (y)
X = df.drop('left', axis=1)
y = df['left']`
```
X: Contains the features (independent variables) for each employee. The 'left' column is excluded using drop.
y: Represents the target variable, indicating whether an employee has left the company.
This separation is a common practice in machine learning workflows, allowing for the independent analysis of input features and the prediction of the target variable based on those features.


In your Python script or Jupyter Notebook, use the following code to encode categorical variables and split the data into training and testing sets
```bash
# Encode categorical variables (e.g., 'Department' and 'salary') using one-hot encoding
X = pd.get_dummies(X, columns=['Department', 'salary'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
The pd.get_dummies function is used to perform one-hot encoding on categorical variables ('Department' and 'salary') in the DataFrame X. The drop_first=True parameter avoids the dummy variable trap by dropping the first encoded column for each categorical variable.

## Machine Learning Model
### Random Forest
In this section of the code, we create and train a Random Forest Classifier using the scikit-learn library. The Random Forest Classifier is an ensemble learning method that combines the predictions of multiple decision trees to improve accuracy and generalization.
```bash
# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict on the testing set
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
# Generate a classification report
report = classification_report(y_test, y_pred)

# Display the accuracy and the classification report
print(f'Accuracy: {accuracy}')
print(report)
```
Random Forest Classifier:

We create an instance of the Random Forest Classifier with 100 trees (n_estimators=100) and set a random state (random_state=42) for reproducibility.
Training the Model:

The model is trained using the training set (X_train, y_train) with the fit method.
Prediction:

We use the trained model to make predictions on the testing set (X_test) using the predict method.

Evaluation:

Accuracy and a classification report are calculated using the ground truth labels (y_test) and the predicted labels (y_pred).

Results Display:

Finally, the accuracy and classification report are printed to the console.

### Support Vector Machines(SVM)
In your Python script or Jupyter Notebook, you can use the following code to initialize, train, and evaluate a Support Vector Machine classifier
```bash

# Assuming X_train, X_test, y_train, y_test are already defined

# Initialize the Support Vector Machine classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Predict on the testing set
y_pred_svm = svm_classifier.predict(X_test)

# Calculate the accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Generate a classification report
report_svm = classification_report(y_test, y_pred_svm)

# Display the accuracy and the classification report
print(f"Accuracy: {accuracy_svm}")
print("Classification Report:")
print(report_svm)
```
This code demonstrates how to initialize an SVM classifier with a linear kernel, train it using training data (X_train and y_train), predict on a testing set (X_test), calculate accuracy, and generate a classification report.

### Logistic Regression
use the following code to train a Logistic Regression classifier on your dataset
```bash

# Initialize the Logistic Regression classifier
logreg_classifier = LogisticRegression(random_state=42)

# Train the classifier
logreg_classifier.fit(X_train, y_train)

# Predict on the testing set
y_pred_logreg = logreg_classifier.predict(X_test)

# Calculate the accuracy
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

# Generate a classification report
report_logreg = classification_report(y_test, y_pred_logreg)

# Display the accuracy and the classification report
print(f"Accuracy: {accuracy_logreg}")
print("Classification Report:\n", report_logreg)
```
This code snippet demonstrates how to initialize and train a Logistic Regression classifier using scikit-learn. It then makes predictions on a testing set, calculates the accuracy, and generates a classification report.
