# Employee Turnover Analysis 
A Data Science Project on #GeeksInformation 

An analysis of employee turnover in a company, using Python and data analysis techniques and Machine Learning.

## Table of Contents
 - Data Description
 - Installation
 - Import Libraries and Dependencies Used
 - Visualize Miss Value
 - Data manipulation
 - Data Visualization
 -  Univariate Analysis
 - Data Preprocessing
 - Split the Dataset
 - ML Model
 - License

 ## Data Description
 What is Employee Turnover?
 
Employee Turnover or Employee Turnover ratio is the measurement of the total number of employees who leave an organization in a particular year. Employee Turnover Prediction means to predict whether an employee is going to leave the organization in the coming period.
A Company uses this predictive analysis to measure how many employees they will need if the potential employees will leave their organization. A company also uses this predictive analysis to make the workplace better for employees by understanding the core reasons for the high turnover ratio

### Data Structure
- Number of Rows: 7043
- Number of Columns: 21

### Columns/Features
1. `customerID` (str): Unique identifier for each customer.
2. `gender` (str): Customer's gender (e.g., Male, Female).
3. `SeniorCitizen` (int): Binary indicator of whether the customer is a senior citizen (0 for No, 1 for Yes).
4. `Partner` (str): Binary indicator of whether the customer has a partner (Yes, No).
5. `Dependents` (str): Binary indicator of whether the customer has dependents (Yes, No).
6. `tenure` (int): Number of months the customer has been with the company.
7. `PhoneService` (str): Binary indicator of whether the customer has phone service (Yes, No).
8. `MultipleLines` (str): Type of phone service (e.g., No phone service, Single line, Multiple lines).
9. `InternetService` (str): Type of internet service (e.g., DSL, Fiber optic, No internet service).
10. `OnlineSecurity` (str): Binary indicator of whether the customer has online security (Yes, No, No internet service).
11. `OnlineBackup` (str): Binary indicator of whether the customer has online backup (Yes, No, No internet service).
12. `DeviceProtection` (str): Binary indicator of whether the customer has device protection (Yes, No, No internet service).
13. `TechSupport` (str): Binary indicator of whether the customer has tech support (Yes, No, No internet service).
14. `StreamingTV` (str): Binary indicator of whether the customer has streaming TV (Yes, No, No internet service).
15. `StreamingMovies` (str): Binary indicator of whether the customer has streaming movies (Yes, No, No internet service).
16. `Contract` (str): Type of contract (e.g., Month-to-month, One year, Two year).
17. `PaperlessBilling` (str): Binary indicator of whether the customer uses paperless billing (Yes, No).
18. `PaymentMethod` (str): Customer's payment method (e.g., Electronic check, Mailed check, Bank transfer, Credit card).
19. `MonthlyCharges` (float): Monthly charges for the customer's services in USD.
20. `TotalCharges` (float): Total charges for the customer's services over their tenure.
21. `Churn` (str): Binary indicator of customer churn (Yes, No).

 ## Installation

To run this project, you need to have the required Python libraries installed. You can install them using pip:

```bash
pip install pandas==1.3.3
pip install numpy==1.21.2
pip install matplotlib==3.4.3
pip install seaborn==0.11.2
pip install scikit-learn==0.24.2
```

##  Import Libraries and Dependencies Used
```bash
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
```
## Visualize Missing Value
By using msno.matrix(churn), you can quickly get an overview of which columns in your dataset have missing values and the extent of the missing data. This visualization can be helpful for data cleaning and preprocessing tasks before performing data analysis or modeling.
```bash
# Visualize missing values as a matrix
msno.matrix(churn);
```
##  Data manipulation
churn = churn.drop(['customerID'], axis=1): This code removes the 'customerID' column from the DataFrame "churn." The drop function is used to drop a specific column (specified by its name) along the specified axis (in this case, axis=1 indicates columns).
```bash
churn= churn.drop(['customerID'], axis = 1)
churn.head()
```
This code converts the 'TotalCharges' column to a numeric data type using the pd.to_numeric function. The 'coerce' option is used to handle conversion errors by converting them to NaN (Not-a-Number) values.
```bash
churn['TotalCharges'] = pd.to_numeric(churn.TotalCharges, errors='coerce')
```
Check this nan values
```bash
churn.isnull().sum()
```
This line locates and displays rows where the 'TotalCharges' column has missing values (NaN).
```bash

churn.loc[churn['TotalCharges'].isnull()==True]
```
 This code removes rows with any missing values (NaN) from the DataFrame. The 'how' parameter is set to 'any' to indicate that any row containing at least one missing value should be dropped
 ```bash
#Remove missing value
churn.dropna(how='any',inplace=True)
```
This code maps the values in the 'SeniorCitizen' column, which originally contains 0 and 1, to more descriptive labels "No" and "Yes." It's a way to make the data more interpretable.
```bash
churn["SeniorCitizen"] = churn["SeniorCitizen"].map({0: "No", 1: "Yes"})
```
## Data Visualization
This code will produce a horizontal bar chart that displays the count of 'Churn' values, making it easy to visualize the distribution of the 'Churn' variable in your dataset. The x-axis represents the count, the y-axis represents the "Target Variable," and the title provides an overall description of the char
```bash
churn['Churn'].value_counts().plot(kind='barh',figsize=(8,6))
plt.xlabel('Count',labelpad=14)
plt.ylabel('Target Variable',labelpad=14)
plt.title('Count of Target Variable per category',y=1.02)
```
The code will generate a grouped histogram with 'Churn' on the x-axis, 'Contract' as the color-coding, and the specified title. The bars in the histogram will be grouped by contract type, making it easy to visualize the distribution of customer contracts with respect to churn.

```bash
fig = px.histogram(churn, x="Churn", color="Contract", barmode="group", title="<b>Customer contract distribution<b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()
```
This code, when executed, will produce a pie chart showing the distribution of payment methods used in the "churn" dataset, making it easy to visualize and understand how payments are distributed among the dataset's records.

```bash
labels = churn['PaymentMethod'].unique()
values = churn['PaymentMethod'].value_counts()

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_layout(title_text="<b>Payment Method Distribution</b>")
fig.show()
```
The resulting histogram will show the distribution of customer payment methods in different colors (corresponding to payment methods) with respect to customer churn. This visualization can help you understand how different payment methods relate to customer churn and provide insights into payment method preferences of customers who either churned or did not churn.
```bash
fig = px.histogram(churn, x="Churn", color="PaymentMethod", title="<b>Customer Payment Method distribution w.r.t. Churn</b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()
```
is code generates histograms for the 'tenure,' 'MonthlyCharges,' and 'TotalCharges' columns in your dataset using Plotly Express. Each histogram will show the distribution of data for the corresponding numeric column, and the number of bins for each histogram is set to 20. This can help you visualize the distribution of these numeric variables and understand their characteristics.
```bash
numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in numeric_columns:
    fig = px.histogram(churn, x=col, nbins=20, title=col)
    fig.show()
```
The resulting plot will be a box plot showing the distribution of 'Tenure' for each 'Churn' category (e.g., 'Yes' and 'No'). It provides insights into how customer tenure varies for those who churn and those who do not.
```bash
fig = px.box(churn, x='Churn', y = 'tenure')

# Update yaxis properties
fig.update_yaxes(title_text='Tenure (Months)', row=1, col=1)
# Update xaxis properties
fig.update_xaxes(title_text='Churn', row=1, col=1)

# Update size and title
fig.update_layout(autosize=True, width=750, height=600,
    title_font=dict(size=25, family='Courier'),
    title='<b>Tenure vs Churn</b>',
)

fig.show()
```

## Univariate Analysis
 This code generates a series of countplots for each categorical predictor in your dataset, helping you explore how the distribution of each category differs between customers who churn ('Yes') and those who don't ('No'). These plots can be valuable for understanding the impact of each categorical variable on customer churn.
 ```bash
for i, predictor in enumerate(churn.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=churn, x=predictor, hue='Churn')
```

## Data Preprocessing
The purpose of this function is to automate the process of converting categorical (object) data into numerical form so that it can be used in machine learning algorithms or other data analysis tasks that require numerical input. The LabelEncoder class assigns a unique integer to each category in the data, which can be useful for encoding categorical features into a format that can be processed by various machine learning models.

The code snippet you provided suggests that you are using a lambda function to convert object data types to integers for all columns in the "churn" DataFrame. 
```bash
def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

churn = churn.apply(lambda x: object_to_int(x))
churn.head()
```

## Splitting the Dataset
he code you provided is preparing the data for a machine learning model by splitting the "churn" DataFrame into features (X) and the target variable (y)
 X represents the input data, and y represents the corresponding target labels

 The train_test_split function is used to split your data into these sets. The test_size parameter is set to 0.2, which means that 20% of the data will be allocated to the testing set, and the remaining 80% will be used for training
 ```bash
X = churn.drop(columns = ['Churn'])
y = churn['Churn'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
```
## ML Model

### Decision Tree
The code is training a decision tree classifier, making predictions, calculating the accuracy, and then providing a detailed classification report for binary classification (class labels 0 and 1) to assess the model's performance. This is a common workflow when building and evaluating machine learning models.
```bash
model_dt=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)
model_dt.fit(X_train,y_train)
y_pred=model_dt.predict(X_test)
model_dt.score(X_test,y_test)
print(classification_report(y_test, y_pred, labels=[0,1]))
```
The output you provided appears to be the result of a classification model's performance evaluation using common metrics, such as precision, recall, and F1-score
```bash
  precision    recall  f1-score   support

           0       0.82      0.86      0.84      1028
           1       0.58      0.50      0.54       379

    accuracy                           0.77      1407
   macro avg       0.70      0.68      0.69      1407
weighted avg       0.76      0.77      0.76      1407
```
Precision: Precision is a measure of how many of the predicted positive instances are actually true positives. In this case:

For class 0 (not churned), the precision is 0.82, which means that 82% of the predicted not churned instances are correct.
For class 1 (churned), the precision is 0.58, indicating that 58% of the predicted churned instances are correct.

Recall: Recall, also known as sensitivity or true positive rate, measures how many of the actual positive instances the model correctly predicted. In this case:

For class 0, the recall is 0.86, indicating that 86% of the actual not churned instances are correctly predicted.
For class 1, the recall is 0.50, suggesting that 50% of the actual churned instances are correctly predicted.

F1-Score: The F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall, especially when there's an imbalance in the dataset. A higher F1-score indicates a better balance between precision and recall.

Accuracy: Accuracy measures the overall correctness of the model's predictions. In this case, the overall accuracy is 0.77, meaning that the model correctly predicted 77% of all instances.


### KNN - k-Nearest Neighbors
```bash
knn_model = KNeighborsClassifier(n_neighbors = 11)
knn_model.fit(X_train,y_train)
predicted_y = knn_model.predict(X_test)
accuracy_knn = knn_model.score(X_test,y_test)
print("KNN accuracy:",accuracy_knn)
```
### SVM - Support Vector Machine
```bash
svc_model = SVC(random_state = 1)
svc_model.fit(X_train,y_train)
predict_y = svc_model.predict(X_test)
accuracy_svc = svc_model.score(X_test,y_test)
print("SVM accuracy is :",accuracy_svc)
```
### Random Forest
```bash
rf_model = RandomForestClassifier(random_state=1)
rf_model.fit(X_train, y_train)
predict_y_rf = rf_model.predict(X_test)
accuracy_rf = rf_model.score(X_test, y_test)
print("Random Forest accuracy is:", accuracy_rf)
```
### Logistic Regression
```bash
lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)
accuracy_lr = lr_model.score(X_test,y_test)
print("Logistic Regression accuracy is :",accuracy_lr)
```
### Grandient Boosting Classifier
```bash
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
print("Gradient Boosting Classifier", accuracy_score(y_test, gb_pred))
```
### Voting Classifier
The Voting Classifier combines the predictions of the individual classifiers, and the final accuracy score is a measure of how well the ensemble model performs on the test data. It leverages the strengths of the individual classifiers to potentially improve predictive accuracy.
```bash
from sklearn.ensemble import VotingClassifier
clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression()
clf3 = RandomForestClassifier()
eclf1 = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('rf_model', clf3)], voting='soft')
eclf1.fit(X_train, y_train)
predictions = eclf1.predict(X_test)
print("Final Accuracy Score ")
print(accuracy_score(y_test, predictions))
```

The "Final Accuracy Score" of approximately 0.7989, or 79.89%, indicates the accuracy of the ensemble model on the test data. In this context, an accuracy of 79.89% means that roughly 79.89% of the test data points were correctly classified by the ensemble model.
