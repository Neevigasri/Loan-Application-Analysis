#!/usr/bin/env python
# coding: utf-8

# In[ ]:


LOAN APPLICATION ANALYSIS


# In[75]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[76]:


#loading data set
import pandas as pd

#dataset path
file_path = r"C:\Users\neevi\Downloads\archive (1)\loan_data.csv"

#read the CSV file
df = pd.read_csv(file_path)

#display first few rows
print(df.head())


# In[50]:


#dataset information
df.info()


# In[51]:


#dataset shape
df.shape


# In[52]:


#---DATA CLEANING---
#checking missing values
df.isnull().sum()


# In[53]:


#fill the missing value

#null values in "LoanAmount" column is replaced by "mean"
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())

#null values in "Credit_History" column is replaced by "median"
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())


# In[54]:


#check null values in LoanAmount & Credit_History
df.isnull().sum()


# In[55]:


#drop remaining missing values 
df.dropna(inplace=True)


# In[56]:


#check missing values for the final time
df.isnull().sum()


# In[57]:


#final dataset shape
df.shape


# In[58]:


#EXPLORATORY DATA ANALYSIS (Visualisation)


# In[59]:


#comparision between paerameter in getting the loan
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20,20))

sns.set(font_scale=1.5)

#Gender vs Loan_Status
plt.subplot(3,2,1)
sns.countplot(x='Gender', hue='Loan_Status',data=df)
plt.title('Gender vs LoanStatus')

#Married vs Loan_Status
plt.subplot(3,2,2)
sns.countplot(x='Married', hue='Loan_Status',data=df)
plt.title('Married vs LoanStatus')

#Education vs Loan_Status
plt.subplot(3,2,3)
sns.countplot(x='Education', hue='Loan_Status',data=df)
plt.title('Education vs LoanStatus')

#Self_Employed vs Loan_Status
plt.subplot(3,2,4)
sns.countplot(x='Self_Employed', hue='Loan_Status',data=df)
plt.title('Self_Employed vs LoanStatus')

#Property_Area vs Loan_Status
plt.subplot(3,2,5)
sns.countplot(x='Property_Area', hue='Loan_Status',data=df)
plt.title('Property_Area vs LoanStatus')

plt.tight_layout()

plt.show()


# In[60]:


# REPLACE THE VARIABLE VALUES TO NUMERICAL VALUES FOR BUILDING MODEL


# In[77]:


df.Loan_Status=df.Loan_Status.map({'Y':1, 'N':0})
df['Loan_Status'].value_counts()


# In[78]:


df.Gender=df.Gender.map({'Male':1, 'Female':0})
df['Gender'].value_counts()


# In[79]:


df.Married=df.Married.map({'Yes':1, 'No':0})
df['Married'].value_counts()


# In[80]:


df.Dependents=df.Dependents.map({'0':0, '1':1, '2':2, '3+':3})
df['Dependents'].value_counts()


# In[81]:


df.Education=df.Education.map({'Graduate':1, 'Not Graduate':0 })
df['Education'].value_counts()


# In[83]:


df.Self_Employed=df.Self_Employed.map({'Yes':1, 'No':0 })
df['Self_Employed'].value_counts()


# In[84]:


df.Property_Area=df.Property_Area.map({'Rural':0, 'Semiurban':1, 'Urban':2 })
df['Property_Area'].value_counts()


# In[85]:


df.info()


# In[86]:


df.head()


# In[88]:


#IMPORTING PACKAGE FOR CLASSIFICATION ALGORITHM

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[91]:


#splitting the data into "train" and "test" set

X=df.iloc[1:381,1:13].values
Y=df.iloc[1:381,1:13].values


# In[92]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,random_state=0)


# In[96]:


#LOGISTIC REGRESSION

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#dataset path
file_path = r"C:\Users\neevi\Downloads\archive (1)\loan_data.csv"

#read the CSV file
df = pd.read_csv(file_path)

#seperate features and target variables
X=df.drop('Loan_Status', axis=1)
Y=df['Loan_Status']

#identify numerical and categorial columns
numerical_cols = X.select_dtypes(include=[np.number]).columns
categorial_cols = X.select_dtypes(exclude=[np.number]).columns

#Apply SimpleImputer
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

#Impute missing values
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
X[categorial_cols] = cat_imputer.fit_transform(X[categorial_cols])

#Convert categorial varial to numerical 
X = pd.get_dummies(X, drop_first=True)

#Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#Intialize the standardScaler
scaler = StandardScaler()

#Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

#Transform the test data
X_test_scaled = scaler.transform(X_test)

#Initialize the logistic regression model with incresed max_iter
model = LogisticRegression(max_iter=2000)

#fit the model with scaled training data
model.fit(X_train_scaled, Y_train)

#make prediction on the scaled test set
lr_prediction = model.predict(X_test_scaled)

#Evaluate the model
print('Logistic Regression Accuracy=', metrics.accuracy_score(Y_test, lr_prediction))


# In[97]:


print("Y_Predicted",lr_prediction)
print("Y_test",Y_test)


# In[ ]:


CONCLUSION:
    The Loan Status is heavily dependent on the Credit History for Predictions.
    The Logistic Regression algorithm gives us the maximum Accuracy (80% approx) compared to the other 3 Machine Learning 
Classification Algorithm.

