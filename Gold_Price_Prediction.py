#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# Data Collection and Processing

# In[2]:


# loading the csv data to a Pandas DataFrame
gold_data = pd.read_csv(r"C:\Users\mp2fb\Downloads\gld_price_data.csv")


# In[3]:


# print first 5 rows in the dataframe
gold_data.head()


# In[4]:


# print last 5 rows of the dataframe
gold_data.tail()


# In[5]:


# number of rows and columns
gold_data.shape


# In[6]:


# getting some basic informations about the data
gold_data.info()


# In[7]:


# checking the number of missing values
gold_data.isnull().sum()


# In[8]:


# getting the statistical measures of the data
gold_data.describe()

Correlation:

1.Positive Correlation
2.Negative Correlation
# In[8]:


correlation = gold_data.corr(numeric_only=True)


# # constructing a heatmap to understand the correlatiom

# In[10]:


plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Blues')


# In[9]:


# correlation values of GLD
print(correlation['GLD'])


# # checking the distribution of the GLD Price

# In[11]:


sns.distplot(gold_data['GLD'],color='green')


# In[12]:


X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']


# In[13]:


print(X)


# In[15]:


print(Y)


# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)


# ## Model Training: Random Forest Regressor

# In[17]:


model = RandomForestRegressor(n_estimators=100)


# In[18]:


# training the model
model.fit(X_train,Y_train)


# ### Model Evaluation

# In[20]:


# prediction on Test Data
test_data_prediction = model.predict(X_test)


# In[21]:


print(test_data_prediction)


# In[22]:


import numpy as np
from sklearn import metrics


# In[23]:


print('MAE:', metrics.mean_absolute_error(Y_test, test_data_prediction))
print('MSE:', metrics.mean_squared_error(Y_test, test_data_prediction)) 
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test,test_data_prediction))) 



# In[24]:


model.score(X_train,Y_train)


# In[ ]:





# In[28]:


plt.figure(figsize=(8, 6))
plt.scatter(Y_test, test_data_prediction, color='blue', label='Actual vs Predicted')  # Adding label for blue points
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--', label='Ideal Line')  # Adding label for red
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.legend()  # Displaying the legend
plt.show()


# ## Model Training: LinearRegression

# In[29]:


from sklearn.linear_model import LinearRegression


# In[30]:


model = LinearRegression()


# In[31]:


model.fit(X_train,Y_train)


# In[32]:


test_data_prediction = model.predict(X_test)


# In[33]:


print(test_data_prediction)


# In[34]:


model.score(X_train,Y_train)


# In[ ]:




