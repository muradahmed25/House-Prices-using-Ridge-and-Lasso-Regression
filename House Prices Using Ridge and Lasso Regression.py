#!/usr/bin/env python
# coding: utf-8

# # PROBLEM STATEMENT

# - Dataset includes house sale prices for King County in USA. 
# - Homes that are sold in the time period: May, 2014 and May, 2015.
# - Data Source: https://www.kaggle.com/harlfoxem/housesalesprediction
# 
# - Columns:
#     - ida: notation for a house
#     - date: Date house was sold
#     - price: Price is prediction target
#     - bedrooms: Number of Bedrooms/House
#     - bathrooms: Number of bathrooms/House
#     - sqft_living: square footage of the home
#     - sqft_lot: square footage of the lot
#     - floors: Total floors (levels) in house
#     - waterfront: House which has a view to a waterfront
#     - view: Has been viewed
#     - condition: How good the condition is ( Overall )
#     - grade: overall grade given to the housing unit, based on King County grading system
#     - sqft_abovesquare: footage of house apart from basement
#     - sqft_basement: square footage of the basement
#     - yr_built: Built Year
#     - yr_renovated: Year when house was renovated
#     - zipcode: zip
#     - lat: Latitude coordinate
#     - long: Longitude coordinate
#     - sqft_living15: Living room area in 2015(implies-- some renovations) 
#     - sqft_lot15: lotSize area in 2015(implies-- some renovations)

# # STEP #0: LIBRARIES IMPORT
# 

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# # STEP #1: IMPORT DATASET

# In[2]:


house_df = pd.read_csv('kc_house_data.csv', encoding = 'ISO-8859-1')


# In[3]:


house_df


# In[4]:


house_df.head(20)


# In[5]:


house_df.tail(10)


# In[6]:


house_df.describe()


# In[7]:


house_df.info()


# # STEP #2: VISUALIZE DATASET

# In[8]:


sns.scatterplot(x = 'bedrooms', y = 'price', data = house_df)


# In[9]:


sns.scatterplot(x = 'sqft_living', y = 'price', data = house_df)


# In[10]:


sns.scatterplot(x = 'sqft_lot', y = 'price', data = house_df)


# In[11]:


house_df.hist(bins=20,figsize=(20,20), color = 'r')


# In[12]:


f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(house_df.corr(), annot = True)


# In[13]:


sns.pairplot(house_df)


# In[14]:


# pick a sample of the data
house_df_sample =house_df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built']]


# In[15]:


sns.pairplot(house_df_sample)


# # STEP #3: CREATE TESTING AND TRAINING DATASET/DATA CLEANING

# In[16]:


selected_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors', 'sqft_above', 'sqft_basement', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'yr_built', 
'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']

X = house_df[selected_features]


# In[17]:


X 


# In[18]:


y = house_df['price']


# In[19]:


y


# In[20]:


X.shape


# In[21]:


y.shape


# # STEP#4: TRAINING THE MODEL

# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[23]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept =True)
regressor.fit(X_train,y_train)
print('Linear Model Coefficient (m): ', regressor.coef_)
print('Linear Model Coefficient (b): ', regressor.intercept_)


# # STEP #5: EVALUATE MODEL

# In[24]:


y_predict = regressor.predict( X_test)
y_predict


# In[25]:


plt.plot(y_test, y_predict, "^", color = 'r')
plt.xlim(0, 3000000)
plt.ylim(0, 3000000)

plt.xlabel("Model Predictions")
plt.ylabel("True Value (ground Truth)")
plt.title('Linear Regression Predictions')
plt.show()


# In[26]:


k = X_test.shape[1]
n = len(X_test)


# In[27]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 


# # RIDGE REGRESSION

# In[28]:


from sklearn.linear_model import Lasso, Ridge
regressor_ridge = Ridge(alpha = 50)
regressor_ridge.fit(X_train, y_train)
print('Linear Model Coefficient (m): ', regressor_ridge.coef_)
print('Linear Model Coefficient (b): ', regressor_ridge.intercept_)

y_predict = regressor_ridge.predict( X_test)
y_predict


# # LASSO REGRESSION

# In[31]:


from sklearn.linear_model import Lasso
regressor_lasso = Lasso(alpha = 500)
regressor_lasso.fit(X_train,y_train)
print('Linear Model Coefficient (m): ', regressor_lasso.coef_)
print('Linear Model Coefficient (b): ', regressor_lasso.intercept_)

y_predict = regressor_lasso.predict( X_test)
y_predict


# In[32]:



plt.plot(y_test, y_predict, "^", color = 'r')
plt.xlim(0, 3000000)
plt.ylim(0, 3000000)

plt.xlabel("Model Predictions")
plt.ylabel("True Value (ground Truth)")
plt.title('Linear Regression Predictions')
plt.show()


# In[33]:


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 

