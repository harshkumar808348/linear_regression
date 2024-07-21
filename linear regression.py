#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[11]:


from sklearn.datasets import fetch_california_housing


# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[13]:


df = fetch_california_housing()


# In[14]:


df


# In[16]:


dataset = pd.DataFrame(df.data)


# In[17]:


dataset


# In[19]:


dataset.columns = df.feature_names


# In[20]:


dataset.head()


# In[25]:


## devide the data in dependent and independent features
X = dataset
y = df.target


# In[22]:


y


# In[26]:


### train test isplat 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)


# In[27]:


X_train


# In[28]:


## standardizing the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[30]:


X_train = scaler.fit_transform(X_train)


# In[31]:


X_test = scaler.transform(X_test)


# In[32]:


X_train


# In[33]:


scaler.inverse_transform(X_train)


# In[36]:


from sklearn.linear_model import LinearRegression
##cross validtion
from sklearn.model_selection import cross_val_score


# In[44]:


regression = LinearRegression()
regression.fit(X_train,y_train)


# In[40]:


mse  = cross_val_score(regression , X_train , y_train , scoring = 'neg_mean_squared_error' , cv =5)


# In[41]:


np.mean(mse)


# In[42]:


## prediction


# In[45]:


reg_pred = regression.predict(X_test)


# In[46]:


reg_pred


# In[49]:


import seaborn as sns
sns.displot(reg_pred-y_test,kind  = 'kde')


# In[50]:


from sklearn.metrics import r2_score


# In[51]:


score = r2_score(reg_pred , y_test)


# In[52]:


score


# In[ ]:




