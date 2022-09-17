#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/chaitanyabaranwal/ParkinsonAnalysis/master/parkinsons.csv')


# In[3]:


df.head()


# In[4]:


X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[5]:


X.shape


# In[6]:


y.shape


# In[23]:


from sklearn.preprocessing import MinMaxScaler
obj = MinMaxScaler((-1,1))
X = obj.fit_transform(X)


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[24]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)


# In[26]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)*100


# In[ ]:




