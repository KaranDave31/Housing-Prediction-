#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


df = pd.read_csv('../DATA/kc_house_data.csv')


# In[29]:


df.isnull()


# In[30]:


df.isnull().sum()


# In[31]:


df.head()


# In[32]:


df.describe()


# In[33]:


df.describe().transpose()


# In[34]:


plt.figure(figsize=(10,6))


# In[35]:


sns.distplot(df['price'])


# In[36]:


sns.countplot(data=df,x='bedrooms')


# In[37]:


df.corr(numeric_only=True)['price'].sort_values()


# In[38]:


sns.scatterplot(x='price',y='sqft_living',data=df)


# In[39]:


sns.scatterplot(x='price',y='grade',data=df)


# In[40]:


sns.boxplot(x='bedrooms',y='price',data=df)


# In[41]:


sns.scatterplot(x='price',y='long',data=df)


# In[42]:


sns.scatterplot(x='price',y='lat',data=df)


# In[43]:


sns.scatterplot(x='long',y='lat',data=df,hue='price')


# In[44]:


df.sort_values('price',ascending=False).head()


# In[45]:


len(df)


# In[46]:


non_top_one = df.sort_values('price',ascending=False).iloc[216:]


# In[47]:


plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=non_top_one,hue='price',edgecolor=None,alpha=0.2,palette='RdYlGn')


# In[48]:


sns.boxplot(x='waterfront',y='price',data=df)


# In[51]:


df = df.drop('id',axis=1)


# In[52]:


df.head()


# In[53]:


df['date'] = pd.to_datetime(df['date'])


# In[54]:


df['date']


# In[55]:


df['year'] = df['date'].apply(lambda date: date.year)


# In[56]:


df['month'] = df['date'].apply(lambda date: date.month)


# In[57]:


df.head()


# In[58]:


sns.boxplot(x='month',y='price',data=df)


# In[61]:


df.groupby('month').mean()['price']


# In[62]:


df.groupby('month').mean()['price'].plot()


# In[63]:


df.groupby('year').mean()['price'].plot()


# In[64]:


df = df.drop('date',axis=1)


# In[65]:


df = df.drop('zipcode',axis=1)


# In[66]:


df['yr_renovated'].value_counts()


# In[67]:


df['sqft_basement'].value_counts()


# In[80]:


X = df.drop('price',axis=1).values
y = df['price'].values


# In[81]:


from sklearn.model_selection import train_test_split


# In[82]:


X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=101)


# In[83]:


from sklearn.preprocessing import MinMaxScaler


# In[84]:


scaler = MinMaxScaler()


# In[85]:


X_train = scaler.fit_transform(X_train)


# In[86]:


X_test = scaler.fit_transform(X_test)


# In[87]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[88]:


X_train.shape


# In[89]:


model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# In[91]:


model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),
         batch_size=128, epochs=400)


# In[95]:


losses = pd.DataFrame(model.history.history)


# In[96]:


losses.plot()


# In[97]:


from sklearn.metrics import mean_absolute_error, mean_squared_error,explained_variance_score


# In[98]:


predictions = model.predict(X_test)


# In[99]:


predictions


# In[100]:


mean_absolute_error(y_test,predictions)


# In[101]:


mean_squared_error(y_test,predictions)


# In[102]:


df['price'].describe()


# In[103]:


explained_variance_score(y_test,predictions)


# In[106]:


plt.figure(figsize=(12,6))
plt.scatter(y_test,predictions)
plt.plot(y_test,y_test,'r')


# In[107]:


single_house = df.drop('price',axis=1).iloc[0]


# In[109]:


single_house = scaler.transform(single_house.values.reshape(-1,19))


# In[110]:


model.predict(single_house)


# In[111]:


df.head(1)


# In[ ]:




