#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[15]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


df = pd.read_csv("C:\\Users\\dhruven\\cancer_classification.csv")


# In[ ]:


df


# In[18]:


df.columns


# In[19]:


plt.figure(figsize=(20,30))
sns.heatmap(df.corr(),annot=True)


# In[20]:


X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


from sklearn.preprocessing import MinMaxScaler


# In[24]:


scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train.shape


# In[25]:


from keras.models import Sequential
from keras.layers import Dense,Dropout


# In[26]:


model = Sequential()


# In[27]:


model.add(Dense(units = 30,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')


# In[ ]:





# In[191]:


model.fit(X_train,y_train,epochs=600,validation_data=(X_test,y_test))


# In[192]:


loss = pd.DataFrame(model.history.history)


# In[193]:


loss.plot()


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[217]:


print(classification_report(y_test,pred))

print(confusion_matrix(y_test,pred))


# In[194]:


model.add(Dense(units = 30,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy')


# In[195]:


from tensorflow.keras.callbacks import EarlyStopping


# In[196]:


early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=20)


# In[197]:


model.fit(X_train,y_train,epochs=250,validation_data=(X_test,y_test),callbacks=[early_stop])


# In[198]:


loss = pd.DataFrame(model.history.history)


# In[218]:


plt.figure(figsize=(12,12))
loss.plot()


# In[200]:


pred = model.predict_classes(X_test)


# In[201]:


from sklearn.metrics import classification_report,confusion_matrix


# In[202]:


print(classification_report(y_test,pred))


# In[203]:


print(confusion_matrix(y_test,pred))


# In[212]:


patient = df.drop('benign_0__mal_1',axis=1).iloc[48]


# In[213]:


patient
df.shape


# In[214]:


patient = scaler.transform(patient.values.reshape(-1,30))


# In[215]:


model.predict_classes(patient)


# In[ ]:




