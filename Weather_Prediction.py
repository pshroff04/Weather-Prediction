
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[2]:


data = pd.read_csv('./weather/daily_weather.csv')


# In[7]:


#check for NaN
data[data.isnull().any(axis=1)]


# In[8]:


data =data.dropna()


# In[9]:


del data['number']


# In[11]:


data.shape


# In[12]:


clean_data = data.copy()


# In[13]:


#creating target variable- 'high_humidity_label'
clean_data['high_humidity_label'] = (clean_data['relative_humidity_3pm']>24.99)*1


# In[14]:


print(clean_data['high_humidity_label'])


# In[72]:


y = clean_data[['high_humidity_label']].copy()


# In[73]:


clean_data.columns


# In[19]:


features = ['air_pressure_9am', 'air_temp_9am', 'avg_wind_direction_9am',
       'avg_wind_speed_9am', 'max_wind_direction_9am', 'max_wind_speed_9am',
       'rain_accumulation_9am', 'rain_duration_9am', 'relative_humidity_9am',
       'relative_humidity_3pm']


# In[22]:


#Preparing features
X = clean_data[features].copy()



# In[75]:


#Building model
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,random_state=324)


# In[77]:


type(y_test)


# In[41]:


y_test.describe()


# In[78]:


#create classifier object
humidity_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
humidity_classifier.fit(X_train,y_train)


# In[79]:


# In[80]:


Predictions = humidity_classifier.predict(X_test)


# In[83]:


accuracy_score(y_true=y_test, y_pred=Predictions)


# In[82]:


y_test['high_humidity_label'][:10]


# In[84]:


Predictions[:10]

