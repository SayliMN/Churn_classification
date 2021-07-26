#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('/Users/saylinarkhede/Documents/Jupyter/Projects/Churn_classification/train.csv')
test = pd.read_csv('/Users/saylinarkhede/Documents/Jupyter/Projects/Churn_classification/test.csv')


# In[6]:


print(train.shape)
train.head()


# In[7]:


print(test.shape)
test.head()


# In[8]:


train.columns


# In[10]:


from sklearn.preprocessing import OrdinalEncoder


# In[11]:


feature_ohn = ['state','area_code','international_plan','voice_mail_plan','churn']


# In[13]:


encoder = OrdinalEncoder()
train[feature_ohn] = encoder.fit_transform(train[feature_ohn])


# In[15]:


train.head()


# ## Correlation coefficient plot

# In[16]:


import seaborn as sns


# In[22]:


fig = plt.figure(figsize=(20,16))
sns.heatmap(data=train.corr(method='kendall'),
            cmap='Spectral',
            fmt='.3f',
            annot=True)


# In[25]:


corr_fea = train.corr(method='kendall')['churn']
corr_fea


# In[31]:


cols = corr_fea[corr_fea >= 0.1].index


# In[32]:


vals = corr_fea[corr_fea >= 0.1].values


# In[37]:


ken_dataFrame = pd.DataFrame({
                        'Features': cols,
                        'Correlation coefficient': vals
                    })
ken_dataFrame.sort_values(by=['Correlation coefficient'], ascending=False)


# # Model 

# In[41]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


# In[42]:


train['churn'].value_counts()


# In[48]:


X = train.drop('churn', axis=1)
print(X.shape)
y = train['churn']
print(y.shape)


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[57]:


X_test.shape


# In[52]:


rf = RandomForestClassifier()

cv = StratifiedKFold(n_splits=6)

param_distributions = ({
    'n_estimators':[100,150,200,250,300,400,500],
    'max_depth': [5,6,7,8,9]
})

model = RandomizedSearchCV(rf,
                           cv=cv,
                           param_distributions=param_distributions,
                           verbose=1)


# In[58]:


model.fit(X_train,
          y_train)


# In[59]:


model.best_estimator_


# In[75]:


X_train.columns[model.best_estimator_.feature_importances_ >= 0.09]


# In[76]:


model.best_estimator_.score(X_train,y_train)


# In[77]:


model.best_estimator_.score(X_test,y_test)


# In[79]:


y_pred = model.best_estimator_.predict(X_test)


# In[80]:


from sklearn.metrics import accuracy_score, recall_score, precision_score


# In[84]:


print('Accuracy: ',accuracy_score(y_test, y_pred))
print('Recall: ',recall_score(y_test, y_pred))
print('Precision: ',precision_score(y_test, y_pred))

