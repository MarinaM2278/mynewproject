import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer,  StandardScaler, LabelEncoder

import sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import PolynomialFeatures

#from sklearn.base import TransformerMixin
from sklearn_pandas import DataFrameMapper

import pickle
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
from sklearn.metrics import classification_report



from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split


pd.set_option('display.max_rows', 100)
#np.set.printoptions(prediction = 4)


# In[2]:


df1 = pd.read_csv('data/14100307.csv', usecols=['REF_DATE', 'GEO', 'Wages', 'Type of work','National Occupational Classification (NOC)','Sex', 'Age group','UOM', 'VALUE'])


# In[ ]:


df1.head()


# In[3]:


df1 = df1[df1.GEO=='Canada']


# In[ ]:


#df1


# In[4]:


df1 = df1[df1.Wages =='Total employees, all wages']


# In[5]:


df1 = df1[df1.UOM =='Persons']


# In[6]:


df1 = df1.drop(['REF_DATE','Wages','GEO', 'UOM'], axis = 1)


# In[ ]:


df1


# In[7]:


df1 = df1.rename(columns={'Type of work': 'fulltime_parttime', 'National Occupational Classification (NOC)': 'occupation', 'Sex': 'gender', 'Age group': 'age_group', 'VALUE':'value'})


# In[8]:


df1 = df1[df1.fulltime_parttime != 'Both full- and part-time employees']


# In[9]:


df1 = df1[df1.gender !='Both sexes']


# In[10]:


df1 = df1[df1.occupation != 'Total employees, all occupations']


# In[11]:


df1 = df1[df1.age_group != '15 years and over']


# In[12]:


#df.fillna(df.mean(), inplace=True)

df1['value'] = df1['value'].fillna((df1['value'].mean()))


# In[60]:


#df1['value']


# In[13]:


df1['gender'] = df1['gender'].apply({'Females': 1, 'Males': 0}.get)


# In[14]:


df1


# In[ ]:


#df1["age"].replace({"15 to 24 years": 15, "25 to 54 years": 25, "55 years and over": 55}, inplace=True)


# In[15]:


df1['gender'].value_counts()


# In[18]:


#sns.lmplot(x='gender', y='value', data=df1)


# In[19]:


df1['value'].isnull().sum()


# In[20]:


target = 'value'
y = df1[target]
X = df1.drop(target, axis = 1)


# In[ ]:


X.info()


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)


# In[22]:


y_train


# In[ ]:


#y_train = np.nan_to_num(y_train)
#y_test = np.nan_to_num(y_test)


# In[23]:


y_train


# In[24]:


X_train.head()


# In[ ]:





# In[25]:


le = preprocessing.LabelEncoder()

le.fit(X_train['occupation'])
le.fit(X_test['occupation'])

le.fit(X_train['fulltime_parttime'])
le.fit(X_test['fulltime_parttime'])

le.fit(X_train['age_group'])
le.fit(X_test['age_group'])

le.fit(X_train['gender'])
le.fit(X_test['gender'])


# In[29]:


mapper = DataFrameMapper([
    (['occupation'], [LabelBinarizer()]),
    (['gender'], [LabelBinarizer()]),
    (['fulltime_parttime'], [LabelBinarizer()]),
    (['age_group'], [LabelBinarizer()])
    ], df_out= True)


# In[30]:


mapper


# In[31]:


Z_train= mapper.fit(X_train)


# In[32]:


Z_train= mapper.transform(X_train)


# In[33]:


Z_train


# In[34]:


Z_test = mapper.transform(X_test)


# In[35]:


Z_test


# In[36]:


model = LinearRegression(normalize=False)


# In[37]:


model.fit(Z_train, y_train)


# In[38]:


y_pred = model.predict(Z_test)


# In[39]:


y_pred


# In[40]:


from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:





# In[ ]:





# In[ ]:


Z_train.shape


# In[41]:


from sklearn.metrics import mean_squared_error, r2_score

def rmse_score(model, Z_train, Z_test, y_train, y_test):
    mse_train = mean_squared_error(y_true = y_train,
                                  y_pred = model.predict(Z_train))
    mse_test = mean_squared_error(y_true = y_test,
                                  y_pred = model.predict(Z_test))
    rmse_train = mse_train ** 0.5/100
    rmse_test = mse_test ** 0.5/100

    print("The training RMSE for " + str(model) + " is: " + str(rmse_train))
    print("The testing RMSE for " + str(model) + "\ is: " + str(rmse_test))
    return (rmse_train, rmse_test)


# In[42]:


rmse_score(model, Z_train, Z_test, y_train, y_test)


# In[43]:


y_test


# In[44]:


y_pred


# In[45]:


r2_score(y_test, y_pred)


# In[46]:


Z_test.shape


# In[47]:


y_test.size


# In[48]:


y_pred.size


# In[49]:


knn_reg = KNeighborsRegressor()
knn_reg.fit(Z_train, y_train)

rmse_score(knn_reg, Z_train, Z_test, y_train, y_test)


# In[50]:


cart_reg = DecisionTreeRegressor()
cart_reg.fit(Z_train, y_train)

rmse_score(cart_reg, Z_train, Z_test, y_train, y_test)


# In[51]:


bagged_reg = BaggingRegressor()
bagged_reg.fit(Z_train, y_train)

rmse_score(bagged_reg, Z_train, Z_test, y_train, y_test)




# In[52]:


adaboost_reg = AdaBoostRegressor()
adaboost_reg.fit(Z_train, y_train)

rmse_score(adaboost_reg, Z_train, Z_test, y_train, y_test)


# In[53]:


support_vector_reg = SVR()
support_vector_reg.fit(Z_train, y_train)

rmse_score(support_vector_reg, Z_train, Z_test, y_train, y_test)


# In[54]:


pipe = Pipeline([("mapper", mapper), ("model", adaboost_reg)])
pipe.fit(X_train, y_train)


# In[55]:


employed_value = pipe.predict(X_test)


# In[56]:


employed_value = (np.round(employed_value, 1)).astype('int')


# In[57]:


employed_value


# In[58]:


pickle.dump(pipe, open('pipe.pkl', 'wb'))
del pipe
pipe = pickle.load(open('pipe.pkl', 'rb'))
pipe
