import pandas as pd
#import seaborn as sns
import numpy as np
import glob, os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import preprocessing

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer,  StandardScaler, LabelEncoder

import sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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
pd.set_option('display.max_colwidth', None)


# In[2]:


##Rename multiple csv files in a folder with Python
def rename(dir, pathAndfilename, pattern, tittlePattern):
    os.rename(pathAndfilename, os.path.join(dir, tittlePattern))


# In[3]:


##search for csv files in the working folderp
path = os.path.expanduser('data/*.csv*')


# In[4]:


##iterate and rename them one by one with the number of the iteration
try:
    for i, fname in enumerate(glob.glob(path)):
        rename(os.path.expanduser('data/'), fname, r'*csv', r'test{}.csv'.format(i))
except:
    pass


# In[5]:


result = pd.DataFrame()


# In[6]:


for  fname in  glob.glob(path):
    head, tail = os.path.split(fname)
    df = pd.read_csv(fname, sep = ',')
    df3 = df.sort_values(by=['REF_DATE'], ascending = True).drop(['DGUID', 'SCALAR_ID'], axis =1)
    df3['channel']= tail
    result = pd.concat([result, df3])


# In[7]:


df1 = result.drop(['UOM_ID','SCALAR_FACTOR', 'VECTOR', 'STATUS','COORDINATE','SYMBOL', 'TERMINATED', 'DECIMALS', 'channel'], axis = 1)


# In[8]:


df1 = df1[df1.GEO=='Canada']


# In[9]:


df1.isnull().sum()


# In[10]:


df1 = df1[df1.Wages =='Total employees, all wages']


# In[11]:


df1 = df1[df1.UOM =='Persons']


# In[12]:


df1 = df1.drop(['REF_DATE','Wages','GEO', 'UOM'], axis = 1)


# In[13]:


df1 = df1.rename(columns={'Type of work': 'fulltime_parttime', 'National Occupational Classification (NOC)': 'occupation', 'Sex': 'gender', 'Age group': 'age_group', 'VALUE':'value'})


# In[14]:


df1 = df1[df1.fulltime_parttime != 'Both full- and part-time employees']


# In[15]:


df1 = df1[df1.gender !='Both sexes']


# In[16]:


df1 = df1[df1.occupation != 'Total employees, all occupations']


# In[17]:


df1 = df1[df1.age_group != '15 years and over']


# In[18]:


#df.fillna(df.mean(), inplace=True)

df1['value'] = df1['value'].fillna((df1['value'].mean()))


# In[19]:


#df1['value']


# In[20]:


df1['gender'] = df1['gender'].apply({'Females': 1, 'Males': 0}.get)


# In[21]:


df1['gender'].value_counts()


# In[22]:


#sns.lmplot(x='gender', y='value', data=df1)


# In[23]:


df1['value'].isnull().sum()


# In[24]:


target = 'value'
y = df1[target]
X = df1.drop(target, axis = 1)


# In[25]:


X.info()


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)


# In[27]:


y_train


# In[28]:


y_train


# In[29]:


le = preprocessing.LabelEncoder()

le.fit(X_train['occupation'])
le.fit(X_test['occupation'])

le.fit(X_train['fulltime_parttime'])
le.fit(X_test['fulltime_parttime'])

le.fit(X_train['age_group'])
le.fit(X_test['age_group'])

le.fit(X_train['gender'])
le.fit(X_test['gender'])


# In[30]:


mapper = DataFrameMapper([
    (['occupation'], [LabelBinarizer()]),
    (['gender'], [LabelBinarizer()]),
    (['fulltime_parttime'], [LabelBinarizer()]),
    (['age_group'], [LabelBinarizer()])
    ], df_out= True)


# In[31]:


mapper


# In[32]:


Z_train= mapper.fit(X_train)


# In[33]:


Z_train= mapper.transform(X_train)


# In[34]:


Z_train


# In[35]:


Z_test = mapper.transform(X_test)


# In[36]:


Z_test


# In[41]:


model = LinearRegression(normalize=False)


# In[42]:


model.fit(Z_train, y_train)


# In[43]:


y_pred = model.predict(Z_test)


# In[44]:


y_pred


# In[45]:


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


# In[46]:


rmse_score(model, Z_train, Z_test, y_train, y_test)


# In[47]:


y_test


# In[48]:


y_pred


# In[49]:


r2_score(y_test, y_pred)


# In[50]:


Z_test.shape


# In[51]:


y_test.size


# In[52]:


y_pred.size


# In[53]:


knn_reg = KNeighborsRegressor()
knn_reg.fit(Z_train, y_train)

rmse_score(knn_reg, Z_train, Z_test, y_train, y_test)


# In[54]:


cart_reg = DecisionTreeRegressor()
cart_reg.fit(Z_train, y_train)

rmse_score(cart_reg, Z_train, Z_test, y_train, y_test)


# In[55]:


bagged_reg = BaggingRegressor()
bagged_reg.fit(Z_train, y_train)

rmse_score(bagged_reg, Z_train, Z_test, y_train, y_test)




# In[56]:


adaboost_reg = AdaBoostRegressor()
adaboost_reg.fit(Z_train, y_train)

rmse_score(adaboost_reg, Z_train, Z_test, y_train, y_test)


# In[57]:


support_vector_reg = SVR()
support_vector_reg.fit(Z_train, y_train)

rmse_score(support_vector_reg, Z_train, Z_test, y_train, y_test)


# In[58]:


pipe = Pipeline([("mapper", mapper), ("model", model)])
pipe.fit(X_train, y_train)


# In[59]:


employed_value = pipe.predict(X_test)


# In[60]:


employed_value = (np.round(employed_value, 1)).astype('int')


# In[61]:


employed_value


# In[62]:


pickle.dump(pipe, open('pipe.pkl', 'wb'))
del pipe
pipe = pickle.load(open('pipe.pkl', 'rb'))
pipe
