#!/usr/bin/env python
# coding: utf-8

# #  House Predictor

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


housing =  pd.read_csv('housing_data.csv')


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


#housing.hist(bins=30,figsize=(20,15))
#plt.show()


# # Train test split

# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


#X = housing.iloc[:, 0:-1].values
#y= housing.iloc[:,13].values


# In[11]:


#X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[12]:


train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
print(f"size of training set ==> {len(train_set)}\nsize of test set ==> {len(test_set)}" )


# # Stratified Shuffle - To distribute 'CHAS' column data to train and test sets properly

# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
print(split)

#shuffle the CHAS column data to train and test set equally

for train_index,test_index in split.split(housing,housing['CHAS']):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]


# In[14]:


#start_test_set['CHAS'].value_counts()


# In[15]:


94/8


# In[16]:


housing = start_train_set.copy()


# # Looking for correlations

# In[17]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
#RM has strong positive corelation with MEDV than ZN,B,DIS and LSTAT has negative corelation means if LSTAT decrease price will increase


# # Now draw a correlation graph

# In[18]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))


# In[19]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)

#After analysing this scatter plot we can ask 1 questions to DB team -
# why MEDV is capped at 50 for every value of RM ? -# we can remove the outliers/noise from the data so that it won't affect our predictions
# RM increasing hence MEDV increasing - same thing we have find above using housing.corr() method of sklearn


# # Trying Attribute/Feature combinations to make new feature like tax per RM

# In[20]:


#housing["TAXRM"] = housing['TAX']/housing['RM']


# In[21]:


#housing.head()


# In[22]:


#corr_matrix = housing.corr()
#corr_matrix['MEDV'].sort_values(ascending=False)


# In[23]:


#housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# # Assign X features & Y labels 

# In[24]:


housing_labels = housing["MEDV"].copy() # this is Y label
housing = housing.drop("MEDV", axis=1) # this is X features/ Variables attributes


# In[25]:


#housing_labels.head()


# 
# # Handle Missing Values -Preprocessing Step
# To take care of missing attributes, you have three options:
#     1. Get rid of the missing data points
#     2. Get rid of the whole attribute
#     3. Set the value to some value(0, mean or median)
# In[26]:


#a = housing.dropna(subset=["RM"]) #Option 1
#a.shape
# Note that the original housing dataframe will remain unchanged


# In[27]:


#housing.drop("RM", axis=1).shape # Option 2
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged


# In[28]:


#median = housing["RM"].median() # Compute median for Option 3 - we will do it with sklearn only


# In[29]:


#housing.shape


# In[30]:


housing.describe() # before we started filling missing attributes


# In[31]:


from sklearn.impute import SimpleImputer
#imputer = SimpleImputer(strategy="median")
#imputer.fit(housing)


# In[32]:


#imputer.statistics_


# In[33]:


#X = imputer.transform(housing)


# In[34]:


#housing_tr = pd.DataFrame(X, columns=housing.columns)


# In[35]:


#housing_tr.describe()


# # SkLearn Training

# Primarily, three types of objects
# 
# Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters
# 
# Transformers - transform method takes input and returns output based on the learnings from fit(). It also has a convenience function called fit_transform() which fits and then transforms.
# 
# Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also gives score() function which will evaluate the predictions.

# In[36]:


#housing_tr.head()


# Primarily, two types of feature scaling methods:
# 
# Min-max scaling (Normalization) (value - min)/(max - min) Sklearn provides a class called MinMaxScaler for this
# 
# Standardization (value - mean)/std Sklearn provides a class called StandardScaler 

# # Creating pipeline inside which we will fill NA values & Feature scalling

# In[37]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")), #FILL NA VALUES
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()), # FEATURE SCALLING
])


# In[38]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[39]:


housing_num_tr


# In[40]:


housing_num_tr.shape


# # Selecting a desired model

# In[41]:


#from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[42]:


some_data = housing.iloc[:5]


# In[43]:


some_labels = housing_labels.iloc[:5]


# In[44]:


prepared_data = my_pipeline.transform(some_data)


# In[45]:


model.predict(prepared_data)


# In[46]:


list(some_labels)


# # Evaluate our Model

# In[47]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions) # mean_squared_error(y_train,y_pred)
rmse = np.sqrt(mse)


# In[48]:


rmse


# # Using better evaluation technique - Cross Validation

# In[49]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[50]:


rmse_scores


# In[51]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# In[52]:


print_scores(rmse_scores)


# # Saving the model

# In[53]:


from joblib import dump, load
dump(model, 'boston_uci.joblib')


# # Testing the model on test data
# 

# In[54]:


X_test = start_test_set.drop("MEDV", axis=1)
Y_test = start_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[55]:


final_rmse


# In[56]:


print(final_predictions, list(Y_test))


# # USE this model to predict unseen values

# In[57]:


from joblib import dump, load
import numpy as np
model = load('boston_uci.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)

