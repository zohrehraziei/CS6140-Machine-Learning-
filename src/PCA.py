

"""
Principal Component Analysis (PCA)

Author: Zohreh Raziei - raziei.z@husky.neu.edu
"""

# In[1]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve,auc
import random
from scipy import interp

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns',100)


# ## Extracting dataset 

# ### 1. banknote authentication Data Set

# In[2]:


columns1 = ['variance','skewness','curtosis','entropy','class']


# In[3]:


df1 = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt',header=None,names=columns1)
df1.head()


# In[4]:


df1.shape


# In[5]:


df1['class'].value_counts()


# ### 2. Diabetic dataset
# 
# link to the [dataset](https://www.kaggle.com/edubrq/diabetes#diabetes.csv)

# In[6]:


df2 = pd.read_csv('diabetes.csv')
df2.head()


# In[7]:


df2['Outcome'].value_counts()


# In[8]:


df2.shape


# ### 3. Breast Cancer Wisconsin (Diagnostic) Data Set

# In[9]:


columns3 = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']


# In[10]:


df3 = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None,names=columns3)
df3 = df3.drop(['id'],axis=1)

class_dict_3 = {'B' : 0, 'M' : 1}
df3['class'] = df3['diagnosis'].replace(class_dict_3)
df3.drop(['diagnosis'],axis=1,inplace=True)

df3.head()


# In[11]:


df3.shape


# #  Modeling
# 
# ## 1. Only with Logistic Regression

# 
# ### Pseudo code
# 
# ```Python
# X = df1.drop(['class'],axis=1).values
# y = df1['class'].values.reshape(-1,1)
# 
# 
# m = len(y)
# print((np.ones((m,1)).shape,X.shape))
# X = np.hstack((np.ones((m,1)),X))
# n = np.size(X,1)
# params = np.zeros((n,1))
# 
# iterations = 1500
# learning_rate = 0.03
# 
# initial_cost = compute_cost(X, y, params)
# 
# #print("Initial Cost is: {} \n".format(initial_cost))
# 
# (cost_history, params_optimal) = gradient_descent(X, y, params, learning_rate, iterations)
# 
# print("Optimal Parameters are: \n", params_optimal, "\n")
# 
# ```

# In[12]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[13]:


def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost.sum()


# In[14]:


def gradient_descent(X, y, params, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros((iterations,1))
    for i in range(iterations):
        params = params - (learning_rate/m) * X.T @ (sigmoid(X @ params) - y)
        cost_history[i] = compute_cost(X, y, params)

    return (cost_history, params)


# In[15]:


def predict(X, params):
    return np.round(sigmoid(X @ params))


# In[16]:


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# In[17]:


iterations = 1000
learning_rate = 0.01


# In[18]:


def LG(train_set, test_set):
    
    train_set = np.array(train_set)
    test_set = np.array(test_set)
    
    X_train = train_set[:,:-1]
    y_train = train_set[:,-1:]
    m = len(y_train)
    
    X_train = np.hstack((np.ones((m,1)),np.array(X_train)))
    n = np.size(X_train,1)
    params = np.zeros((n,1))
    
    (cost_history, params_optimal) = gradient_descent(X_train, y_train, params, learning_rate, iterations)
    
    X_test = test_set[:,:-1]
    y_test = test_set[:,-1:]
    m = len(y_test)
    
    X_test = np.hstack((np.ones((m,1)),X_test))
    
    fpr, tpr, _ = roc_curve(y_test,sigmoid(X_test @ params_optimal).reshape(-1))

    
    return fpr, tpr, auc(fpr, tpr)
    #return roc_auc_score(y_test,predict(X_test,params_optimal))


# In[19]:


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds,title):
    # get the list of dataset after k-fold valiation
    folds = cross_validation_split(dataset, n_folds)
    
    scores = list()
    tprs = []
    fprs = []
        
    mean_fpr = np.linspace(0,1,100)
    
    for i,fold in enumerate(folds):
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)

        fpr, tpr, auc_score = LG(train_set, test_set)
        tprs.append(interp(mean_fpr, fpr, tpr))     
        scores.append(auc_score)
        fprs.append(fpr)


    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    
    mean_tpr[0] = 0
    mean_fpr[0] = 0
    
    plt.plot(mean_fpr, mean_tpr, color='r',
            lw=2, alpha=1,label=r'Mean ROC (AUC = %0.5f)' % (mean_auc))
    plt.plot([0, 1], ls="--",label='Random chances')
   # plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    print(title)
    print('Area under ROC ',mean_auc,'\n\n')
    
    
    return None


# ### 1.banknote authentication Data

# In[20]:


evaluate_algorithm(df1.values.tolist(),LG,10,'With Logistic Regression - banknote authentication Data')


# ### 2. Diabetic dataset

# In[21]:


evaluate_algorithm(df2.values.tolist(),LG,10,'With Logistic Regression - Diabetic dataset')


# ### 3. Breast Cancer Wisconsin (Diagnostic) Data Set

# In[22]:


evaluate_algorithm(df3.values.tolist(),LG,10,'With Logistic Regression - Breast Cancer Wisconsin Data')


# ## 2. Logostic Regression with z-score normalisation

# ### 1. banknote authentication Data

# In[23]:


cols = df1.columns.tolist()[:-1]
df1_z_norm = df1.copy()

for col in cols:
    df1_z_norm[col] = (df1[col] - df1[col].mean())/df1[col].std(ddof=0)


# In[24]:


evaluate_algorithm(df1_z_norm.values.tolist(),LG,10,'Logostic Regression with z-score normalisation - banknote authentication Data')


# ### 2. Diabetic dataset

# In[25]:


cols = df2.columns.tolist()[:-1]
df2_z_norm = df2.copy()

for col in cols:
    df2_z_norm[col] = (df2[col] - df2[col].mean())/df2[col].std(ddof=0)


# In[26]:


evaluate_algorithm(df2_z_norm.values.tolist(),LG,10, 'Logostic Regression with z-score normalisation - Diabetic dataset')


# ### 3. Breast Cancer Wisconsin (Diagnostic) Data Set

# In[27]:


cols = df3.columns.tolist()[:-1]
df3_z_norm = df3.copy()

for col in cols:
    df3_z_norm[col] = (df3[col] - df3[col].mean())/df3[col].std(ddof=0)


# In[28]:


evaluate_algorithm(df3_z_norm.values.tolist(),LG,10,'Logostic Regression with z-score normalisation - Breast Cancer Wisconsin Data ')


# # Dataset with z-score normalisation and PCA 

# In[29]:


def PCA(feature,retained_variance):
    # calculate covariance matrix
    covariance_matrix = np.cov(feature, rowvar=False)
    # Calculate eigen value and Eigen vector of the covariance matrix
    eigenvalues, eigenvectors, = np.linalg.eig(covariance_matrix)
    # calculate average cumulative significance of the eigenvalue
    significance = [np.abs(i)/np.sum(eigenvalues) for i in eigenvalues]
    # make a set of eigen values and engine vector
    together = zip(eigenvalues, eigenvectors)
    # sorted the list based on the eigen value
    together = sorted(together, key=lambda t: t[0], reverse=True)
    # sorted values saved in the eigen value and eigen vector variable
    eigenvalues[:], eigenvectors[:] = zip(*together)
    # get the minimum index in which significance value is reaching 99% variance
    index_t = [np.cumsum(significance) <= retained_variance][0]
    # take the n_component value based on that index value
    n_component = np.where(index_t == False)[0][0]
        
    #n_component = 10
    # take upto first n_component in eigenvector
    principal_components = eigenvectors[:n_component]
    # make the projection matrix 
    projections = feature.dot(principal_components.T)

    return projections


# ### 1. banknote authentication Data

# In[30]:


cols = df1.columns.tolist()[:-1]
df1_z_norm = df1.copy()

for col in cols:
    df1_z_norm[col] = (df1[col] - df1[col].mean(axis=0))/df1[col].std(ddof=0,axis=0)


# In[31]:


feature = PCA(df1_z_norm.iloc[:,:-1].values,0.99)
new_df = np.hstack((feature,df1_z_norm.iloc[:,-1].values.reshape(-1,1)))


# In[32]:


evaluate_algorithm(new_df.tolist(),LG,10,'PCA with z-score normalization - banknote authentication data')


# ### 2. Diabetic dataset

# In[33]:


cols = df2.columns.tolist()[:-1]
df2_z_norm = df2.copy()

for col in cols:
    df2_z_norm[col] = (df2[col] - df2[col].mean(axis=0))/df2[col].std(ddof=0,axis=0)


# In[34]:


feature = PCA(df2_z_norm.iloc[:,:-1].values,0.99)
new_df = np.hstack((feature,df2_z_norm.iloc[:,-1].values.reshape(-1,1)))


# In[35]:


evaluate_algorithm(new_df.tolist(),LG,10,'PCA with z-score normalization -  Diabetic data')


# ### 3. Breast Cancer Wisconsin (Diagnostic) Data Set

# In[36]:


cols = df3.columns.tolist()[:-1]
df3_z_norm = df3.copy()

for col in cols:
    df3_z_norm[col] = (df3[col] - df3[col].mean(axis=0))/df3[col].std(ddof=0,axis=0)


# In[37]:


feature = PCA(df3_z_norm.iloc[:,:-1].values,0.99)
new_df = np.hstack((feature,df3_z_norm.iloc[:,-1].values.reshape(-1,1)))


# In[38]:


evaluate_algorithm(new_df.tolist(),LG,10, 'PCA with z-score normalization -  Breast Cancer data')


# # Dataset with zero mean normalisation and PCA 

# ### 1. banknote authentication Data

# In[39]:


cols = df1.columns.tolist()[:-1]
df1_zero_norm = df1.copy()

for col in cols:
    df1_zero_norm[col] = (df1[col] - df1[col].min())/(df1[col].max() - df1[col].min())


# In[40]:


feature = PCA(df1_zero_norm.iloc[:,:-1].values,0.99)
new_df = np.hstack((feature,df1_zero_norm.iloc[:,-1].values.reshape(-1,1)))


# In[41]:


evaluate_algorithm(new_df.tolist(),LG,10,'PCA with zero mean normalization -  banknote authentication Data')


# ### 2. Diabetic dataset

# In[42]:


cols = df2.columns.tolist()[:-1]
df2_zero_norm = df2.copy()

for col in cols:
    df2_zero_norm[col] = (df2[col] - df2[col].min())/(df2[col].max() - df2[col].min())


# In[43]:


feature = PCA(df2_zero_norm.iloc[:,:-1].values,0.99)
new_df = np.hstack((feature,df2_zero_norm.iloc[:,-1].values.reshape(-1,1)))


# In[44]:


evaluate_algorithm(new_df.tolist(),LG,10,'PCA with zero mean normalization -  Diabetic Data')


# ### 3. Breast Cancer Wisconsin (Diagnostic) Data Set

# In[45]:


cols = df3.columns.tolist()[:-1]
df3_zero_norm = df3.copy()

for col in cols:
    df3_zero_norm[col] = (df3[col] - df3[col].min())/(df3[col].max() - df3[col].min())


# In[46]:


feature = PCA(df3_zero_norm.iloc[:,:-1].values,0.99)
new_df = np.hstack((feature,df3_z_norm.iloc[:,-1].values.reshape(-1,1)))


# In[47]:


evaluate_algorithm(new_df.tolist(),LG,10, 'PCA with zero mean normalization -  Breast Cancer Wisconsin Data')


# In[ ]:





# ## Finish

# In[48]:


#Plotting the Cumulative Summation of the Explained Variance
#plt.figure()
#plt.plot(np.cumsum(significance),color='red')
#plt.axvline(x=n_component)
#plt.xlabel('Number of Components')
#plt.ylabel('Variance (%)') #for each component
#plt.title('Pulsar Dataset Explained Variance')
#plt.show()


# In[ ]:




