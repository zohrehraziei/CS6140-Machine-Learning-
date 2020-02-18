# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:23:54 2020

MultiVariant linear regression problem of mapping R_d to R, with two diffrent objective functions

Author: Zohreh Raziei - raziei.z@husky.neu.edu

"""

#!/usr/bin/env python
# coding: utf-8

# # <center>Dataset : 1 </center>

# ## 1. Minimizing Sum of squared Error

# In[1]:


#Install dependecy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use, cm
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load the dataset

df = pd.read_csv('housing.data.txt',header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()


# In[3]:


# making data
X = X = df.iloc[:,[0,5]].values
y = df['MEDV'].values


# In[4]:


X.shape


# In[5]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X,axis=1)


# In[6]:


# Compute Cost function
def computeCost(X, y, theta):
    m  = y.size
    h = np.dot(X,theta)
    J = (np.sum(np.square(h-y))) / (2 * m)
    return J


# In[7]:


# Calucating cost function and theta


def gradientDescent(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """
    final_theta = theta
    # Initialize some useful values
    J_history = [np.inf]
    m = y.size  # number of training examples

    for i in range(100):
        h = np.dot(X, theta)
        theta = theta - ((alpha / m) * (np.dot(X.T,(h - y.reshape(-1,1))))) 
        # Save the cost J in every iteration
        
        current_cost = computeCost(X, y, theta)
        while J_history[len(J_history)-1] > current_cost:
            J_history.append(computeCost(X, y, theta))
            final_theta = theta
    return final_theta,J_history[len(J_history)-1]


# In[8]:


X_.shape


# In[9]:


y.shape


# In[10]:


theta.shape


# In[11]:


final_theta,loss = gradientDescent(X_, y, theta, 0.01, 10000)
print('loss :',loss)
print('final theta :',final_theta)


# In[12]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2]
print('R2 score For Dataset 1 (MSE): ',r2_score(y,y_pred))


# ## 2. Minimising sum of distance

# In[13]:


import numpy.linalg as la
def tls(X,y):
    
    if X.ndim is 1: 
        n = 1 # the number of variable of X
        X = X.reshape(len(X),1)
    else:
        n = np.array(X).shape[1] 
    
    Z = np.vstack((X.T,y)).T
    U, s, Vt = la.svd(Z, full_matrices=True)
    
    V = Vt.T
    Vxy = V[:n,n:]
    Vyy = V[n:,n:]
    a_tls = - Vxy  / Vyy # total least squares soln
    
    Xtyt = - Z.dot(V[:,n:]).dot(V[:,n:].T)
    Xt = Xtyt[:,:n] # X error
    y_tls = (X+Xt).dot(a_tls)
    fro_norm = la.norm(Xtyt, 'fro')
    
    return a_tls


# In[14]:


final_theta = tls(X_,y)


# In[15]:


final_theta


# In[16]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2]
print('R2 score For Dataset 1 (MSD): ',r2_score(y,y_pred))


# ## 3. Closed Form Solution

# In[17]:


X.shape


# In[18]:


X_ = np.append(X,np.ones((m, 1)) ,axis=1)


# In[19]:


X_.shape


# In[20]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[21]:


coeffs


# In[22]:


y_pred = coeffs[0]* X_[:,0] + coeffs[1] * X_[:,1] + coeffs[2] * X_[:,2]
print('R2 score For Dataset 1 (CFS) : ',r2_score(y,y_pred))


# # <center>Dataset : 2</center>

# In[23]:


X, y = make_regression(n_samples=100,n_features=3,noise=10)


# ## 1. Minimizing Sum of squared Error

# In[24]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X,axis=1)


# In[25]:


final_theta,loss = gradientDescent(X_, y, theta, 0.6, 200)
print('loss :',loss)
print('final theta :',final_theta)


# In[26]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2] + final_theta[3] * X_[:,3]
print('R2 score For Dataset 2 (MSE): ',r2_score(y,y_pred))


# ## 2. Minimising sum of distance

# In[27]:


final_theta = tls(X_,y)


# In[28]:


final_theta


# In[29]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2] + final_theta[3] * X_[:,3]
print('R2 score For Dataset 2 (MSD): ',r2_score(y,y_pred))


# ## 3. Closed Form Solution

# In[30]:


X_ = np.append(X,np.ones((m, 1)) ,axis=1)


# In[31]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[32]:


coeffs


# In[33]:


y_pred = coeffs[0]* X_[:,0] + coeffs[1] * X_[:,1] + coeffs[2] * X_[:,2] + coeffs[3] * X_[:,3]
print('R2 score For Dataset 2 (CFS) : ',r2_score(y,y_pred))


# # <center>Dataset : 3</center>

# In[34]:


X, y = make_regression(n_samples=200,n_features=4,noise=20)


# ## 1. Minimizing Sum of squared Error

# In[35]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X,axis=1)


# In[36]:


final_theta,loss = gradientDescent(X_, y, theta, 0.5, 100)
print('loss :',loss)
print('final theta :',final_theta)


# In[37]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2] + final_theta[3] * X_[:,3] + final_theta[4] * X_[:,4]
print('R2 score For Dataset 3 (MSE): ',r2_score(y,y_pred))


# ## 2. Minimising sum of distance

# In[38]:


final_theta = tls(X_,y)


# In[39]:


final_theta


# In[40]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2] + final_theta[3] * X_[:,3] + final_theta[4] * X_[:,4]
print('R2 score For Dataset 3 (MSD): ',r2_score(y,y_pred))


# ## 3. Closed Form Solution

# In[41]:


X_ = np.append(X,np.ones((m, 1)) ,axis=1)


# In[42]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[43]:


coeffs


# In[44]:


y_pred = coeffs[0]* X_[:,0] + coeffs[1] * X_[:,1] + coeffs[2] * X_[:,2] + coeffs[3] * X_[:,3] + coeffs[4] * X_[:,4]
print('R2 score For Dataset 3 (CFS) : ',r2_score(y,y_pred))


# # <center>Dataset : 4</center>

# In[45]:


X, y = make_regression(n_samples=300,n_features=3,noise=15)


# ## 1. Minimizing Sum of squared Error

# In[46]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X,axis=1)


# In[47]:


final_theta,loss = gradientDescent(X_, y, theta, 0.5, 100)
print('loss :',loss)
print('final theta :',final_theta)


# In[48]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2] + final_theta[3] * X_[:,3]


# In[49]:


print('R2 score For Dataset 4 (MSE): ',r2_score(y,y_pred))


# ## 2. Minimising sum of distance

# In[50]:


final_theta = tls(X_,y)


# In[51]:


final_theta


# In[52]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2] + final_theta[3] * X_[:,3]
print('R2 score For Dataset 4 (MSD): ',r2_score(y,y_pred))


# ## 3. Closed Form Solution

# In[53]:


X_ = np.append(X,np.ones((m, 1)) ,axis=1)


# In[54]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[55]:


coeffs


# In[56]:


y_pred = coeffs[0]* X_[:,0] + coeffs[1] * X_[:,1] + coeffs[2] * X_[:,2] + coeffs[3] * X_[:,3] 
print('R2 score For Dataset 4 (CFS) : ',r2_score(y,y_pred))


# # <center>Dataset : 5</center>

# In[57]:


X, y = make_regression(n_samples=500,n_features=5,noise=25)


# ## 1. Minimizing Sum of squared Error

# In[58]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X,axis=1)


# In[59]:


final_theta,loss = gradientDescent(X_, y, theta, 0.5, 500)
print('loss :',loss)
print('final theta :',final_theta)


# In[60]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2] +          final_theta[3] * X_[:,3] + final_theta[4] * X_[:,4] + final_theta[5] * X_[:,5]



# In[61]:


print('R2 score For Dataset 5 (MSE): ',r2_score(y,y_pred))


# ## 2. Minimising sum of distance

# In[62]:


final_theta = tls(X_,y)


# In[63]:


final_theta


# In[64]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2] +          final_theta[3] * X_[:,3] + final_theta[4] * X_[:,4] + final_theta[5] * X_[:,5]



print('R2 score For Dataset 5 (MSD): ',r2_score(y,y_pred))


# ## 3. Closed Form Solution

# In[65]:


X_ = np.append(X,np.ones((m, 1)) ,axis=1)


# In[66]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[67]:


coeffs


# In[68]:


y_pred = coeffs[0]* X_[:,0] + coeffs[1] * X_[:,1] + coeffs[2] * X_[:,2] +          coeffs[3] * X_[:,3] + coeffs[4] * X_[:,4] + coeffs[5] * X_[:,5]

print('R2 score For Dataset 5 (CFS) : ',r2_score(y,y_pred))


# In[ ]:





# In[ ]:




