# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:23:54 2020

MultiVariant linear regression problem of mapping R_d to R, with two diffrent objective functions

Author: Zohreh Raziei - raziei.z@husky.neu.edu

"""


# ## 1. Minimizing Sum of squared Error

# In[290]:


#Install dependecy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use, cm
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[291]:


# load the dataset

df = pd.read_csv('housing.data.txt',header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()


# In[292]:


# making data
X = X = df.iloc[:,[0,5]].values
y = df['MEDV'].values


# In[293]:


X.shape


# In[294]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X,axis=1)


# In[295]:


# Compute Cost function
def computeCost(X, y, theta):
    m  = y.size
    h = np.dot(X,theta)
    J = (np.sum(np.square(h-y))) / (2 * m)
    return J


# In[296]:


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


# In[297]:


X_.shape


# In[298]:


y.shape


# In[299]:


theta.shape


# In[300]:


final_theta,loss = gradientDescent(X_, y, theta, 0.01, 10000)
print('loss :',loss)
print('final theta :',final_theta)


# In[301]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2]


# In[302]:


print('R2 score For Dataset 1 (MSE): ',r2_score(y,y_pred))


# ## 2. Minimising sum of distance

# In[198]:


# load the dataset

df = pd.read_csv('ex1data1.txt',names = ['Population','Profit'])
df.head()


# In[199]:


# making data
X = df['Population'].values
y = df['Profit'].values


# In[232]:


# Total Least Squares:
def line_total_least_squares(x,y):
    n = len(x)
    
    x_ = np.sum(x)/n
    y_ = np.sum(y)/n
    
    # Calculate the x~ and y~ 
    x1 = x - x_
    y1 = y - y_
    
    # Create the matrix array
    X = np.vstack((x1, y1))
    print(X.shape)
    X_t = np.transpose(X)
    
    # Finding A_T_A and it's Find smallest eigenvalue::
    prd = np.dot(X,X_t)
    W,V = np.linalg.eig(prd)
    small_eig_index = W.argmin()
    a,b = V[:,small_eig_index] 
    
    # Compute C:
    c = (-1*a*x_) + (-1*b*y_)
    
    return a,b,c


# In[233]:


a1,b1,c1 = line_total_least_squares(X,y)


# In[ ]:


x_line = np.arange(1,23).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)


# In[ ]:


plt.scatter(df['Population'],df['Profit'],c='r',label='given')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 1')
plt.show()


# ## 3. Closed Form Solution

# In[247]:


X_ = np.append(X,np.ones((m, 1)) ,axis=1)


# In[248]:


X_.shape


# In[249]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[250]:


coeffs


# In[ ]:





# # <center>Dataset : 2</center>

# In[303]:


X, y = make_regression(n_samples=100,n_features=3,noise=10)


# ## 1. Minimizing Sum of squared Error

# In[305]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X,axis=1)


# In[306]:


final_theta,loss = gradientDescent(X_, y, theta, 0.6, 200)
print('loss :',loss)
print('final theta :',final_theta)


# In[307]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2] + final_theta[3] * X_[:,3]


# In[308]:


print('R2 score For Dataset 2 (MSE): ',r2_score(y,y_pred))


# ## 2. Minimising sum of distance

# In[ ]:


a1,b1,c1 = line_total_least_squares(X.reshape(-1),y)


# In[ ]:


x_line = np.arange(-5,5).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)


# In[ ]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 2')
plt.show()


# ## 3. Closed Form Solution

# In[ ]:


X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)


# In[ ]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[ ]:


coeffs


# In[ ]:


x_line = np.arange(-3,5).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]


# In[ ]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Close Form Solution : Dataset 2')
plt.show()


# In[ ]:





# # <center>Dataset : 3</center>

# In[309]:


X, y = make_regression(n_samples=200,n_features=4,noise=20)


# ## 1. Minimizing Sum of squared Error

# In[310]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X,axis=1)


# In[311]:


final_theta,loss = gradientDescent(X_, y, theta, 0.5, 100)
print('loss :',loss)
print('final theta :',final_theta)


# In[312]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2] + final_theta[3] * X_[:,3] + final_theta[4] * X_[:,4]


# In[313]:


print('R2 score For Dataset 1 (MSE): ',r2_score(y,y_pred))


# ## 2. Minimising sum of distance

# In[ ]:


a1,b1,c1 = line_total_least_squares(X.reshape(-1),y)


# In[ ]:


x_line = np.arange(-5,5).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)


# In[ ]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 3')
plt.show()


# ## 3. Closed Form Solution

# In[ ]:


X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)


# In[ ]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[ ]:


coeffs


# In[ ]:


x_line = np.arange(-3,5).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]


# In[ ]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Close Form Solution : Dataset 3')
plt.show()


# In[ ]:





# # <center>Dataset : 4</center>

# In[314]:


X, y = make_regression(n_samples=300,n_features=3,noise=15)


# ## 1. Minimizing Sum of squared Error

# In[317]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X,axis=1)


# In[318]:


final_theta,loss = gradientDescent(X_, y, theta, 0.5, 100)
print('loss :',loss)
print('final theta :',final_theta)


# In[320]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2] + final_theta[3] * X_[:,3]


# In[321]:


print('R2 score For Dataset 1 (MSE): ',r2_score(y,y_pred))


# ## 2. Minimising sum of distance

# In[ ]:


a1,b1,c1 = line_total_least_squares(X.reshape(-1),y)


# In[ ]:


x_line = np.arange(-5,5).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)


# In[ ]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 4')
plt.show()


# ## 3. Closed Form Solution

# In[ ]:


X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)


# In[ ]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[ ]:


coeffs


# In[ ]:


x_line = np.arange(-3,5).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]


# In[ ]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Close Form Solution : Dataset 4')
plt.show()


# # <center>Dataset : 5</center>

# In[322]:


X, y = make_regression(n_samples=500,n_features=5,noise=25)


# ## 1. Minimizing Sum of squared Error

# In[323]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X,axis=1)


# In[324]:


final_theta,loss = gradientDescent(X_, y, theta, 0.5, 500)
print('loss :',loss)
print('final theta :',final_theta)


# In[327]:


y_pred = final_theta[0]* X_[:,0] + final_theta[1] * X_[:,1] + final_theta[2] * X_[:,2] +          final_theta[3] * X_[:,3] + final_theta[4] * X_[:,4] + final_theta[5] * X_[:,5]



# In[328]:


print('R2 score For Dataset 1 (MSE): ',r2_score(y,y_pred))


# ## 2. Minimising sum of distance

# In[ ]:


a1,b1,c1 = line_total_least_squares(X.reshape(-1),y)


# In[ ]:


x_line = np.arange(-5,5).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)


# In[ ]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 5')
plt.show()


# ## 3. Closed Form Solution

# In[ ]:


X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)


# In[ ]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[ ]:


coeffs


# In[ ]:


x_line = np.arange(-3,5).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]


# In[ ]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Close Form Solution : Dataset 5')
plt.show()


# In[ ]:





# In[ ]:




