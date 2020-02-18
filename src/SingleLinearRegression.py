# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:23:54 2020

linear regression problem of mapping R_d to R, with two diffrent objective functions

Author: Zohreh Raziei - raziei.z@husky.neu.edu

"""


#1. Minimizing Sum of squared Error

# In[1]:


#Install dependency

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use, cm
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load the dataset

df = pd.read_csv('ex1data1.txt',names = ['Population','Profit'])
df.head()


# In[3]:


#Plot the data with scaled units

plt.scatter(df['Population'],df['Profit'],c='r')
plt.xlabel('Population/(10,000)')
plt.ylabel('Profit/(10,000)')
plt.title('Original Data')
plt.show()


# In[4]:


# making data
X = df['Population'].values
y = df['Profit'].values


# In[5]:


X = X.reshape(-1,1)
X.shape


# In[6]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X.reshape((-1,1)),axis=1)


# In[7]:


# Compute Cost function
def computeCost(X, y, theta):
    m  = y.size
    h = np.dot(X,theta)
    J = (np.sum(np.square(h-y))) / (2 * m)
    return J


# In[8]:


# Calucating cost function and theta


def gradientDescent(X, y, theta, alpha, num_iters):
    """
     Performs gradient descent to learn theta
       theta = gradientDescent(x, y, theta, alpha, num_iters) updates theta by
       taking num_iters gradient steps with learning rate alpha
    """

    # Initialize some useful values
    J_history = []
    m = y.size  # number of training examples

    for i in range(num_iters):
        h = np.dot(X, theta)
        theta = theta - ((alpha / m) * (np.dot(X.T,(h - y.reshape(-1,1))))) 
        # Save the cost J in every iteration
        J_history.append(computeCost(X, y, theta)) 
        break
    return theta,J_history[len(J_history)-1]


# In[9]:


final_theta,loss = gradientDescent(X_, y, theta, 0.01, 100)
print('loss :',loss)
print('final theta :',final_theta)


# In[10]:


x_line = np.arange(1,23).reshape(-1,1)
y_line = final_theta[0] + x_line * final_theta[1]


# In[11]:


def linefitline(x):
    return final_theta[0] + final_theta[1] * x
y_pred = linefitline(X)


# In[12]:


print('R2 score For Dataset 1 (MSE): ',r2_score(y,y_pred))


# In[13]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimizing Sum of squared Error : Dataset 1')
plt.show()


# ## 2. Minimising sum of distance

# In[14]:


# load the dataset

df = pd.read_csv('ex1data1.txt',names = ['Population','Profit'])
df.head()


# In[15]:


# making data
X = df['Population'].values
y = df['Profit'].values


# In[16]:


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
    X_t = np.transpose(X)
    
    # Finding A_T_A and it's Find smallest eigenvalue::
    prd = np.dot(X,X_t)
    W,V = np.linalg.eig(prd)
    small_eig_index = W.argmin()
    a,b = V[:,small_eig_index] 
    
    # Compute C:
    c = (-1*a*x_) + (-1*b*y_)
    
    return a,b,c


# In[17]:


a1,b1,c1 = line_total_least_squares(X,y)


# In[18]:


x_line = np.arange(1,23).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)


# In[19]:


def linefitline(x):
    return -1*(c1/b1) + -1*(a1/b1) * x
y_pred = linefitline(X)


# In[20]:


print('R2 score For Dataset 1 (MSD): ',r2_score(y,y_pred))


# In[21]:


plt.scatter(df['Population'],df['Profit'],c='r',label='given')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 1')
plt.show()


# ## 3. Closed Form Solution

# In[22]:


X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)


# In[23]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[24]:


coeffs


# In[25]:


x_line = np.arange(1,23).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]


# In[26]:


def linefitline(x):
    return coeffs[1][0] + coeffs[0][0] * x
y_pred = linefitline(X)


# In[27]:


print('R2 score For Dataset 1 (CFS) : ',r2_score(y,y_pred))


# In[28]:


plt.scatter(df['Population'],df['Profit'],c='r',label='given')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Close Form Solution : Dataset 1')
plt.show()


# # <center>Dataset : 2</center>

# In[29]:


X, y = make_regression(n_samples=100,n_features=1,noise=10)


# In[30]:


plt.scatter(X,y)
plt.show()


# ## 1. Minimizing Sum of squared Error

# In[31]:


X = X.reshape(-1,1)
X.shape


# In[32]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X.reshape((-1,1)),axis=1)


# In[33]:


final_theta,loss = gradientDescent(X_, y, theta, 0.6, 200)
print('loss :',loss)
print('final theta :',final_theta)


# In[34]:


x_line = np.arange(-5,5).reshape(-1,1)
y_line = final_theta[0] + x_line * final_theta[1]


# In[35]:


def linefitline(x):
    return final_theta[0] + final_theta[1] * x
y_pred = linefitline(X)


# In[36]:


print('R2 score For Dataset 2 (MSE) : ',r2_score(y,y_pred))


# In[37]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimizing Sum of squared Error : Dataset 2')
plt.show()


# ## 2. Minimising sum of distance

# In[38]:


a1,b1,c1 = line_total_least_squares(X.reshape(-1),y)


# In[39]:


x_line = np.arange(-5,5).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)


# In[40]:


def linefitline(x):
    return -1*(c1/b1) + -1*(a1/b1) * x
y_pred = linefitline(X)


# In[41]:


print('R2 score For Dataset 2 (MSD): ',r2_score(y,y_pred))


# In[42]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 2')
plt.show()


# ## 3. Closed Form Solution

# In[43]:


X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)


# In[44]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[45]:


print(coeffs)


# In[46]:


def linefitline(x):
    return coeffs[1][0] + coeffs[0][0] * x
y_pred = linefitline(X)


# In[47]:


print('R2 score For Dataset 2 (CFS) : ',r2_score(y,y_pred))


# In[48]:


x_line = np.arange(-3,5).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]


# In[49]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Close Form Solution : Dataset 2')
plt.show()


# In[ ]:





# # <center>Dataset : 3</center>

# In[50]:


X, y = make_regression(n_samples=200,n_features=1,noise=20)


# In[51]:


plt.scatter(X,y)
plt.show()


# ## 1. Minimizing Sum of squared Error

# In[52]:


X = X.reshape(-1,1)
X.shape


# In[53]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X.reshape((-1,1)),axis=1)


# In[54]:


final_theta,loss = gradientDescent(X_, y, theta, 0.5, 100)
print('loss :',loss)
print('final theta :',final_theta)


# In[55]:


x_line = np.arange(-5,5).reshape(-1,1)
y_line = final_theta[0] + x_line * final_theta[1]


# In[56]:


def linefitline(x):
    return final_theta[0] + final_theta[1] * x
y_pred = linefitline(X)


# In[57]:


print('R2 score For Dataset 3 (MSE) : ',r2_score(y,y_pred))


# In[58]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimizing Sum of squared Error : Dataset 3')
plt.show()


# ## 2. Minimising sum of distance

# In[59]:


a1,b1,c1 = line_total_least_squares(X.reshape(-1),y)


# In[60]:


x_line = np.arange(-5,5).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)


# In[61]:


def linefitline(x):
    return -1*(c1/b1) + -1*(a1/b1) * x
y_pred = linefitline(X)


# In[62]:


print('R2 score For Dataset 3 (MSD): ',r2_score(y,y_pred))


# In[63]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 3')
plt.show()


# ## 3. Closed Form Solution

# In[64]:


X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)


# In[65]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[66]:


print(coeffs)


# In[67]:


def linefitline(x):
    return coeffs[1][0] + coeffs[0][0] * x
y_pred = linefitline(X)


# In[68]:


print('R2 score For Dataset 3 (CFS) : ',r2_score(y,y_pred))


# In[69]:


x_line = np.arange(-3,5).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]


# In[70]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Close Form Solution : Dataset 3')
plt.show()


# In[ ]:





# # <center>Dataset : 4</center>

# In[71]:


X, y = make_regression(n_samples=300,n_features=1,noise=15)


# In[72]:


plt.scatter(X,y)
plt.show()


# ## 1. Minimizing Sum of squared Error

# In[73]:


X = X.reshape(-1,1)
X.shape


# In[74]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X.reshape((-1,1)),axis=1)


# In[75]:


final_theta,loss = gradientDescent(X_, y, theta, 0.5, 100)
print('loss :',loss)
print('final theta :',final_theta)


# In[76]:


x_line = np.arange(-5,5).reshape(-1,1)
y_line = final_theta[0] + x_line * final_theta[1]


# In[77]:


def linefitline(x):
    return final_theta[0] + final_theta[1] * x
y_pred = linefitline(X)


# In[78]:


print('R2 score For Dataset 4 (MSE) : ',r2_score(y,y_pred))


# In[79]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimizing Sum of squared Error : Dataset 4')
plt.show()


# ## 2. Minimising sum of distance

# In[80]:


a1,b1,c1 = line_total_least_squares(X.reshape(-1),y)


# In[81]:


x_line = np.arange(-5,5).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)


# In[82]:


def linefitline(x):
    return -1*(c1/b1) + -1*(a1/b1) * x
y_pred = linefitline(X)


# In[83]:


print('R2 score For Dataset 4 (MSD): ',r2_score(y,y_pred))


# In[84]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 4')
plt.show()


# ## 3. Closed Form Solution

# In[85]:


X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)


# In[86]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[87]:


print(coeffs)


# In[88]:


def linefitline(x):
    return coeffs[1][0] + coeffs[0][0] * x
y_pred = linefitline(X)


# In[89]:


print('R2 score For Dataset 4 (CFS) : ',r2_score(y,y_pred))


# In[90]:


x_line = np.arange(-3,5).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]


# In[91]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Close Form Solution : Dataset 4')
plt.show()


# # <center>Dataset : 5</center>

# In[92]:


X, y = make_regression(n_samples=500,n_features=1,noise=25)


# In[93]:


plt.scatter(X,y)
plt.show()


# ## 1. Minimizing Sum of squared Error

# In[94]:


X = X.reshape(-1,1)
X.shape


# In[95]:


# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X.reshape((-1,1)),axis=1)


# In[96]:


final_theta,loss = gradientDescent(X_, y, theta, 0.5, 500)
print('loss :',loss)
print('final theta :',final_theta)


# In[97]:


x_line = np.arange(-5,5).reshape(-1,1)
y_line = final_theta[0] + x_line * final_theta[1]


# In[98]:


def linefitline(x):
    return final_theta[0] + final_theta[1] * x
y_pred = linefitline(X)


# In[99]:


print('R2 score For Dataset 5 (MSE) : ',r2_score(y,y_pred))


# In[100]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimizing Sum of squared Error : Dataset 5')
plt.show()


# ## 2. Minimising sum of distance

# In[101]:


a1,b1,c1 = line_total_least_squares(X.reshape(-1),y)


# In[102]:


x_line = np.arange(-5,5).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)


# In[103]:


def linefitline(x):
    return -1*(c1/b1) + -1*(a1/b1) * x
y_pred = linefitline(X)


# In[104]:


print('R2 score For Dataset 5 (MSD): ',r2_score(y,y_pred))


# In[105]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 5')
plt.show()


# ## 3. Closed Form Solution

# In[106]:


X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)


# In[107]:


# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))


# In[108]:


x_line = np.arange(-3,5).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]


# In[109]:


print(coeffs)


# In[110]:


def linefitline(x):
    return coeffs[1][0] + coeffs[0][0] * x
y_pred = linefitline(X)


# In[111]:


print('R2 score For Dataset 5 (CFS) : ',r2_score(y,y_pred))


# In[112]:


plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Close Form Solution : Dataset 5')
plt.show()


# In[ ]:





# In[ ]:






