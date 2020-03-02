"""

MultiVariant linear regression problem of mapping R_d to R, with two diffrent objective functions

Author: Zohreh Raziei - raziei.z@husky.neu.edu

"""


####Dataset : 1 #####

# 1. Minimizing Sum of squared Error

# In[1]:


#Install dependecy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use, cm
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
import sys

get_ipython().run_line_magic('matplotlib', 'inline')




# load the dataset

df = pd.read_csv('housing.data.txt',header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS', 'RAD', 
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()





# making data
X = X = df.iloc[:,[0,5]].values
y = df['MEDV'].values





X.shape




# No of examples 
m = X.shape[0]
y.reshape(m,1)

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.ones((n,1))*0.001
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X = np.append(np.ones((m, 1)), X,axis=1)




# Compute Cost function
def computeCost(X, y, theta):
    m  = y.size
    h = np.dot(X,theta)
    #print(theta)
    J = (np.sum(np.square(h-y))) / (2 * m)
    return J



# Calucating cost function and theta


def gradientDescent(X, y, theta, alpha, num_iters):

    final_theta = theta
    # Initialize some useful values
    J_history = [np.inf]

    for i in range(num_iters):
        tsize = np.dot(theta.T, theta)
        h = tsize * X
        reg = np.dot(X, theta) - y
        reg = reg[0]
        xreg = np.dot(X.T, reg)
        reg2 = np.dot(reg.T, reg)
        wreg2 = reg2 * theta
        t1 = 2 * tsize * xreg
        t2 = 2*wreg2
        if tsize == 0:
            tsize = 0.0001
        D = (t1.T - t2)/tsize
        #print (tsize)
        #print(t1.T)
        #print(t2)
        #print(D)
        #sys.exit()
        theta = theta - (alpha / m) * D
        
        
        # Save the cost J in every iteration
        
        current_cost = computeCost(X, y, theta)
        print ('iter:',i,' - cost: ',current_cost)
        final_theta = theta
        if J_history[len(J_history)-1] > current_cost:
            J_history.append(computeCost(X, y, theta))
        else:
            break
    return final_theta,J_history[len(J_history)-1]





X.shape

y.shape

theta.shape





final_theta,loss = gradientDescent(X, y, theta, 0.00001, 10000)
print('loss :',loss)
print('final theta :',final_theta)




y_pred = np.dot(X,final_theta)
print('R2 score For Dataset 1 (MSE): ',r2_score(y,y_pred))

