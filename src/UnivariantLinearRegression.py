
"""

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



# load the dataset

df = pd.read_csv('ex1data1.txt',names = ['Population','Profit'])
df.head()



#Plot the data with scaled units

plt.scatter(df['Population'],df['Profit'],c='r')
plt.xlabel('Population/(10,000)')
plt.ylabel('Profit/(10,000)')
plt.title('Original Data')
plt.show()





# making data
X = df['Population'].values
y = df['Profit'].values




X = X.reshape(-1,1)
X.shape



# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X.reshape((-1,1)),axis=1)



# Compute Cost function
def computeCost(X, y, theta):
    m  = y.size
    h = np.dot(X,theta)
    J = (np.sum(np.square(h-y))) / (2 * m)
    return J



# Calucating cost function and theta


def gradientDescent(X, y, theta, alpha, num_iters):
    

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


final_theta,loss = gradientDescent(X_, y, theta, 0.01, 100)
print('loss :',loss)
print('final theta :',final_theta)



x_line = np.arange(1,23).reshape(-1,1)
y_line = final_theta[0] + x_line * final_theta[1]



def linefitline(x):
    return final_theta[0] + final_theta[1] * x
y_pred = linefitline(X)



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




# load the dataset

df = pd.read_csv('ex1data1.txt',names = ['Population','Profit'])
df.head()




# making data
X = df['Population'].values
y = df['Profit'].values



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




a1,b1,c1 = line_total_least_squares(X,y)




x_line = np.arange(1,23).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)




def linefitline(x):
    return -1*(c1/b1) + -1*(a1/b1) * x
y_pred = linefitline(X)




print('R2 score For Dataset 1 (MSD): ',r2_score(y,y_pred))




plt.scatter(df['Population'],df['Profit'],c='r',label='given')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 1')
plt.show()


# ## 3. Closed Form Solution



X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)




# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))



coeffs




x_line = np.arange(1,23).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]



def linefitline(x):
    return coeffs[1][0] + coeffs[0][0] * x
y_pred = linefitline(X)


print('R2 score For Dataset 1 (CFS) : ',r2_score(y,y_pred))




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




plt.scatter(X,y)
plt.show()


# ## 1. Minimizing Sum of squared Error



X = X.reshape(-1,1)
X.shape



# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X.reshape((-1,1)),axis=1)




final_theta,loss = gradientDescent(X_, y, theta, 0.6, 200)
print('loss :',loss)
print('final theta :',final_theta)




x_line = np.arange(-5,5).reshape(-1,1)
y_line = final_theta[0] + x_line * final_theta[1]




def linefitline(x):
    return final_theta[0] + final_theta[1] * x
y_pred = linefitline(X)



print('R2 score For Dataset 2 (MSE) : ',r2_score(y,y_pred))



plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimizing Sum of squared Error : Dataset 2')
plt.show()


# ## 2. Minimising sum of distance


a1,b1,c1 = line_total_least_squares(X.reshape(-1),y)





x_line = np.arange(-5,5).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)




def linefitline(x):
    return -1*(c1/b1) + -1*(a1/b1) * x
y_pred = linefitline(X)




print('R2 score For Dataset 2 (MSD): ',r2_score(y,y_pred))




plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 2')
plt.show()


# ## 3. Closed Form Solution



X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)



# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))



print(coeffs)




def linefitline(x):
    return coeffs[1][0] + coeffs[0][0] * x
y_pred = linefitline(X)



print('R2 score For Dataset 2 (CFS) : ',r2_score(y,y_pred))




x_line = np.arange(-3,5).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]




plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Close Form Solution : Dataset 2')
plt.show()






# # <center>Dataset : 3</center>

# In[50]:


X, y = make_regression(n_samples=200,n_features=1,noise=20)




plt.scatter(X,y)
plt.show()


# ## 1. Minimizing Sum of squared Error



X = X.reshape(-1,1)
X.shape




# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X.reshape((-1,1)),axis=1)



final_theta,loss = gradientDescent(X_, y, theta, 0.5, 100)
print('loss :',loss)
print('final theta :',final_theta)




x_line = np.arange(-5,5).reshape(-1,1)
y_line = final_theta[0] + x_line * final_theta[1]



def linefitline(x):
    return final_theta[0] + final_theta[1] * x
y_pred = linefitline(X)




print('R2 score For Dataset 3 (MSE) : ',r2_score(y,y_pred))




plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimizing Sum of squared Error : Dataset 3')
plt.show()


# ## 2. Minimising sum of distance



a1,b1,c1 = line_total_least_squares(X.reshape(-1),y)




x_line = np.arange(-5,5).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)




def linefitline(x):
    return -1*(c1/b1) + -1*(a1/b1) * x
y_pred = linefitline(X)




print('R2 score For Dataset 3 (MSD): ',r2_score(y,y_pred))




plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 3')
plt.show()


# ## 3. Closed Form Solution


X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)



# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))




print(coeffs)




def linefitline(x):
    return coeffs[1][0] + coeffs[0][0] * x
y_pred = linefitline(X)



print('R2 score For Dataset 3 (CFS) : ',r2_score(y,y_pred))




x_line = np.arange(-3,5).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]




plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Close Form Solution : Dataset 3')
plt.show()







# # <center>Dataset : 4</center>

# In[71]:


X, y = make_regression(n_samples=300,n_features=1,noise=15)




plt.scatter(X,y)
plt.show()


# ## 1. Minimizing Sum of squared Error


X = X.reshape(-1,1)
X.shape




# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X.reshape((-1,1)),axis=1)



final_theta,loss = gradientDescent(X_, y, theta, 0.5, 100)
print('loss :',loss)
print('final theta :',final_theta)




x_line = np.arange(-5,5).reshape(-1,1)
y_line = final_theta[0] + x_line * final_theta[1]




def linefitline(x):
    return final_theta[0] + final_theta[1] * x
y_pred = linefitline(X)


print('R2 score For Dataset 4 (MSE) : ',r2_score(y,y_pred))




plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimizing Sum of squared Error : Dataset 4')
plt.show()


# ## 2. Minimising sum of distance


a1,b1,c1 = line_total_least_squares(X.reshape(-1),y)




x_line = np.arange(-5,5).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)



def linefitline(x):
    return -1*(c1/b1) + -1*(a1/b1) * x
y_pred = linefitline(X)




print('R2 score For Dataset 4 (MSD): ',r2_score(y,y_pred))




plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 4')
plt.show()


# ## 3. Closed Form Solution



X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)




# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))




print(coeffs)




def linefitline(x):
    return coeffs[1][0] + coeffs[0][0] * x
y_pred = linefitline(X)




print('R2 score For Dataset 4 (CFS) : ',r2_score(y,y_pred))




x_line = np.arange(-3,5).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]




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



plt.scatter(X,y)
plt.show()


# ## 1. Minimizing Sum of squared Error



X = X.reshape(-1,1)
X.shape




# No of examples 
m = X.shape[0]

# No of features 
n = X.shape[1] + 1

#Initialize theta
theta = np.zeros((n,1))
#theta = theta.reshape((len(theta), 1))

#adding extra feature theta0 as 1 
X_ = np.append(np.ones((m, 1)), X.reshape((-1,1)),axis=1)



final_theta,loss = gradientDescent(X_, y, theta, 0.5, 500)
print('loss :',loss)
print('final theta :',final_theta)




x_line = np.arange(-5,5).reshape(-1,1)
y_line = final_theta[0] + x_line * final_theta[1]




def linefitline(x):
    return final_theta[0] + final_theta[1] * x
y_pred = linefitline(X)



print('R2 score For Dataset 5 (MSE) : ',r2_score(y,y_pred))



plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimizing Sum of squared Error : Dataset 5')
plt.show()


# ## 2. Minimising sum of distance



a1,b1,c1 = line_total_least_squares(X.reshape(-1),y)




x_line = np.arange(-5,5).reshape(-1,1)
y_line = (-1*(c1/b1)) + x_line * -1*(a1/b1)




def linefitline(x):
    return -1*(c1/b1) + -1*(a1/b1) * x
y_pred = linefitline(X)




print('R2 score For Dataset 5 (MSD): ',r2_score(y,y_pred))




plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Minimising sum of distance : Dataset 5')
plt.show()


# ## 3. Closed Form Solution



X_ = np.append(X.reshape((-1,1)),np.ones((m, 1)) ,axis=1)



# calculate coefficients using closed-form solution
from numpy.linalg import inv
coeffs = inv(X_.transpose().dot(X_)).dot(X_.transpose()).dot(y.reshape(-1,1))




x_line = np.arange(-3,5).reshape(-1,1)
y_line = (coeffs[1][0]) + x_line * coeffs[0][0]




print(coeffs)




def linefitline(x):
    return coeffs[1][0] + coeffs[0][0] * x
y_pred = linefitline(X)




print('R2 score For Dataset 5 (CFS) : ',r2_score(y,y_pred))




plt.scatter(X,y,c='r',label='Original')
plt.xlabel('X')
plt.ylabel('y')
plt.plot(x_line,y_line,label='predicted')
plt.legend()
plt.title('Close Form Solution : Dataset 5')
plt.show()









