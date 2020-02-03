
"""
Created on Fri Feb 02 2020

Curse of Dimentionality

Author: Zohreh Raziei - raziei.z@husky.neu.edu
"""

import numpy as np
import matplotlib.pyplot as plt
#import math as mth
from tqdm import tqdm

# define value of n
ns=[100,1000]

# define value of k
#k=range(1,101)
k = [1,25,50,100]


'''
I declare a function which calculate the distance between two vector by calculating the square root of the sum of the square of the difference
between each point.
'''
# Distance function
def Distance(i,j,dp,dimension):
    sum=0
    for dim in range(dimension):
        sum=sum+np.square(dp[i][dim]-dp[j][dim])
    distance = np.sqrt(sum)
    return distance

'''
I store each point distance with each other and store them in an array.
'''

## Calucate distance between all the points
def Difference(dim,dp):
    diffrow=[]
    diff=[]
    for i in range(n):
        sum=0
        for j in range(0,n):
            dist_=Distance(i,j,dp,dim)
            diffrow.insert(j,dist_)
        diff.insert(i,diffrow)        
        diffrow=[]
    return diff
            
    
final_n = []
value_list=[]


'''
Now we will iterate through n = [100, 1000]
for value of k = [1,100]
We will sort distance array. 
Now we will take last element as maximum value element
and nth index element of the array as minimum element as upto index n distance will be zero 
because those are the same points.
Now we will apply the given function to calulate the final value

'''


## Go through each value of n and k
for n in ns:
    print('Calculating for n : {}'.format(n))
    for dim in tqdm(k):
        dp=[]
        Total_run = 10 
        nOfI=[]
        maxval=0
        minval=0
        rk=0.0
        for i in range(Total_run):
            dp=np.random.random((n,dim))
            diff=Difference(dim,dp)
            diff=np.asarray(diff)
            diff=diff.flatten()
            diff=np.sort(diff)
            maxval=diff[len(diff)-1]
            minval=diff[n]
            rk=np.log10((maxval-minval)/minval)
            nOfI.insert(i,rk)
            rk=np.mean(nOfI)
        
        #print ("k={} : r(k) = {}".format(dim,rk))
        value_list.append((dim,rk))
    final_n.append(value_list)
    value_list = []
    

'''
We will plot between k and r(k) function.
We will represent plot both for n = 100 and  n = 1000
in a single plot for comparison.

k value is in x axis and r(k) is in y axis.
'''


### Plot between k and r(k)    
plt.plot([item[0] for item in final_n[0]],[item[1] for item in final_n[0]],label='n=100',color='red')
plt.plot([item[0] for item in final_n[1]],[item[1] for item in final_n[1]],label='n=1000',color='blue')
plt.title('Understanding the curse of dimensionality for n=100, 1000')
plt.xlabel('Value of k')
plt.ylabel('r(k)')
plt.legend()
plt.show()





