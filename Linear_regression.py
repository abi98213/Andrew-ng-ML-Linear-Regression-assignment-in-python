import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os



data = pd.read_csv("D:\ML exercises\CourseraML-master\CourseraML-master\ex1\data\ex1data1.txt",header = None , names = ["population" ,"profit"])
data.head()
#---------------------------------- COST FUNCTION ------------------------------------ #
def compute_cost(X,y,theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

data.insert(0,"ones",1)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 
theta = np.matrix(np.array([0,0]))
X = np.matrix(X.values)  
y = np.matrix(y.values)  





## ------------------------------- Gradient descent ---------------------------------#
def grad_des(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(2))
    parameters = 2
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X*theta.T)-y

        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        #cost[i] = compute_cost(X,y,theta)
        
    return theta,cost

alpha = 0.01
iters = 2000

g,cost = grad_des(X,y,theta,alpha,iters)
print(g)

x = np.linspace(data.population.min(), data.population.max(), 2000)
f = g[0, 0] + (g[0, 1] * x)
print(f)
fig, ax = plt.subplots(figsize=(12,12))  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.population, data.profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('population')  
ax.set_ylabel('profit')  
ax.set_title('predicted profit vs. population Size')

plt.show()

#---------------------------    predict with values ------------------------#
get = int(input("please tell the value to predict"))
print(g[0,0] + g[0,1]*get,"is the approximation value")

