"""
Basic linear regression, optimization using analytical method or gradient descent
Tested on sklearn's simplified boston dataset
"""
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import pandas as pd
import math

import matplotlib
matplotlib.use('TkAgg');
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

"""
Implements an analytical method to find the coefficient matrix
"""
class LinReg:
    #parameters
    theta = None;
    #predicted = None;


    def __init__(self):
        pass

    #Analytical fit
    def ana_fit(self, X_train, y_train):
        m_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        self.theta = np.dot(np.dot(np.linalg.inv((np.dot(np.transpose(X_train), X_train))), np.transpose(X_train)),y_train)
        #print(self.cost(array=self.theta, m_samples=m_samples, X_train=X_train, y_train=y_train))

    #Batch gradient descent fit
    def gd_fit(self, X_train, y_train, alpha, threshold, lmbda):
        m_samples = X_train.shape[0]
        n_features = X_train.shape[1]

        #initialize theta
        self.theta = np.random.uniform(size=(n_features,))

        #Main part, gradient descent
        cost1 = self.cost(self.theta, m_samples, X_train, y_train)
        xpartial = self.partial(m_samples, self.theta, X_train,y_train)
        print("x partial is " + str(xpartial))
        self.theta = self.theta*(1-alpha*lmbda/m_samples) - alpha*xpartial
        cost2 = self.cost(self.theta, m_samples, X_train, y_train)
        epochs = 0
        print("cost 1 is " + str(cost1))
        print("cost 2 is " + str(cost2))
        while cost1-cost2 > threshold:
            print("cost is " + str(cost1) + " and distance to next is " + str(cost1 - cost2))
            #With regularization
            self.theta = self.theta*(1-alpha*lmbda/m_samples) - alpha * self.partial(m_samples, self.theta, X_train, y_train)
            cost1 = cost2
            cost2 = self.cost(self.theta, m_samples, X_train, y_train)
            epochs += 1
            print("GD trained for " + str(epochs) + " epochs")

    def cost(self, array, m_samples, X_train, y_train):
        sum = 0
        for k in range(0,m_samples):
            sum += (np.dot(X_train.iloc[k], array.T)-y_train.iloc[k])**2
        return float(sum * (1/2/m_samples))

    def partial(self, m_samples, theta, X_train, y_train):
        vectorsum = [];
        matrixsum = 0;
        for i in range(0, m_samples):
            vectorsum.append(np.dot(X_train.iloc[i], theta.T) - y_train.iloc[i])

        vectorsum = np.asarray(vectorsum)
        return np.dot(vectorsum.T, X_train)/m_samples

    def predict(self, X_test):
        return np.dot(X_test, self.theta.T)




boston= load_boston();
#print(boston.feature_names)
bos = pd.DataFrame(boston.data)
#print(bos.head(10))
#bos.columns = boston.feature_names

#print(boston.target.shape)
bos['PRICE'] = boston.target
#print(bos.head())

'''
Important stuff
'''
X = bos.drop('PRICE', axis=1)
Y = bos['PRICE']

#Now, split into train-test
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20, random_state = 1)
#Some preprocessing
X_train_scaled = pd.DataFrame(preprocessing.scale(X_train))
y_train_scaled = pd.DataFrame(preprocessing.scale(y_train))
#print(X_train_scaled)


#Learning
linreg = LinReg()

#Scaling turned out to be garbage, so lets just do it with none of that scaling
#linreg.ana_fit(X_train, y_train)
#linreg.gd_fit(X_train, y_train, alpha=0.000001, threshold=0.015)
linreg.gd_fit(X_train, y_train, alpha=.000002, threshold=0.001, lmbda = 0)
prediction = linreg.predict(X_test)
#print(prediction)

plt.scatter(y_test, prediction)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()

"""
def partial(self, m_samples, theta, X_train,y_train):
    i = 0;
    self.summation(m_samples,theta, X_train, y_train, 0)
    for element in theta:
        element = 1/m_samples * self.summation(m_samples, theta, X_train, y_train, i)
        #element = 1 / m_samples * self.costsum*X_train.iloc[k][i]
        i += 1

    return theta

def summation(self, m, theta, X_train, y_train, i):
    sum = 0
    for k in range(0,m):
        sum = sum + (np.dot(X_train.iloc[k], theta) - y_train.iloc[k])*X_train.iloc[k][i]
        #sum = sum + (np.dot(X_train.iloc[k], theta) - y_train.iloc[k])
    #self.costsum = sum;
    return sum

"""
