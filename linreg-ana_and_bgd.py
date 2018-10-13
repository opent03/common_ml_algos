"""
Basic linear regression, optimization using analytical method or gradient descent
Tested on sklearn's simplified boston dataset
"""
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

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
    predicted = None;


    def __init__(self):
        pass

    #Analytical fit
    def ana_fit(self, X_train, y_train):
        self.theta = np.dot(np.dot(np.linalg.inv((np.dot(np.transpose(X_train), X_train))), np.transpose(X_train)),y_train)

    #Batch gradient descent fit
    def gd_fit(self, X_train, y_train, alpha, threshold):
        m_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        #initialize theta
        self.theta = np.random.uniform(low=min(X_train), high=max(X_train), size=(n_features,))

        #Main part, gradient descent
        cost1 = self.cost(self.theta, m_samples, X_train, y_train)
        self.theta = self.theta - alpha*self.partial(m_samples, self.theta, y_train)
        cost2 = self.cost(self.theta, m_samples, X_train, y_train) + 10
        epochs = 0
        while(cost1-cost2 > threshold):
            print("cost is " + str(cost1) + " and threshold is " + str(threshold))
            cost1 = cost2
            self.theta = self.theta - alpha * self.partial(m_samples, self.theta, y_train)
            cost2 = self.cost(self.theta, m_samples, X_train, y_train)
            epochs += 1
            print("GD trained for " + str(epochs) + " epochs")

    def cost(self, theta, m_samples, X_train, y_train):
        sum = 0
        for k in range(0,m_samples):
            sum += (np.dot(X_train.iloc[k], theta)-y_train.iloc[k])**2
        return sum * (1/2/m_samples)

    def partial(self, m_samples, theta, y_train):
        i = 0
        for element in theta:
            element = 1/m_samples * self.summation(m_samples, theta, X_train, y_train, i)
            i+=1

        return theta

    def summation(self, m, theta, X_train, y_train, i):
        sum = 0
        for k in range(0,m):
            sum = sum + (np.dot(X_train.iloc[k], theta) - y_train.iloc[k])*X_train.iloc[k][i]
        return sum

    def predict(self, X_test):
        return np.dot(X_test, self.theta)




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

#Some preprocessing

#Now, split into train-test
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20, random_state = 1)



#Learning
linreg = LinReg()
linreg.gd_fit(X_train, y_train, alpha=0.005, threshold=0.000001)

prediction = linreg.predict(X_test)
#print(prediction)
plt.scatter(y_test, prediction)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()





