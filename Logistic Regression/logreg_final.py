"""
Breast cancer prediction implementing logistic regression
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import math

import numpy as np


# Logistic Regression Class. Will be called by main.
class LogReg:
    #class variables
    theta = None
    counter = 0
    #constructor
    def __init__(self):
        pass

    def logistic(self, z):
        """
        value = 1 / (1 + math.exp(-z))
        if value == 1.0:
            return 0.99
        elif z > 10 or z < -10:
            return 0.1
        else:
            return value
        :param z:
        :return:
        """
        return 1 / (1 + np.exp(-z))


    def inner_cost(self, hyp, label):
        return -label * math.log(hyp + 0.01) - (1-label) * math.log(1.01 - hyp)

    def Jcost(self, m_samples, X_train, y_train):
        total = 0
        for i in range(0, m_samples):
            #print(np.dot(self.theta, X_train.iloc[i]))
            #print(X_train.iloc[i])
            hyp = self.logistic(np.dot(self.theta.T, X_train.iloc[i]))
            label = y_train.iloc[i]
            #print(self.inner_cost(hyp, label))
            #print(hyp)
            total += self.inner_cost(hyp, label)

        return total

    def partial(self, m_samples, X_train, y_train):
        hyp_minus_y = []
        #get hyp minus y
        for i in range(0, m_samples):
            hyp = self.logistic(np.dot(self.theta.T, X_train.iloc[i]))
            hyp_minus_y.append(hyp - y_train.iloc[i])
        hyp_minus_y = np.asarray(hyp_minus_y)

        partialarray = []
        for i in range (0, X_train.shape[1]):
            # append inner product with every column
            partialarray.append(np.dot(hyp_minus_y, X_train.ix[:,i]))
        partialarray = np.asarray(partialarray)
        return partialarray/m_samples

    def fit(self, X_train, y_train, alpha, threshold):
        #recall shape[1] = features, shape[0] = sample points
        #self.theta = np.random.uniform(low=-0.01, high = 0.01, size=(X_train.shape[1],))
        self.theta = np.zeros(shape=X_train.shape[1])
        m_samples = X_train.shape[0]
        n_features = X_train.shape[1]

        cost1 = self.Jcost(m_samples, X_train, y_train)
        print(self.theta - alpha * self.partial(m_samples, X_train, y_train))
        self.theta = self.theta - alpha * self.partial(m_samples, X_train, y_train)
        cost2 = self.Jcost(m_samples, X_train, y_train)
        print(cost1)
        print(cost2)
        print("delta = " + str(cost1-cost2))
        self.counter += 1

        while(math.fabs(cost1-cost2) > threshold):
            print("epochs: " + str(self.counter))
            print("delta = " + str(cost1-cost2))
            cost1 = cost2
            self.counter += 1
            self.theta = self.theta - alpha * self.partial(m_samples, X_train, y_train)
            cost2 = self.Jcost(m_samples, X_train, y_train)

    def predict(self, X_test):
        result = np.dot(X_test, self.theta)
        fullresult = []
        for element in result:
            if element > 0.5:
                fullresult.append(1)
            else:
                fullresult.append(0)
        fullresult = np.asarray(fullresult)
        return fullresult

    def evaluate(self, y_test, prediction):
        #honestly though, this method shouldn't even be here
        count = 0
        m_samples = 0
        y_test = y_test.values
        for i in range(0, y_test.size):
            m_samples += 1
            if y_test.flat[i] == prediction.flat[i]:
                count += 1
        print(count/m_samples)

cancer_data = load_breast_cancer()
cancer = pd.DataFrame(cancer_data.data)
cancer["MALIGNANT"] = cancer_data.target
# print(cancer.head())
# print(cancer_data.target)
Xdata = cancer.drop("MALIGNANT", axis=1)
Ydata = cancer["MALIGNANT"]

X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size=0.20, random_state=1)

model = LogReg()

model.fit(X_train, y_train, alpha=8e-6, threshold=0.01)

prediction = model.predict(X_test)
model.evaluate(y_test, prediction)
