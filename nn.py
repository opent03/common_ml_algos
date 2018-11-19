import pickle
import os
import gzip
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
path = os.getcwd() + "/mnist.pkl.gz"
import time
"""
Train a neural network on MNIST data (LeCun et al.) with 1 hidden layer
Input layer: 28x28 = 784 neurons ( + 1 bias)
Hidden layer: 20 neurons ( + 1 bias)
Output layer : 10 neurons corresponding to 0-9 digits
"""
class Neural_Network:

    def __init__(self):
        pass


    def relu(self, vector):
        return vector * (vector > 0)

    def drelu(self, vector):
        return 1 * (vector > 0)

    def logistic(self, z):
        return 1 / (1 + np.exp(-z))

    def dlogistic(self, z):
        return np.multiply(self.logistic(z), (1-self.logistic(z)))

    def softmax(self, z):
        return np.exp(z)/np.sum(np.exp(z), axis=0)

    # Vectorized implementation of for/back prop in an Andrew Ng~ish style
    def fit(self, X_train, y_train, alpha, epochs,bounds, skip_train, beta):
        m_samples = X_train.shape[0]
        self.in_size = X_train.shape[1]
        self.hid_size = 20
        self.out_size = 10

        # Initialize the essentials
        self.W1 = np.random.uniform(low=-bounds, high=bounds, size=(self.in_size, self.hid_size))
        self.W2 = np.random.uniform(low=-bounds, high=bounds, size=(self.hid_size, self.out_size))
        self.b1 = np.zeros((self.hid_size, 1))
        self.b2 = np.zeros((self.out_size, 1))

        # Momentum terms
        self.VdW1 = np.zeros(shape=self.W1.shape)
        self.VdW2 = np.zeros(shape=self.W2.shape)
        self.Vdb1 = np.zeros(shape=self.b1.shape)
        self.Vdb2 = np.zeros(shape=self.b2.shape)

        # 4ward prop, cost, and backprop, a decent number of times
        if not skip_train:
            for i in range(epochs):
                self.hyp = self.forward_prop(X_train)

                self.cost1 = self.cost(y_train)
                dW2, dW1, db2, db1 = self.back_prop(X_train, y_train)

                # Momentum
                self.VdW1 = beta * self.VdW1 + (1-beta)*dW1
                self.VdW2 = beta * self.VdW2 + (1-beta)*dW2
                self.Vdb1 = beta * self.Vdb1 + (1-beta)*db1
                self.Vdb2 = beta * self.Vdb2 + (1-beta)*db2

                #Update
                self.W2 = self.W2 - alpha*self.VdW2
                self.W1 = self.W1 - alpha*self.VdW1
                self.b2 = self.b2 - alpha*self.Vdb2
                self.b1 = self.b1 - alpha*self.Vdb1

                print("epochs: " + str(i+1) + " --- " + "cost: " + "{0:5f}".format(self.cost1))

    def forward_prop(self, X):
        #Standard stuff
        self.Z1 = np.matmul(X, self.W1)
        # add the bias manually, cuz I suck at numpy
        for i in range(self.Z1.shape[0]):
            self.Z1[i] = np.add(self.Z1[i], self.b1.T)

        self.A2 = self.logistic(self.Z1)
        self.Z2 = np.matmul(self.A2,self.W2)
        for i in range(self.Z2.shape[0]):
            self.Z2[i] = np.add(self.Z2[i], self.b2.T)

        # Softmax, again cuz I couldn't figure out a vectorized implementation
        A3 = self.Z2
        for i in range(self.Z2.shape[0]):
            A3[i] = self.softmax(A3[i])
        return A3

    def back_prop(self,X, Y):
        # dLdW2
        dLdA3 = (self.hyp - Y)
        dA3dZ2 = self.hyp * (1 - self.hyp)   # derivative of softmax, kind of
        delta2 = np.multiply(dLdA3, dA3dZ2)  # delta2 is (A3-y) times S'(Z2)
        dLdW2 = np.matmul(self.A2.T, delta2)

        #dLdW1
        dLdA2 = np.matmul(delta2, self.W2.T)
        delta1 = np.multiply(dLdA2, self.dlogistic(self.Z1))
        dLdW1 = np.matmul(X.T, delta1)

        # bias
        db2 = (1. / X.shape[0]) * np.sum(dA3dZ2, axis=0, keepdims=True)
        db1 = (1. / X.shape[0]) * np.sum(delta1, axis=0, keepdims=True)
        return dLdW2, dLdW1, db2.T, db1.T  # Don't ask why biases are transposed. Just don't.

    def cost(self, Y):
        # Quadratic loss
        return np.sum(1/2 * np.power((Y - self.hyp), 2))

    def predict(self, X):
        hyp = self.forward_prop(X)
        # the zero one thingy, convert from a probability distr thing to something that resembles y_test
        for i in range(hyp.shape[0]):
            max = np.amax(hyp[i])
            for j in range(hyp.shape[1]):
                if hyp[i][j] == max:
                    hyp[i][j] = 1.
                else:
                    hyp[i][j] = 0
        return hyp

    def evaluate(self, Y, pred):
        good = 0
        for i in range(Y.shape[0]):
            if np.allclose(Y[i], pred[i]):
                good += 1
        return float(good) / Y.shape[0]

"""
MAIN STUFF BEGINS HERE
"""
# Loads the data
def load_data():
    f = gzip.open(path, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

    X, Y = training_data[0], training_data[1]

    # Minimal preprocessing. Convert stuff into format that's easier to work with
    X = np.array([np.reshape(element, (784,)) for element in X])
    Y = np.array([np.reshape(vectorize(element), (10,)) for element in Y])

    return X, Y


# Vectorize the labels
def vectorize(label):
    e = np.zeros((10, 1))
    e[label] = 1.0
    return e


X, Y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Instantiates a neural net object
neural_net = Neural_Network()

# Training
start = time.time()
neural_net.fit(X_train, y_train, alpha=5e-3, epochs=400, bounds=0.05, skip_train=False, beta=0.9)
end = time.time()

# Prediction and evaluation
prediction = neural_net.predict(X_test)
eval = neural_net.evaluate(y_test, prediction)
print()
print("your accuracy on test scores was: " + str(eval * 100) + "%")
print("Model took " + "{0:.2f}".format(end-start) + "s to finish")
