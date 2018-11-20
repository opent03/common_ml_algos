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
Hidden layer1: 20 neurons ( + 1 bias)
Hidden layer2: 20 neurons ( + 1 bias)
Output layer : 10 neurons corresponding to 0-9 digits
"""
class Neural_Network:
    costs = []

    def __init__(self):
        pass


    def relu(self, vector):
        # return vector * (vector > 0)
        return np.where(vector > 0, vector, vector * 0.01)

    def drelu(self, vector, alpha=.01):
        for i in range(vector.shape[0]):
            for j in range(vector.shape[1]):
                if vector[i][j] < 0:
                    vector[i][j] = 0.01
                else:
                    vector[i][j] = 1
        #return 1 * (vector > 0)
        return vector

            # return alpha if x < 0 else 1

        #return np.array([1 if i >= 0 else alpha for i in vector])

    def logistic(self, z):
        return 1 / (1 + np.exp(-z))

    def dlogistic(self, z):
        return np.multiply(self.logistic(z), (1-self.logistic(z)))

    def softmax(self, z):
        return np.exp(z)/np.sum(np.exp(z), axis=0)

    # Vectorized implementation of for/back prop in an Andrew Ng~ish style
    def fit(self, X_train, y_train, alpha, epochs, skip_train, beta):
        m_samples = X_train.shape[0]
        self.in_size = X_train.shape[1]
        self.hid1_size = 16
        self.hid2_size = 16
        self.out_size = 10

        # Initialize the essentials
        self.W1 = 2/self.in_size * np.random.randn(self.in_size, self.hid1_size)
        self.W2 = 2/self.hid1_size * np.random.randn(self.hid1_size, self.hid2_size)
        self.W3 = 2/self.hid2_size * np.random.randn(self.hid2_size, self.out_size)
        self.b1 = np.zeros((self.hid1_size, 1))
        self.b2 = np.zeros((self.hid2_size, 1))
        self.b3 = np.zeros((self.out_size, 1))

        # Momentum terms
        self.VdW1 = np.zeros(shape=self.W1.shape)
        self.VdW2 = np.zeros(shape=self.W2.shape)
        self.VdW3 = np.zeros(shape=self.W3.shape)
        self.Vdb1 = np.zeros(shape=self.b1.shape)
        self.Vdb2 = np.zeros(shape=self.b2.shape)
        self.Vdb3 = np.zeros(shape=self.b3.shape)
        alpha_init = alpha
        # 4ward prop, cost, and backprop, a decent number of times
        if not skip_train:
            for i in range(epochs):
                self.hyp = self.forward_prop(X_train)
                self.cost1 = self.cost(y_train)
                dW3, dW2, dW1, db3, db2, db1 = self.back_prop(X_train, y_train)

                # Alpha decay, lmaoo sounds so physics-y
                alpha = np.power(0.995, i/2) * alpha_init
                print("alpha now: " + str(alpha))
                # Momentum
                self.VdW1 = beta * self.VdW1 + (1-beta)*dW1
                self.VdW2 = beta * self.VdW2 + (1-beta)*dW2
                self.VdW3 = beta * self.VdW3 + (1-beta)*dW3
                self.Vdb1 = beta * self.Vdb1 + (1-beta)*db1
                self.Vdb2 = beta * self.Vdb2 + (1-beta)*db2
                self.Vdb3 = beta * self.Vdb3 + (1-beta)*db3
                
                #Update
                self.W3 = self.W3 - alpha*self.VdW3
                self.W2 = self.W2 - alpha*self.VdW2
                self.W1 = self.W1 - alpha*self.VdW1

                self.b3 = self.b3 - alpha*self.Vdb3
                self.b2 = self.b2 - alpha*self.Vdb2
                self.b1 = self.b1 - alpha*self.Vdb1
                print("epochs: " + str(i+1) + " --- " + "cost: " + "{0:5f}".format(self.cost1))
                self.costs.append(self.cost1)

        return self.costs

    def forward_prop(self, X):
        # Activation 2
        self.Z1 = np.matmul(X, self.W1)
        # add the bias manually, cuz I suck at numpy
        for i in range(self.Z1.shape[0]):
            self.Z1[i] = np.add(self.Z1[i], self.b1.T)
        # self.A2 = self.logistic(self.Z1)
        self.A2 = self.relu(self.Z1)

        # Activation 3
        self.Z2 = np.matmul(self.A2,self.W2)
        for i in range(self.Z2.shape[0]):
            self.Z2[i] = np.add(self.Z2[i], self.b2.T)
        # self.A3 = self.logistic(self.Z2)
        self.A3 = self.relu(self.Z2)

        # Activation 4
        self.Z3 = np.matmul(self.A3, self.W3)
        A4 = self.Z3
        for i in range(self.Z3.shape[0]):
            A4[i] = self.softmax(A4[i])
        return A4

    def back_prop(self,X, Y):
        # dLdW3
        dLdA4 = (self.hyp - Y)
        dA4dZ3 = self.hyp * (1 - self.hyp)   # derivative of softmax, kind of
        delta3 = np.multiply(dLdA4, dA4dZ3)
        dLdW3 = np.matmul(self.A3.T, delta3)

        #dLdW2
        dLdA3 = np.matmul(delta3, self.W3.T)
        # delta2 = np.multiply(dLdA3, self.dlogistic(self.Z2))
        delta2 = np.multiply(dLdA3, self.drelu(self.Z2))
        dLdW2 = np.matmul(self.A2.T, delta2)

        # dLdW1
        dLdA2 = np.matmul(delta2, self.W2.T)
        # delta1 = np.multiply(dLdA2, self.dlogistic(self.Z1))
        delta1 = np.multiply(dLdA2, self.drelu(self.Z1))
        dLdW1 = np.matmul(X.T, delta1)

        # bias
        db3 = (1. / X.shape[0]) * np.sum(dA4dZ3, axis=0, keepdims=True)
        db2 = (1. / X.shape[0]) * np.sum(delta2, axis=0, keepdims=True)
        db1 = (1. / X.shape[0]) * np.sum(delta1, axis=0, keepdims=True)
        return dLdW3, dLdW2, dLdW1, db3.T, db2.T, db1.T  # Don't ask why biases are transposed. Just don't.

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
epochs = 300
# Training
start = time.time()
costs = neural_net.fit(X_train, y_train, alpha=1e-4, epochs=epochs, skip_train=False, beta=0.95)
end = time.time()

# Prediction and evaluation
prediction = neural_net.predict(X_test)
eval = neural_net.evaluate(y_test, prediction)
print()
print("your accuracy on test scores was: " + str(eval * 100) + "%")
print("Model took " + "{0:.2f}".format(end-start) + "s to finish")

plt.xlabel("epochs")
plt.ylabel("cost")
plt.title("Cost over time")
plt.plot(range(1, epochs + 1), costs)
plt.show()
