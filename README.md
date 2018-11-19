# Entry-level machine learning algos
This repository contains custom implementations of popular statistical learning algorithms in Python3. Datasets tested on taken from sklearn.datasets.

# Linear regression
Train on sklearn-modified boston dataset. Implements quadratic loss, batch gradient descent, as well as an optional analytical method of O(n^3) complexity. Also added regularization. 

# Logistic regression
Trained on sklearn-modified cancer dataset (and does a surprisingly decent job). Implements batch gradient descent on max likelihood cost function.

# Neural network
Trained on MNIST, layers 784-20-10. Achieved ~ 93% accuracy at around 200 epochs. It runs surprisingly quickly, if your processor is not an intel pentium or anything below. For the actual maths, email me. 
The mnist.pkl.gz was borrowed from https://www.kaggle.com/pablotab/mnistpklgz. 


