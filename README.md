# Some ml algos
This repository contains custom implementations of some learning algorithms in Python3. Datasets tested on taken from sklearn.datasets.

# Linear regression
Train on sklearn-modified boston dataset. Implements quadratic loss, batch gradient descent, as well as an optional analytical method. Added regularization and momentum to gd.

# Logistic regression
Trained on sklearn-modified cancer dataset (and does a surprisingly decent job). Implements batch gradient descent on max likelihood cost function.

# Neural networks
nn.py: Trained on MNIST, layers 784-20-10. Achieved ~ 94% accuracy at around 100 epochs. It runs surprisingly quickly, if your processor is not an intel pentium or anything below. For the actual maths, email me. 
The mnist.pkl.gz was borrowed from https://www.kaggle.com/pablotab/mnistpklgz. 

run with python3 nn.py --dir mnist.pkl.gz


nn_3layers.py: Trained on MNIST also, structure 784-10-10-10. Unsurprisingly worse than the 2 layer one training with the same #epochs because converges slower. Implements momentum, currently working on implementing nesterov. 


UPDATE: Added momentum for bgd. By observation, accuracy on test data peaks at around 93%-94% due to the nature of the model. Lowering the costs faster by momentum bgd only causes the model to overfit. Switched to relu, Xavier init, and lr exp decay.


# the extra jupyter notebook i uploaded most recently
Takes a meme approach to classifying mnist by using umap+tsne features and xgboost. ~98%.
