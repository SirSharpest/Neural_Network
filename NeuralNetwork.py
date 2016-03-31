import numpy as np


class NeuralNetwork(object):
    def __init__(self):
        # Define Structure
        self.inputSize = 2 # Going to pass in 2x3 matrix of entries
        self.outputSize = 1 # Have a single output value
        self.hiddenSize = 3 # Bounce around between 3 other hidden nodes

        # Setup some weight s
        self.W1 = np.random.randn(self.inputSize, \
                                 self.hiddenSize)

        self.W2 = np.random.randn(self.hiddenSize, \
                                  self.outputSize)

    def forward(self, X):
        # Push data forward through the network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yGuess = self.sigmoid(self.z3)
        return yGuess


    def sigmoid(self, z):
        # Apply the sigmoid activation
        return 1 / (1 + np.exp(-z))



NN = NeuralNetwork()

guessScore = NN.forward([2,2])

print(guessScore)
