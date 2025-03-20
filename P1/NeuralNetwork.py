import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDeriv(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self):
        self.inputLayer = 2
        self.hiddenLayer = 2
        self.outputLayer = 1
        self.W1 = np.random.rand(self.inputLayer, self.hiddenLayer)
        self.W2 = np.random.rand(self.hiddenLayer, self.outputLayer)
        self.b1 = np.random.rand(1, self.hiddenLayer)
        self.b2 = np.random.rand(1, self.outputLayer)

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = sigmoid(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = sigmoid(z2)

        return a1, a2

    def backward(self, x, y, a1, a2, lr):
        dz2 = y - a2
        dw2 = dz2 * sigmoidDeriv(a2)
        dz1 = dw2.dot(self.W2.T)
        dw1 = dz1 * sigmoidDeriv(a1)

        self.W1 += lr * x.T.dot(dw1)
        self.b1 += lr * np.sum(dw1)
        self.W2 += lr * a1.T.dot(dw2)
        self.b2 += lr * np.sum(dw2)

    def train(self, x, y, epochs, lr=0.1):
        for _ in range(epochs):
            a1, a2 = self.forward(x)
            self.backward(x, y, a1, a2, lr=lr)
