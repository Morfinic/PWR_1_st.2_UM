import numpy as np
import NeuralNetwork as nn

x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0], [1], [1], [0]
])

NN = nn.NeuralNetwork()

# Create and train the neural network
print("Before training:")
predictions = NN.forward(x)
print("Predictions:\n", predictions[1])

NN.train(x, y, epochs=10_000)

# Test the trained network
print("\nAfter training:")
predictions = NN.forward(x)
print("Predictions:\n", predictions[1], "\n")

for i in range(x.shape[0]):
    pred = NN.forward(x[i])
    if pred[-1][-1][0] >= 0.5:
        pred = 1
    else:
        pred = 0
    print(f"Data: {x[i]} | Prediction: {pred} | {y[i][0]}")