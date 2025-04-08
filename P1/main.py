import numpy as np
import NeuralNetwork as nn
import matplotlib.pyplot as plt

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

w1_his, w2_his, mse_history, class_error = NN.train(
    x, y,
    epochs=300,
    lr=0.3,
    momentum=0.9
)

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

# # Wykresy wag
# plt.subplot(221)
# plt.plot(w1_his[:,0])
# plt.title("Wagi W11, W21")
#
# plt.subplot(222)
# plt.plot(w1_his[:,1])
# plt.title("Wagi W12, W22")
#
# plt.subplot(223)
# plt.plot(w2_his[:,0])
# plt.title("Waga W3")
#
# plt.subplot(224)
# plt.plot(w2_his[:,1])
# plt.title("Waga W4")
#
# plt.tight_layout(pad=0.75)
# # plt.suptitle("Wykresy wag")
# plt.show()
#
# # Wykres mse
# plt.plot(mse_history[:,0])
# plt.plot(mse_history[:,1])
# plt.grid(True)
# plt.title("MSE")
# plt.legend(["MSE warstwy ukrytej", "MSE warstwy wyjściowej"])
# plt.xlabel("Epoki")
# plt.show()
#
# # Wykres błedu klasyfikacji
# plt.plot(class_error)
# plt.grid(True)
# plt.title("Błąd klasyfikacji")
# plt.ylabel("Procent błędnych klasyfikacji")
# plt.xlabel("Epoki")
# plt.show()
