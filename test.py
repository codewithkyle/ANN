from ann import Accuracy, CategoricalCrossentropy, Layer, ReLU, Softmax
from data import create_data

# Prep layers
layer1 = Layer(2, 3)
layer2 = Layer(3, 3)

# Prep activation methods
act1 = ReLU()
act2 = Softmax()

loss_func = CategoricalCrossentropy()
accuracy_func = Accuracy()

# Do work
X, y = create_data(100, 3)

layer1.forward(X)
act1.forward(layer1.output)
layer2.forward(act1.output)
act2.forward(layer2.output)

# print(act2.output[:5])

loss = loss_func.calculate(act2.output, y)
print("Loss:", loss)

accuracy = accuracy_func.calculate(act2.output, y)
print("Accuracy:", accuracy)
