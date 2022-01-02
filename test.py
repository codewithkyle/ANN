from ann import Accuracy, CategoricalCrossEntropy, Layer, ReLU, Softmax, Softmax_CategoricalCrossEntropy
from data import create_data

# Prep layers
layer1 = Layer(2, 3)
layer2 = Layer(3, 3)

# Prep activation methods
act1 = ReLU()
accuracy_func = Accuracy()
loss_activation = Softmax_CategoricalCrossEntropy()

# Do work
X, y = create_data(100, 3)

layer1.forward(X)
act1.forward(layer1.output)
layer2.forward(act1.output)
loss = loss_activation.forward(layer2.output, y)

# Output stuffs
print(loss_activation.output[:5])

accuracy = accuracy_func.calculate(loss_activation.output, y)
print("accuracy:", accuracy)
print("loss:", loss)

# Backward pass
loss_activation.backward(loss_activation.output, y)
layer2.backward(loss_activation.dinputs)
act1.backward(layer2.dinputs)
layer1.backward(act1.dinputs)

print(layer1.dweights)
print(layer1.dbiases)
print(layer2.dweights)
print(layer2.dbiases)