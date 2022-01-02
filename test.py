from ann import Accuracy, Layer, ReLU, SGD_Optimizer, Softmax_CategoricalCrossEntropy
from data import create_data

# Prep
layer1 = Layer(2, 64)
layer2 = Layer(64, 3)
act1 = ReLU()
accuracy_func = Accuracy()
loss_activation = Softmax_CategoricalCrossEntropy()
optimizer = SGD_Optimizer()

# Create test data
X, y = create_data(100, 3)

for epoch in range(10001):
    # Do work
    layer1.forward(X)
    act1.forward(layer1.output)
    layer2.forward(act1.output)
    loss = loss_activation.forward(layer2.output, y)
    accuracy = accuracy_func.calculate(loss_activation.output, y)
    
    if not epoch % 100:
        print(  f'epoch: {epoch}, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f}')

    # Backpropagation
    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.dinputs)
    act1.backward(layer2.dinputs)
    layer1.backward(act1.dinputs)

    # Optimize
    optimizer.update(layer1)
    optimizer.update(layer2)
