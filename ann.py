import numpy as np

class Layer:
    def __init__(self, inputs, neurons):
        self.weights = 0.1 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Softmax:
    def forward(self, inputs):
        values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = values / np.sum(values, axis=1, keepdims=True)
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for i, (o, dv) in enumerate(zip(self.output, dvalues)):
            o = o.reshape(-1, 1)
            # Calculate the Jacobian matrix of the output
            matrix = np.diagflat(o) - np.dot(o, o.T)
            # Calculate the sample-wise graident and add it to the array of sample gradients
            self.dinputs[i] = np.dot(matrix, dv)

class Loss:
    def calculate(self, output, y):
        samples = self.forward(output, y)
        loss = np.mean(samples)
        return loss

class CategoricalCrossentropy(Loss):
    def forward(self, predictions, actuals):
        samples = len(predictions)
        clipped_samples = np.clip(predictions, 1e-7, 1 - 1e-7)
        if len(actuals.shape) == 1:
            correct_confidences = clipped_samples[range(samples), actuals]
        elif len(actuals.shape) == 2:
            correct_confidences = np.sum(clipped_samples * actuals, axis=1)
        neg_confidences = -np.log(correct_confidences)
        return neg_confidences
    def backward(self, dvalues, actuals):
        samples = len(dvalues)
        labels = len(dvalues[0])
        # If labels are sparse, turn them into a one-hot vector
        if len(actuals.shape) == 1:
            actuals = np.eye(labels)[actuals]
        gradient = -actuals / dvalues
        self.dinputs = gradient / samples

class Accuracy:
    def calculate(self, predictions, actuals):
        max_predictions = np.argmax(predictions, axis=1)
        accuracy = np.mean(max_predictions == actuals)
        return accuracy