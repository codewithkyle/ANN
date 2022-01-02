import numpy as np

class Layer:
    def __init__(self, inputs, neurons):
        self.weights = 0.1 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Softmax:
    def forward(self, inputs):
        values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = values / np.sum(values, axis=1, keepdims=True)

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

class Accuracy:
    def calculate(self, predictions, actuals):
        max_predictions = np.argmax(predictions, axis=1)
        accuracy = np.mean(max_predictions == actuals)
        return accuracy