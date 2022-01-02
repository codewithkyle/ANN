from data import create_data
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
