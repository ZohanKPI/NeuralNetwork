import numpy as np
from itertools import product

class NeuralNetwork:
    # ... the rest of your class ...
    def __init__(self, input_size, output_size, layers, activations):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = self.initialize_layers(layers, input_size, output_size)
        self.activations = activations
        self.activation_derivatives = [self.sigmoid_derivative if act == self.sigmoid else self.relu_derivative for act
                                       in activations]

    def initialize_layers(self, layers, input_size, output_size):
        np.random.seed(0)  # for reproducibility
        layers = [{'weights': np.random.randn(next_layer, prev_layer) * np.sqrt(1. / prev_layer),
                   'biases': np.zeros((next_layer, 1))}
                  for prev_layer, next_layer in zip([input_size] + layers[:-1], layers)]
        layers.append({'weights': np.random.randn(output_size, layers[-1]['weights'].shape[0]) * np.sqrt(
            1. / layers[-1]['weights'].shape[0]),
                       'biases': np.zeros((output_size, 1))})
        return layers

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        sig = NeuralNetwork.sigmoid(z)
        return sig * (1 - sig)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0) * 1.0

    @staticmethod
    def softmax(z):
        exps = np.exp(z - np.max(z))
        return exps / np.sum(exps, axis=0)

    @staticmethod
    def compute_cost(Y, Y_hat):
        m = Y.shape[0]
        log_probs = -np.log(Y_hat[Y, range(m)])
        cost = np.sum(log_probs) / m
        return cost

    def forward(self, input_vector):
        self.a = [input_vector]
        self.z = []
        for i, layer in enumerate(self.layers):
            z = np.dot(layer['weights'], self.a[-1]) + layer['biases']
            self.z.append(z)
            if i < len(self.activations):
                activation_function = self.activations[i]
                a = activation_function(z)
                self.a.append(a)
        return self.a[-1]

    def backward(self, input_vector, output_vector):
        m = output_vector.shape[0]
        Y_hat = self.a[-1]
        dz = Y_hat.copy()
        dz[output_vector, range(m)] -= 1
        dw_db_pairs = []
        for i in reversed(range(len(self.layers))):
            dw = 1 / m * np.dot(dz, self.a[i].T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)
            if i > 0:  # Different calculation for dz if not at input layer
                activation_derivative = self.activation_derivatives[i - 1]
                dz = activation_derivative(self.z[i - 1]) * np.dot(self.layers[i]['weights'].T, dz)
            dw_db_pairs.append((dw, db))
        return dw_db_pairs[::-1]  # reverse the list to match layers order

    def update_weights(self, dw, db, learning_rate):
        for i, layer in enumerate(self.layers):
            layer['weights'] -= learning_rate * dw[i]
            layer['biases'] -= learning_rate * db[i]

    def train(self, input_vectors, output_vectors, learning_rate=0.01, epochs=10):
        output_vectors = np.array(output_vectors).reshape(1, -1)
        for epoch in range(epochs):
            a = self.forward(input_vectors)
            cost = self.compute_cost(output_vectors, a)
            dw_db_pairs = self.backward(input_vectors, output_vectors)
            dw, db = zip(*dw_db_pairs)
            self.update_weights(dw, db, learning_rate)
            print(f'Epoch {epoch + 1} / {epochs}: cost = {cost}')
    def predict(self, input_vector):
        a = self.forward(input_vector)
        return np.argmax(a, axis=0)
def main():
    input_neurons = np.array([
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1,
         0, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0]
    ])
    output_neurons = np.array([0, 1, 2, 3, 4])
    layers = [4, 3]
    activations = [NeuralNetwork.relu, NeuralNetwork.sigmoid, NeuralNetwork.softmax]
    epochs = 100
    nn = NeuralNetwork(input_neurons.shape[1], np.max(output_neurons) + 1, layers, activations)
    nn.train(input_neurons.T, output_neurons, learning_rate=0.01, epochs=epochs)
    correct_predictions = 0
    for i in range(input_neurons.shape[0]):  # For each example
        input_vector = input_neurons[i, :].reshape(-1, 1)
        output_vector = output_neurons[i]
        prediction = nn.predict(input_vector)
        if output_vector == prediction:
            correct_predictions += 1

    accuracy = correct_predictions / len(output_neurons)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()