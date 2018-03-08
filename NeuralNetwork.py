import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def one_minus_x(x):
    return 1 - x


class NeuralNetwork:

    def __init__(self, number_of_markers, number_of_hidden_neurons, learning_factor):
        self.hidden_weights = np.matrix(np.random.rand(number_of_hidden_neurons, number_of_markers))
        self.output_weights = np.matrix(np.random.rand(1, number_of_hidden_neurons))
        self.learning_factor = learning_factor
        self.z = np.matrix([[0]])

    @staticmethod
    def activation_function(value):
        f = np.vectorize(sigmoid, otypes=[np.float])
        return f(value)

    @staticmethod
    def one_minus_x_vectorised(value):
        f = np.vectorize(one_minus_x, otypes=[np.float])
        return f(value)

    def decide(self, environment_markers):
        self.z = self.activation_function(np.matmul(self.hidden_weights, environment_markers))
        y = self.activation_function(np.matmul(self.output_weights, self.z))
        return y

    def learn(self, decision, expected_decision, environment_markers):
        output_weights_error = self.calculate_output_weights_error(decision,
                                                                   expected_decision)
        hidden_weights_error = self.calculate_hidden_weights_error(decision,
                                                                   expected_decision,
                                                                   environment_markers)

        self.output_weights = self.output_weights - self.learning_factor * output_weights_error
        self.hidden_weights = self.hidden_weights - self.learning_factor * hidden_weights_error

    def calculate_output_weights_error(self, decision, expected_decision):
        ea_output = decision - expected_decision
        ei_output = ea_output * decision * (1 - decision)
        return ei_output * np.transpose(self.z)

    def calculate_hidden_weights_error(self, decision, expected_decision, environment_markers):
        ea_output = decision - expected_decision
        ei_output = ea_output * decision * (1 - decision)
        ea_hidden = np.transpose(self.output_weights)*ei_output
        ei_hidden = np.multiply(ea_hidden, np.multiply(self.z, one_minus_x(self.z)))
        return ei_hidden*np.transpose(environment_markers)
