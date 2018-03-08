import unittest
from NeuralNetwork import NeuralNetwork
import random


class TestNeuralNetwork(unittest.TestCase):
    def test_decide(self):
        network = NeuralNetwork(4, 5, 0.5)
        network.decide([[1], [1], [1], [1]])
        assert True

    def test_learn(self):
        network = NeuralNetwork(4, 5, 0.5)
        network.decide([[1], [1], [1], [1]])
        network.learn(1, 0, [[1], [1], [1], [1]])
        assert True

    def test_learn_value_false(self):
        network = NeuralNetwork(4, 5, 0.5)
        for i in range(100):
            network.learn(network.decide([[1], [1], [1], [1]]), 0, [[1], [1], [1], [1]])
        self.assertLess(network.decide([[1], [1], [1], [1]]), 0.5,
                        "network decision is larger than 0.5 after 100 iterations")

    def test_learn_value_true(self):
        network = NeuralNetwork(4, 5, 0.5)
        for i in range(100):
            network.learn(network.decide([[1], [1], [1], [1]]), 1, [[1], [1], [1], [1]])
        self.assertLess( 0.5, network.decide([[1], [1], [1], [1]]),
                        "network decision is lower than 0.5 after 100 iterations")

    def test_not_logic(self):
        values = [[[0]], [[1]]]
        true_decisions = [1, 0]

        network = NeuralNetwork(1, 2, 0.5)

        for i in range(1000):
            for j, value in enumerate(values):
                network.learn(network.decide(value), true_decisions[j], value)
        for j, value in enumerate(values):
            self.assertEqual(network.decide(value) > 0.5, true_decisions[j] > 0.5,
                             "network decision for value '{0}' is '{1}' and not '{2}'"
                             .format(value, network.decide(value), true_decisions[j]))

    def test_and_logic(self):
        values = [[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]]
        true_decisions = [0, 0, 0, 1]

        network = NeuralNetwork(2, 3, 0.5)

        i = True
        number_of_epohs = 0

        while i:
            number_of_epohs = number_of_epohs + 1
            i = False

            c = list(zip(values, true_decisions))
            random.shuffle(c)
            values, true_decisions = zip(*c)

            for j, value in enumerate(values):
                network.learn(network.decide(value), true_decisions[j], value)
            for j, value in enumerate(values):
                if abs(network.decide(value) - true_decisions[j]) > 0.01:
                    i = True

        print(network.hidden_weights)
        print(network.output_weights)
        print(number_of_epohs)

        for value in values:
            print(network.decide(value))

        for j, value in enumerate(values):
            self.assertEqual(network.decide(value) > 0.5, true_decisions[j] > 0.5,
                             "network decision for value '{0}' is '{1}' and not '{2}'"
                             .format(value, network.decide(value), true_decisions[j]))

    def test_or_logic(self):
        values = [[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]]
        true_decisions = [0, 1, 1, 1]

        network = NeuralNetwork(2, 3, 0.5)

        i = True
        number_of_epohs = 0

        while i:
            number_of_epohs = number_of_epohs + 1
            i = False

            c = list(zip(values, true_decisions))
            random.shuffle(c)
            values, true_decisions = zip(*c)

            for j, value in enumerate(values):
                network.learn(network.decide(value), true_decisions[j], value)
            for j, value in enumerate(values):
                if abs(network.decide(value) - true_decisions[j]) > 0.01:
                    i = True

        print(network.hidden_weights)
        print(network.output_weights)
        print(number_of_epohs)

        for value in values:
            print(network.decide(value))

        for j, value in enumerate(values):
            self.assertEqual(network.decide(value) > 0.5, true_decisions[j] > 0.5,
                             "network decision for value '{0}' is '{1}' and not '{2}'"
                             .format(value, network.decide(value), true_decisions[j]))

    def test_xor_logic(self):
        values = [[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]]
        true_decisions = [0, 1, 1, 0]

        network = NeuralNetwork(2, 3, 0.5)

        i = True
        number_of_epohs = 0

        while i:
            number_of_epohs = number_of_epohs + 1
            i = False

            c = list(zip(values, true_decisions))
            random.shuffle(c)
            values, true_decisions = zip(*c)

            for j, value in enumerate(values):
                network.learn(network.decide(value), true_decisions[j], value)
            for j, value in enumerate(values):
                if abs(network.decide(value)-true_decisions[j]) > 0.01:
                    i = True

        print(network.hidden_weights)
        print(network.output_weights)
        print(number_of_epohs)

        for value in values:
            print(network.decide(value))

        for j, value in enumerate(values):
            self.assertEqual(network.decide(value) > 0.5, true_decisions[j] > 0.5,
                             "network decision for value '{0}' is '{1}' and not '{2}'"
                             .format(value, network.decide(value), true_decisions[j]))
