import os

INPUT = 0
OUTPUT = 1


class Perceptron:
    def __init__(self, inputs: list, weights: list, outputs: list, alpha=0.1, bias=-0.1):
        self.alpha = alpha
        self.bias = bias
        self.inputs = inputs
        self.weights = weights
        self.outputs = outputs
        self.training_data = [(input_[i], output_[i]) for i, (input_, output_) in enumerate(zip(inputs, outputs))]

    def __len__(self):
        return len(self.weights)

    def train(self):
        not_converged = True
        total_error = 0
        while not_converged:
            # logic
            for t_datum in self.training_data:
                o = self._compute_output()
                E = t_datum[OUTPUT] - o
                total_error += abs(E)  # TODO: ask if correct & ask about abs(E)
                for i, w in enumerate(self.weights):
                    self.weights[i] = (w[i] + self.alpha) * t_datum[INPUT] * E
            if total_error < self.bias:
                not_converged = False

    def _compute_output(self):
        return self.outputs
