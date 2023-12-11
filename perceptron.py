from random import uniform

INPUT = 0
OUTPUT = 1


class Perceptron:
    def __init__(self, inputs: list, outputs: list, alpha=0.1, bias=-0.1):
        """
        Constructor for a Perceptron. Initializes the list of weights to random number between -0.5 and 0.5.
        Initializes training_data to a list of tuples containing the corresponding inputs with an output.

        :param inputs: list of given inputs; can be 2D, but should be congruent to the length of outputs
        :param outputs: list of expected outputs; should be congruent to the length of inputs
        :param alpha: learning rate, default is set to 0.1
        :param bias: bias ..., default is set to -0.1
        """
        self.alpha = alpha
        self.bias = bias
        self.inputs = inputs
        self.weights = [[uniform(-0.5, 0.5)] for i in range(len(inputs)) for j in range(len(inputs[i]))]
        self.outputs = outputs
        self.training_data = [(input_, output_) for i, (input_, output_) in enumerate(zip(inputs, outputs))]

    def __len__(self):
        return len(self.weights)

    def __str__(self):
        string = "Perceptron\n"
        for t_datum in self.training_data:
            string += f"Inputs: {t_datum[INPUT]}, Output: {t_datum[OUTPUT]}\n"
        return string

    def train(self):
        """
        Method used to adjust the weights of the Perceptron based on the training data provided.
        """
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
