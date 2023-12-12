from random import uniform

INPUT = 0
OUTPUT = 1


class Perceptron:
    def __init__(self, inputs: list, outputs: list, alpha=0.1, bias=-0.1, random=True):
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
        if random:
            self.weights = [[uniform(-0.5, 0.5) for i in range(len(inputs[0]))] for j in range(len(inputs))]
        else:
            self.weights = [[0.0 for i in range(len(inputs[0]))] for j in range(len(inputs))]
        self.outputs = outputs
        self.training_data = [(input_, output_) for i, (input_, output_) in enumerate(zip(inputs, outputs))]

    def __len__(self):
        return len(self.weights)

    def __str__(self):
        """
        An overly complex, but pretty--and readable--customized string representation of this Perceptron

        :return: string representation of this Perceptron
        """
        string = "Perceptron\n"
        for i, t_datum in enumerate(self.training_data):
            string += f"Example {i + 1:2.0f}: "
            weights_str = "["
            for j, w in enumerate(self.weights[i]):
                weights_str += f"{w: 2.2f}"
                if j != len(self.weights[i]) - 1:
                    weights_str += ", "
            weights_str += "]"
            string += f"Inputs: {t_datum[INPUT]}, Weights: {weights_str}, Output: {t_datum[OUTPUT]}\n"
        return string

    def train(self):
        """
        Method used to adjust the weights of the Perceptron based on the training data provided.
        """
        not_converged = True
        while not_converged:
            total_correct = 0
            for i, t_datum in enumerate(self.training_data):
                o = self._compute_output(example_num=i)
                E = t_datum[OUTPUT] - o
                if o == self.outputs[i]:
                    total_correct += 1
                self._calculate_weights(example_num=i, error=E)
            print(total_correct)
            if total_correct == len(self.outputs):
                not_converged = False

    def _compute_output(self, example_num: int):
        """
        Private method for Perceptron that a sum of weighted inputs and runs that through an activation function

        :param example_num: index for a list of inputs and a list weights
        :return: the ReLU value for a list of weighted inputs
        """
        output_sum = 0
        for input_, weight in zip(self.inputs[example_num], self.weights[example_num]):
            output_sum += input_ * weight
        output = ReLU(output_sum)
        return output

    def _calculate_weights(self, example_num, error):
        """
        Private method for Perceptron that updates its own weights based on the error provided. See __init__() for
        default alpha value.

        :param example_num: index for a list of inputs to be updated
        :param error: error calculated to updated individual weights
        """
        for i, w in enumerate(self.weights[example_num]):
            self.weights[example_num][i] = w + self.alpha * self.inputs[example_num][i] * error


def ReLU(input_: int):
    """
    Rectified Linear Unit (ReLU) activation function.
    The ReLU function returns 1 if the input is greater than 0, and 0 otherwise.

    :param input_: input value to be checked
    :return: 0 or 1
    """
    if input_ > 0:
        return 1
    else:
        return 0
