import perceptron

if __name__ == "__main__":
    # Task 1: Perceptron
    inputs = [
        [1, 0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1],
        [0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 1],
        [0, 1, 1, 1, 0, 0]
    ]
    outputs = [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]

    # create a perceptron using the default alpha=0.5 and bias=-0.5
    # p = perceptron.Perceptron(inputs=inputs, outputs=outputs, random=False)
    p = perceptron.Perceptron(inputs=inputs, outputs=outputs)
    print(p)

    p.train()
    print(p)

    # Task 2: TensorFlow
    # See /Task 2/a07_q2.pdf

    # Task 3: Applying Neural Models
    # I was unable to complete this section. I have included the start of my copied Question/Answer notebook tutorial,
    # chosen because I heard from some classmates that that one took the least time to train. I was not able to train
    # it, however. My machine's python environment uses conda, and the recommended `pip install ...`'s didn't resolve
    # the issue.
