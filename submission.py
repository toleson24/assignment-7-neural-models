import perceptron

if __name__ == "__main__":
    inputs = [
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    ]
    outputs = [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]

    # create a perceptron using the default alpha=0.5 and bias=-0.5
    p = perceptron.Perceptron(inputs=inputs, outputs=outputs)
    print(p)