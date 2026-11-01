# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-1.1497984 * x0 + 0.16079807)  # layer 1 neuron 0
    h1_1 = sigmoid(0.14719409 * x0 + 1.0796663)  # layer 1 neuron 1
    h1_2 = sigmoid(0.37265214 * x0 - 0.074617065)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.2486041 * x0 + 0.8105986)  # layer 1 neuron 3
    h1_4 = sigmoid(0.46240366 * x0 - 1.0279318)  # layer 1 neuron 4
    h1_5 = sigmoid(-4.360175 * x0 - 0.96887445)  # layer 1 neuron 5
    h1_6 = sigmoid(0.50649613 * x0 - 1.2584528)  # layer 1 neuron 6
    h1_7 = sigmoid(0.107278086 * x0 - 0.33749154)  # layer 1 neuron 7
    return -0.8899676 * h1_0 + 0.4082859 * h1_1 + 0.69309384 * h1_2 - 0.06917914 * h1_3 + 0.74469894 * h1_4 - 1.4096713 * h1_5 + 0.84985346 * h1_6 - 0.0011336381 * h1_7 + 0.03746224
