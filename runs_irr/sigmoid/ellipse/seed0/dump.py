# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(0.76836306 * x0 + 7.8867226 * x1 - 7.2753673)  # layer 1 neuron 0
    h1_1 = sigmoid(0.304011 * x0 - 8.3048 * x1 - 7.431514)  # layer 1 neuron 1
    h1_2 = sigmoid(-3.3643298 * x0 + 2.486644 * x1 + 7.8534245)  # layer 1 neuron 2
    h1_3 = sigmoid(-3.1531177 * x0 + 1.7082417 * x1 + 7.1566224)  # layer 1 neuron 3
    h1_4 = sigmoid(-1.7031059 * x0 + 8.176261 * x1 - 6.8727174)  # layer 1 neuron 4
    h1_5 = sigmoid(-3.9011428 * x0 - 0.55884695 * x1 + 8.771533)  # layer 1 neuron 5
    h1_6 = sigmoid(-3.4191833 * x0 - 1.817179 * x1 + 7.6493955)  # layer 1 neuron 6
    h1_7 = sigmoid(-1.574728 * x0 + 0.26956674 * x1 + 0.83272856)  # layer 1 neuron 7
    h1_8 = sigmoid(-4.423738 * x0 + 6.1779237 * x1 - 7.2036014)  # layer 1 neuron 8
    h1_9 = sigmoid(-1.9455049 * x0 - 7.5821266 * x1 - 6.40092)  # layer 1 neuron 9
    h1_10 = sigmoid(-2.231107 * x0 - 1.5360057 * x1 - 2.0104637)  # layer 1 neuron 10
    h1_11 = sigmoid(2.8334305 * x0 + 4.970693 * x1 - 7.8660307)  # layer 1 neuron 11
    h1_12 = sigmoid(-4.506056 * x0 - 4.6763687 * x1 - 6.850685)  # layer 1 neuron 12
    h1_13 = sigmoid(2.4672413 * x0 - 6.2469125 * x1 - 7.9990325)  # layer 1 neuron 13
    h1_14 = sigmoid(-1.8820411 * x0 + 0.26375934 * x1 + 3.7360282)  # layer 1 neuron 14
    h1_15 = sigmoid(-5.967695 * x0 + 1.0088217 * x1 - 7.637211)  # layer 1 neuron 15
    return -9.678816 * h1_0 - 8.913276 * h1_1 + 4.241828 * h1_2 + 3.790526 * h1_3 - 9.48836 * h1_4 + 4.036526 * h1_5 + 3.2252834 * h1_6 - 0.12473626 * h1_7 - 8.206323 * h1_8 - 8.84027 * h1_9 - 2.1061187 * h1_10 - 9.590382 * h1_11 - 9.062076 * h1_12 - 9.189526 * h1_13 + 0.6388689 * h1_14 - 7.568587 * h1_15 - 0.7502606
