# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.008854946 * x0 + 2.4060724)  # layer 1 neuron 0
    h1_1 = tanh(0.022936026 * x0 - 1.5555298)  # layer 1 neuron 1
    h1_2 = tanh(-0.01674064 * x0 - 0.111168385)  # layer 1 neuron 2
    h1_3 = tanh(-0.044301346 * x0 + 4.6687326)  # layer 1 neuron 3
    h1_4 = tanh(-0.42943776 * x0 - 5.0267816)  # layer 1 neuron 4
    h1_5 = tanh(0.050122883 * x0 - 9.095751)  # layer 1 neuron 5
    h1_6 = tanh(-0.2718507 * x0 - 3.7730043)  # layer 1 neuron 6
    h1_7 = tanh(0.043727506 * x0 - 6.3360105)  # layer 1 neuron 7
    return 6.9177465 * h1_0 + 6.988588 * h1_1 - 7.008162 * h1_2 - 7.0372314 * h1_3 - 6.675667 * h1_4 + 7.5225267 * h1_5 - 6.8768086 * h1_6 + 7.3729334 * h1_7 + 6.746825
