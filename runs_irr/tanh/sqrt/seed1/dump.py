# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.1755805 * x0 - 0.09782816)  # layer 1 neuron 0
    h1_1 = tanh(-0.1803773 * x0 + 0.35201102)  # layer 1 neuron 1
    h1_2 = tanh(0.12499831 * x0 + 0.45040336)  # layer 1 neuron 2
    h1_3 = tanh(0.7470477 * x0 - 0.011881443)  # layer 1 neuron 3
    h1_4 = tanh(-2.9595203 * x0 - 0.45395112)  # layer 1 neuron 4
    h1_5 = tanh(0.2364951 * x0 - 0.8101719)  # layer 1 neuron 5
    h1_6 = tanh(-0.24410425 * x0 - 0.044794828)  # layer 1 neuron 6
    h1_7 = tanh(0.20335962 * x0 - 0.59275496)  # layer 1 neuron 7
    return 0.32376263 * h1_0 - 0.2941157 * h1_1 + 0.39511123 * h1_2 + 0.40328923 * h1_3 - 0.54350096 * h1_4 + 0.2225142 * h1_5 - 0.37231115 * h1_6 + 0.22138433 * h1_7 + 0.11393125
