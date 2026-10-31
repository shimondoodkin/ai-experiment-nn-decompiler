# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.39654398 * x0 - 0.27132893)  # layer 1 neuron 0
    h1_1 = sigmoid(0.79450506 * x0 - 0.019442862)  # layer 1 neuron 1
    h1_2 = sigmoid(-4.5846996 * x0 - 1.1390285)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.38067526 * x0 + 0.70100695)  # layer 1 neuron 3
    h1_4 = sigmoid(0.41811585 * x0 - 1.2155416)  # layer 1 neuron 4
    h1_5 = sigmoid(0.42725042 * x0 - 0.8865621)  # layer 1 neuron 5
    h1_6 = sigmoid(0.3757912 * x0 - 0.82945895)  # layer 1 neuron 6
    h1_7 = sigmoid(1.3925303 * x0 - 0.23116665)  # layer 1 neuron 7
    return 0.41837227 * h1_0 + 0.46417052 * h1_1 - 1.5369749 * h1_2 - 0.83756316 * h1_3 + 0.50247073 * h1_4 + 0.66114444 * h1_5 + 0.16314809 * h1_6 + 0.54791015 * h1_7 + 0.057546727
