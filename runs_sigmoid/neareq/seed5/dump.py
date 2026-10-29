# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(8.749549 * x0 - 8.738662 * x1 + 2.0211608)  # layer 1 neuron 0
    h1_1 = sigmoid(0.69822484 * x0 + 0.06419884 * x1 + 2.763372)  # layer 1 neuron 1
    h1_2 = sigmoid(10.054672 * x0 - 10.059973 * x1 - 1.9768194)  # layer 1 neuron 2
    h1_3 = sigmoid(-9.968111 * x0 + 9.973679 * x1 + 1.9624091)  # layer 1 neuron 3
    h1_4 = sigmoid(-7.425706 * x0 + 7.4212894 * x1 - 1.870957)  # layer 1 neuron 4
    h1_5 = sigmoid(0.3636079 * x0 - 0.990282 * x1 + 0.91710687)  # layer 1 neuron 5
    h1_6 = sigmoid(-8.954106 * x0 + 8.942184 * x1 - 2.0480843)  # layer 1 neuron 6
    h1_7 = sigmoid(0.6229638 * x0 - 0.12688589 * x1 + 2.1588435)  # layer 1 neuron 7
    return 7.234173 * h1_0 - 1.34526 * h1_1 - 14.30969 * h1_2 + 9.882558 * h1_3 - 9.431426 * h1_4 - 1.3083005 * h1_5 - 13.052932 * h1_6 - 0.9667872 * h1_7 - 1.0346873
