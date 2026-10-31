# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.1, 5]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(0.98786867 * x0 - 0.09686631)  # layer 1 neuron 0
    h1_1 = relu(-0.44137824 * x0 - 0.12244415)  # layer 1 neuron 1
    h1_2 = relu(-0.44164088 * x0 + 0.000010145024)  # layer 1 neuron 2
    h1_3 = relu(0.899174 * x0 - 0.08796322)  # layer 1 neuron 3
    h1_4 = relu(-1.8273487 * x0 + 0.98217255)  # layer 1 neuron 4
    h1_5 = relu(0.8519084 * x0 - 1.5929973)  # layer 1 neuron 5
    h1_6 = relu(-0.20572531 * x0 - 0.07290876)  # layer 1 neuron 6
    h1_7 = relu(0.53601104 * x0 - 0.291266)  # layer 1 neuron 7
    return 0.3732865 * h1_0 - 0.0014123917 * h1_1 + 0.076065466 * h1_2 + 0.44430214 * h1_3 - 1.5052483 * h1_4 - 0.6875714 * h1_5 - 0.059261084 * h1_6 + 0.22641093 * h1_7 - 0.80209684
