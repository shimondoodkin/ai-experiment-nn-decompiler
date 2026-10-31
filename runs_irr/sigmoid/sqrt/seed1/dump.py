# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(1.5155395 * x0 - 1.1113979)  # layer 1 neuron 0
    h1_1 = sigmoid(0.5420739 * x0 - 1.2017393)  # layer 1 neuron 1
    h1_2 = sigmoid(0.45945975 * x0 - 0.0733332)  # layer 1 neuron 2
    h1_3 = sigmoid(0.5675838 * x0 - 1.2661457)  # layer 1 neuron 3
    h1_4 = sigmoid(-3.8963091 * x0 - 0.77327377)  # layer 1 neuron 4
    h1_5 = sigmoid(0.5503278 * x0 - 1.0774451)  # layer 1 neuron 5
    h1_6 = sigmoid(0.58179486 * x0 - 1.6934747)  # layer 1 neuron 6
    h1_7 = sigmoid(0.56008756 * x0 - 1.3019911)  # layer 1 neuron 7
    return 0.45183372 * h1_0 + 0.3587406 * h1_1 + 0.5094356 * h1_2 + 0.5013738 * h1_3 - 1.4704008 * h1_4 + 0.20353442 * h1_5 + 0.36937806 * h1_6 + 0.2573229 * h1_7 - 0.09841253
