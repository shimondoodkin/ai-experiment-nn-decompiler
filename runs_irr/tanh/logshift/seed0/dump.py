# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-1.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.14170983 * x0 - 0.13070558)  # layer 1 neuron 0
    h1_1 = tanh(0.48648033 * x0 + 0.42367)  # layer 1 neuron 1
    h1_2 = tanh(-1.0410706 * x0 - 1.7329288)  # layer 1 neuron 2
    h1_3 = tanh(-0.6624579 * x0 - 0.6379209)  # layer 1 neuron 3
    h1_4 = tanh(-2.1559033 * x0 - 3.482573)  # layer 1 neuron 4
    h1_5 = tanh(0.18633872 * x0 - 0.76840216)  # layer 1 neuron 5
    h1_6 = tanh(-0.28622264 * x0 - 0.29059172)  # layer 1 neuron 6
    h1_7 = tanh(0.29511583 * x0 + 0.01542368)  # layer 1 neuron 7
    return 0.22260498 * h1_0 + 0.2807802 * h1_1 - 0.495316 * h1_2 - 0.20998351 * h1_3 - 0.37507138 * h1_4 + 0.5260174 * h1_5 - 0.15064868 * h1_6 + 0.33609003 * h1_7 - 0.058766637
