# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-1, 3]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(-0.08078086 * x0 + 0.8680082)  # layer 1 neuron 0
    h1_1 = tanh(0.077704966 * x0 - 0.8875587)  # layer 1 neuron 1
    h1_2 = tanh(-0.088789955 * x0 + 0.8431916)  # layer 1 neuron 2
    h1_3 = tanh(-0.122069076 * x0 + 0.7382782)  # layer 1 neuron 3
    h1_4 = tanh(-1.2607439 * x0 - 1.1773298)  # layer 1 neuron 4
    h1_5 = tanh(0.07395393 * x0 - 0.9488779)  # layer 1 neuron 5
    h1_6 = tanh(0.09692164 * x0 - 0.79161847)  # layer 1 neuron 6
    h1_7 = tanh(0.7144647 * x0 - 0.22272377)  # layer 1 neuron 7
    return 0.7220696 * h1_0 - 0.6913667 * h1_1 + 0.9903678 * h1_2 + 0.82561624 * h1_3 + 1.258514 * h1_4 - 0.90722936 * h1_5 - 0.805016 * h1_6 - 1.0762299 * h1_7 + 0.40397
