# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-1.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(-0.00748682 * x0 - 0.088744044)  # layer 1 neuron 0
    h1_1 = relu(0.25899693 * x0 + 0.92314863)  # layer 1 neuron 1
    h1_2 = relu(-1.116676 * x0 - 0.76860905)  # layer 1 neuron 2
    h1_3 = relu(-0.63639164 * x0 + 0.613617)  # layer 1 neuron 3
    h1_4 = relu(-0.38515437 * x0 - 0.9553485)  # layer 1 neuron 4
    h1_5 = relu(0.14953882 * x0 - 0.74987096)  # layer 1 neuron 5
    h1_6 = relu(-0.01981318 * x0 - 0.4122231)  # layer 1 neuron 6
    h1_7 = relu(0.63928926 * x0 + 0.43251845)  # layer 1 neuron 7
    return 0.13977209 * h1_0 + 0.35924762 * h1_1 - 0.724774 * h1_2 - 0.39945436 * h1_3 + 0.12841663 * h1_4 + 0.13343269 * h1_5 - 0.07276136 * h1_6 + 0.21095748 * h1_7 + 0.49139827
