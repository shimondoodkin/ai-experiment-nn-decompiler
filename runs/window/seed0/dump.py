# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = relu(-0.00748682 * x0 - 0.38515437)  # layer 1 neuron 0
    h1_1 = relu(7.717265 * x0 - 2.9967031)  # layer 1 neuron 1
    h1_2 = relu(-5.9997625 * x0 - 2.54685)  # layer 1 neuron 2
    h1_3 = relu(-6.799037 * x0 - 2.846557)  # layer 1 neuron 3
    return -0.044372022 * h1_0 - 6.8681664 * h1_1 - 4.827274 * h1_2 - 6.9872804 * h1_3 + 6.086123
