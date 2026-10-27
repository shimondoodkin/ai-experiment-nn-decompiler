# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = relu(-0.00748682 * x0 - 0.38515437)  # layer 1 neuron 0
    h1_1 = relu(2.9497313 * x0 + 0.6870758)  # layer 1 neuron 1
    h1_2 = relu(-6.032427 * x0 - 2.6576083)  # layer 1 neuron 2
    h1_3 = relu(-6.6844716 * x0 - 2.935136)  # layer 1 neuron 3
    return -0.044372022 * h1_0 - 2.1414557 * h1_1 - 4.9372754 * h1_2 - 6.788889 * h1_3 + 4.5610127
