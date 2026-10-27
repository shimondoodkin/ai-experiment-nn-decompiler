# dumped from a neural network: 2 inputs -> 2 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = relu(1.0082177 * x0 - 0.9861872 * x1 + 0.0062654046)  # layer 1 neuron 0
    h1_1 = relu(-0.9724584 * x0 + 0.9709917 * x1 + 0.008255112)  # layer 1 neuron 1
    return 1.0063982 * h1_0 + 1.0137697 * h1_1 + 0.00087249983
