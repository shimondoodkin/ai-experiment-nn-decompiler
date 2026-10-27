# dumped from a neural network: 8 inputs -> 1 -> 1 outputs
# inputs: x0: int [0, 255], x1: int [0, 255], x2: int [0, 255], x3: int [0, 255], x4: int [0, 255], x5: int [0, 255], x6: int [0, 255], x7: int [0, 255]
# output: regression: the returned value
def net(x0, x1, x2, x3, x4, x5, x6, x7):
    h1_0 = relu(0.02 * x0 + 0.02 * x1 + 0.02 * x2 + 0.02 * x3 + 0.02 * x4 + 0.02 * x5 + 0.02 * x6 + 0.02 * x7 - 20)  # layer 1 neuron 0
    return 1 * h1_0
