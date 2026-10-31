# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(0.16305967 * x0 + 0.5524818)  # layer 1 neuron 0
    h1_1 = relu(-0.23797429 * x0 - 0.8853446)  # layer 1 neuron 1
    h1_2 = relu(0.10113003 * x0 - 0.050020818)  # layer 1 neuron 2
    h1_3 = relu(-0.36469713 * x0 + 0.6583396)  # layer 1 neuron 3
    h1_4 = relu(0.43482074 * x0 - 0.21621978)  # layer 1 neuron 4
    h1_5 = relu(-0.11199996 * x0 + 1.2017558)  # layer 1 neuron 5
    h1_6 = relu(-0.17498848 * x0 + 0.0066823983)  # layer 1 neuron 6
    h1_7 = relu(-1.0735633 * x0 + 0.97426134)  # layer 1 neuron 7
    return 0.24922225 * h1_0 - 0.20857231 * h1_1 - 0.028819049 * h1_2 + 1.2345539 * h1_3 - 0.33671194 * h1_4 + 0.19450846 * h1_5 + 0.032031935 * h1_6 + 1.4523456 * h1_7 + 0.29321197
