# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = relu(2.0802977 * x0 + 3.9619465 * x1 - 2.080285)  # layer 1 neuron 0
    h1_1 = relu(-3.401813 * x0 - 3.4014056 * x1 + 3.4014304)  # layer 1 neuron 1
    h1_2 = relu(-0.27234524 * x0 + 0.1896159 * x1 - 0.2136969)  # layer 1 neuron 2
    h1_3 = relu(-2.825353 * x0 + 2.8253255 * x1 - 0.0000149551815)  # layer 1 neuron 3
    return -5.299233 * h1_0 - 6.0862217 * h1_1 - 0.20611155 * h1_2 + 4.350323 * h1_3 + 9.590704
