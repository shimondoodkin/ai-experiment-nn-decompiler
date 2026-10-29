# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = tanh(4.931534 * x0 - 3.4728825 * x1 + 1.0687357)  # layer 1 neuron 0
    h1_1 = tanh(2.7796848 * x0 - 3.8430114 * x1 - 1.2766601)  # layer 1 neuron 1
    h1_2 = tanh(3.1403697 * x0 + 3.5585437 * x1 - 0.60573095)  # layer 1 neuron 2
    h1_3 = tanh(3.5885487 * x0 - 4.990999 * x1 - 1.530669)  # layer 1 neuron 3
    return -9.449446 * h1_0 + 3.8666673 * h1_1 + 4.302479 * h1_2 + 6.5500083 * h1_3 + 6.060332
