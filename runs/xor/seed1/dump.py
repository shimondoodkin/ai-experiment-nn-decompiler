# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = relu(3.209216 * x0 - 3.1135607 * x1 + 0.037852257)  # layer 1 neuron 0
    h1_1 = relu(-3.029829 * x0 + 3.2472088 * x1 + 3.0290747)  # layer 1 neuron 1
    h1_2 = relu(-2.8020875 * x0 + 2.8020873 * x1 + 0.000002432534)  # layer 1 neuron 2
    h1_3 = relu(-4.805323 * x0 + 4.8053827 * x1 - 0.000023159055)  # layer 1 neuron 3
    return 4.2488623 * h1_0 - 3.6241052 * h1_1 + 3.2231588 * h1_2 + 5.986145 * h1_3 - 1.8743557
