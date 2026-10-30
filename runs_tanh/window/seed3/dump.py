# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = tanh(-8.581059 * x0 - 4.229925)  # layer 1 neuron 0
    h1_1 = tanh(0.019569732 * x0 - 2.3264046)  # layer 1 neuron 1
    h1_2 = tanh(-0.02166792 * x0 + 2.6083045)  # layer 1 neuron 2
    h1_3 = tanh(-8.330184 * x0 + 4.138304)  # layer 1 neuron 3
    return -7.845801 * h1_0 + 3.0901828 * h1_1 - 2.6007807 * h1_2 + 7.913191 * h1_3 - 2.2518291
