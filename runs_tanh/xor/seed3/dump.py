# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = tanh(-4.19795 * x0 - 3.119935 * x1 + 1.5712616)  # layer 1 neuron 0
    h1_1 = tanh(-2.7852428 * x0 - 4.769847 * x1 + 5.1967344)  # layer 1 neuron 1
    h1_2 = tanh(-4.9888563 * x0 - 4.029618 * x1 + 2.003212)  # layer 1 neuron 2
    h1_3 = tanh(3.9698553 * x0 - 3.2773876 * x1 + 2.330267)  # layer 1 neuron 3
    return -3.8260484 * h1_0 + 9.468778 * h1_1 - 6.683126 * h1_2 - 5.413048 * h1_3 - 5.082132
