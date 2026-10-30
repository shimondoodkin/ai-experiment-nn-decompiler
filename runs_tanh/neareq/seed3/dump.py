# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = tanh(-5.520886 * x0 + 5.5211787 * x1 + 1.1934696)  # layer 1 neuron 0
    h1_1 = tanh(-0.29872337 * x0 + 0.22122276 * x1 + 2.3721032)  # layer 1 neuron 1
    h1_2 = tanh(-0.28517064 * x0 + 0.32021883 * x1 + 2.2545319)  # layer 1 neuron 2
    h1_3 = tanh(4.803676 * x0 - 4.806717 * x1 - 1.0870215)  # layer 1 neuron 3
    h1_4 = tanh(3.56023 * x0 - 3.5700958 * x1 - 0.9370002)  # layer 1 neuron 4
    h1_5 = tanh(-6.3329067 * x0 + 6.3210115 * x1 - 1.1619831)  # layer 1 neuron 5
    h1_6 = tanh(-0.280051 * x0 + 0.2660689 * x1 + 2.365532)  # layer 1 neuron 6
    h1_7 = tanh(0.07911021 * x0 - 0.36597002 * x1 - 2.3229399)  # layer 1 neuron 7
    return 7.4596653 * h1_0 - 3.2713218 * h1_1 - 3.1747665 * h1_2 - 6.345746 * h1_3 - 3.5957928 * h1_4 - 12.914624 * h1_5 - 3.1965826 * h1_6 + 3.1930876 * h1_7 - 2.5696857
