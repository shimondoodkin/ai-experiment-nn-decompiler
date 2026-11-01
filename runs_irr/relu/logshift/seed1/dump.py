# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-1.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(0.69356555 * x0 + 0.33943146)  # layer 1 neuron 0
    h1_1 = relu(-1.1224508 * x0 - 0.5579148)  # layer 1 neuron 1
    h1_2 = relu(-0.022900978 * x0 + 0.62353975)  # layer 1 neuron 2
    h1_3 = relu(0.6433501 * x0 + 0.3146945)  # layer 1 neuron 3
    h1_4 = relu(-0.9058427 * x0 + 0.29518864)  # layer 1 neuron 4
    h1_5 = relu(0.37034887 * x0 - 0.8304091)  # layer 1 neuron 5
    h1_6 = relu(-0.7234609 * x0 - 0.8065401)  # layer 1 neuron 6
    h1_7 = relu(0.39446777 * x0 - 0.46958864)  # layer 1 neuron 7
    return 0.24654582 * h1_0 - 0.6152247 * h1_1 + 0.65044343 * h1_2 + 0.30446765 * h1_3 - 0.19328259 * h1_4 - 0.22227605 * h1_5 - 0.8577439 * h1_6 - 0.18723321 * h1_7 + 0.15916224
