# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(-0.9675831 * x0 - 0.9218849 * x1 + 1.0159254)  # layer 1 neuron 0
    h1_1 = sigmoid(1.0515056 * x0 - 0.9950888 * x1 - 1.2737435)  # layer 1 neuron 1
    h1_2 = sigmoid(1.1077025 * x0 - 1.1041028 * x1 - 4.012792)  # layer 1 neuron 2
    h1_3 = sigmoid(-1.5625771 * x0 - 1.5699453 * x1 + 1.4376283)  # layer 1 neuron 3
    h1_4 = sigmoid(-2.043423 * x0 + 2.054188 * x1 - 2.1983342)  # layer 1 neuron 4
    h1_5 = sigmoid(2.671079 * x0 - 2.664358 * x1 - 2.5020027)  # layer 1 neuron 5
    h1_6 = sigmoid(-1.3332583 * x0 + 1.3414316 * x1 - 5.028585)  # layer 1 neuron 6
    h1_7 = sigmoid(1.9078987 * x0 + 1.9712238 * x1 + 0.7066641)  # layer 1 neuron 7
    return 0.11193924 * h1_0 + 0.37809524 * h1_1 + 4.0686545 * h1_2 - 0.034991324 * h1_3 + 2.3091571 * h1_4 + 1.5548137 * h1_5 + 3.5025601 * h1_6 + 0.029824596 * h1_7 - 0.34955162
