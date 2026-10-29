# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-0.6182739 * x0 - 0.77149737 * x1 + 3.9695134)  # layer 1 neuron 0
    h1_1 = sigmoid(-10.7385025 * x0 + 10.737853 * x1 - 2.1923623)  # layer 1 neuron 1
    h1_2 = sigmoid(-10.623384 * x0 + 10.599084 * x1 + 2.205706)  # layer 1 neuron 2
    h1_3 = sigmoid(-11.074528 * x0 + 11.07239 * x1 - 2.2463348)  # layer 1 neuron 3
    h1_4 = sigmoid(1.7260085 * x0 - 1.4223734 * x1 - 0.65520287)  # layer 1 neuron 4
    h1_5 = sigmoid(5.641488 * x0 - 5.638573 * x1 - 1.5761894)  # layer 1 neuron 5
    h1_6 = sigmoid(-9.624256 * x0 + 9.60161 * x1 + 2.052009)  # layer 1 neuron 6
    h1_7 = sigmoid(9.614263 * x0 - 9.591309 * x1 - 2.051349)  # layer 1 neuron 7
    return -1.3202841 * h1_0 - 11.4479475 * h1_1 + 7.433669 * h1_2 - 11.229401 * h1_3 - 0.60660565 * h1_4 - 3.984827 * h1_5 + 6.1978173 * h1_6 - 7.4637494 * h1_7 - 0.32562438
