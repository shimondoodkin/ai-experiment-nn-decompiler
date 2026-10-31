# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.1, 5]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(2.920558 * x0 - 0.7371503)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.6291887 * x0 + 1.6720992)  # layer 1 neuron 1
    h1_2 = sigmoid(0.48162374 * x0 - 0.14454955)  # layer 1 neuron 2
    h1_3 = sigmoid(1.8564974 * x0 - 0.6428555)  # layer 1 neuron 3
    h1_4 = sigmoid(-8.002196 * x0 + 0.398109)  # layer 1 neuron 4
    h1_5 = sigmoid(0.61161554 * x0 - 0.98608255)  # layer 1 neuron 5
    h1_6 = sigmoid(-1.7222905 * x0 + 0.60362595)  # layer 1 neuron 6
    h1_7 = sigmoid(0.7794297 * x0 - 1.0973868)  # layer 1 neuron 7
    return 0.60686547 * h1_0 - 1.3591479 * h1_1 + 0.48588654 * h1_2 + 0.5937213 * h1_3 - 2.5494578 * h1_4 + 0.19652772 * h1_5 - 1.1185839 * h1_6 + 0.2554286 * h1_7 - 0.19941404
