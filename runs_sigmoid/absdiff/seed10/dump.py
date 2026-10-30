# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(-1.822148 * x0 + 1.8201852 * x1 - 2.811874)  # layer 1 neuron 0
    h1_1 = sigmoid(-1.3452752 * x0 + 1.3477148 * x1 - 3.0907555)  # layer 1 neuron 1
    h1_2 = sigmoid(-0.90629375 * x0 + 0.90858394 * x1 - 3.8168862)  # layer 1 neuron 2
    h1_3 = sigmoid(-4.036333 * x0 + 4.0376134 * x1 - 2.400831)  # layer 1 neuron 3
    h1_4 = sigmoid(3.1452408 * x0 - 3.1473548 * x1 - 2.3650498)  # layer 1 neuron 4
    h1_5 = sigmoid(0.8870394 * x0 - 0.8785271 * x1 - 3.9386482)  # layer 1 neuron 5
    h1_6 = sigmoid(1.0953159 * x0 - 1.0982621 * x1 - 3.009903)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.9734821 * x0 + 0.96807516 * x1 - 4.1855617)  # layer 1 neuron 7
    return 0.84983116 * h1_0 + 1.0494318 * h1_1 + 1.2578058 * h1_2 + 0.9006252 * h1_3 + 1.2626919 * h1_4 + 2.627767 * h1_5 + 2.6236236 * h1_6 + 2.5748863 * h1_7 - 0.34618703
