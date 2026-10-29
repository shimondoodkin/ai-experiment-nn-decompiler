# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.06805449 * x0 - 2.4489007)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.14783706 * x0 + 0.70665395)  # layer 1 neuron 1
    h1_2 = sigmoid(-0.07640823 * x0 + 6.910298)  # layer 1 neuron 2
    h1_3 = sigmoid(0.2355187 * x0 + 4.2630224)  # layer 1 neuron 3
    h1_4 = sigmoid(-0.10496228 * x0 + 18.808764)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.08284328 * x0 + 10.636189)  # layer 1 neuron 5
    h1_6 = sigmoid(0.086354285 * x0 - 12.973445)  # layer 1 neuron 6
    h1_7 = sigmoid(0.07901946 * x0 - 8.017792)  # layer 1 neuron 7
    return 11.410747 * h1_0 + 0.083200194 * h1_1 - 7.587355 * h1_2 + 16.398783 * h1_3 - 13.216715 * h1_4 - 7.8798323 * h1_5 + 11.339807 * h1_6 + 11.278129 * h1_7 + 11.522534
