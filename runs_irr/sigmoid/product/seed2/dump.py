# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(1.5032332 * x0 - 1.6571631 * x1 - 6.1284976)  # layer 1 neuron 0
    h1_1 = sigmoid(1.631477 * x0 + 1.7124853 * x1 + 6.2495437)  # layer 1 neuron 1
    h1_2 = sigmoid(1.5521388 * x0 + 1.634221 * x1 - 1.9966896)  # layer 1 neuron 2
    h1_3 = sigmoid(-1.1545751 * x0 - 1.1938545 * x1 - 4.256313)  # layer 1 neuron 3
    h1_4 = sigmoid(0.8812233 * x0 - 0.93058443 * x1 + 2.7344577)  # layer 1 neuron 4
    h1_5 = sigmoid(1.0396113 * x0 + 1.053019 * x1 - 3.6797602)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.88981307 * x0 + 0.934305 * x1 - 2.688)  # layer 1 neuron 6
    h1_7 = sigmoid(-1.2130371 * x0 + 1.2665116 * x1 - 4.813454)  # layer 1 neuron 7
    h1_8 = sigmoid(1.1954162 * x0 - 1.2873158 * x1 - 4.333284)  # layer 1 neuron 8
    h1_9 = sigmoid(1.1888684 * x0 + 1.3770353 * x1 - 4.922114)  # layer 1 neuron 9
    h1_10 = sigmoid(-1.1105841 * x0 - 1.2222099 * x1 + 4.3604736)  # layer 1 neuron 10
    h1_11 = sigmoid(-1.2286143 * x0 + 1.2265023 * x1 + 3.756764)  # layer 1 neuron 11
    h1_12 = sigmoid(-1.6012572 * x0 + 1.6891611 * x1 + 2.0574827)  # layer 1 neuron 12
    h1_13 = sigmoid(-1.0202688 * x0 - 0.95148146 * x1 + 3.0864754)  # layer 1 neuron 13
    h1_14 = sigmoid(-1.0359753 * x0 - 1.0411258 * x1 + 3.6093879)  # layer 1 neuron 14
    h1_15 = sigmoid(-1.0735528 * x0 - 1.1626025 * x1 - 2.2012558)  # layer 1 neuron 15
    return -1.3570213 * h1_0 - 1.7227087 * h1_1 + 0.62774336 * h1_2 + 2.4136736 * h1_3 + 1.9367768 * h1_4 + 1.3034756 * h1_5 - 2.1760025 * h1_6 - 2.470908 * h1_7 - 2.1685076 * h1_8 + 1.6741489 * h1_9 - 1.1764972 * h1_10 + 1.7583058 * h1_11 + 0.87774897 * h1_12 - 1.2599379 * h1_13 - 0.6814403 * h1_14 + 2.0096323 * h1_15 + 0.2677656
