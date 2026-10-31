# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(1.761199 * x0 - 3.2488368)  # layer 1 neuron 0
    h1_1 = sigmoid(-1.8027769 * x0 + 3.2981436)  # layer 1 neuron 1
    h1_2 = sigmoid(3.5444727 * x0 - 7.2882)  # layer 1 neuron 2
    h1_3 = sigmoid(1.5709745 * x0 - 1.369617)  # layer 1 neuron 3
    h1_4 = sigmoid(0.9959632 * x0 - 0.54454243)  # layer 1 neuron 4
    h1_5 = sigmoid(1.9069883 * x0 - 3.6568477)  # layer 1 neuron 5
    h1_6 = sigmoid(2.9722471 * x0 - 6.170906)  # layer 1 neuron 6
    h1_7 = sigmoid(0.26694876 * x0 + 0.96365565)  # layer 1 neuron 7
    return 1.8171854 * h1_0 - 1.4876635 * h1_1 + 2.2319286 * h1_2 + 2.025602 * h1_3 + 0.5370599 * h1_4 + 1.5314425 * h1_5 + 2.4661822 * h1_6 + 1.3693926 * h1_7 + 0.7184336
