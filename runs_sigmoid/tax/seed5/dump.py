# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(0.08169497 * x0 - 11.303006)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.08810667 * x0 + 15.51726)  # layer 1 neuron 1
    h1_2 = sigmoid(0.046411935 * x0 + 2.4653497)  # layer 1 neuron 2
    h1_3 = sigmoid(0.070200615 * x0 - 6.788423)  # layer 1 neuron 3
    h1_4 = sigmoid(0.08831822 * x0 - 15.62757)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.07465082 * x0 + 9.704204)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.10086226 * x0 + 9.484457)  # layer 1 neuron 6
    h1_7 = sigmoid(0.07273815 * x0 - 2.7647653)  # layer 1 neuron 7
    return 10.62973 * h1_0 - 7.9049172 * h1_1 + 9.958709 * h1_2 + 10.595965 * h1_3 + 10.447429 * h1_4 - 6.25257 * h1_5 - 6.210617 * h1_6 + 10.472487 * h1_7 + 10.608412
