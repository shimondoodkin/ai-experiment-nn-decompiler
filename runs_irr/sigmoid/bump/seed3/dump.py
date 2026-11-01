# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(-2.4795656 * x0 - 0.6495197 * x1 - 1.3056657)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.93088996 * x0 - 1.0379755 * x1 - 0.9761138)  # layer 1 neuron 1
    h1_2 = sigmoid(-0.72048324 * x0 - 1.6457499 * x1 - 2.5294807)  # layer 1 neuron 2
    h1_3 = sigmoid(3.3480728 * x0 + 0.26068023 * x1 - 1.7778476)  # layer 1 neuron 3
    h1_4 = sigmoid(0.804156 * x0 + 2.7423267 * x1 - 1.4362539)  # layer 1 neuron 4
    h1_5 = sigmoid(1.2289999 * x0 + 0.6760191 * x1 - 1.2891471)  # layer 1 neuron 5
    h1_6 = sigmoid(-1.5518035 * x0 + 0.36565232 * x1 - 2.149082)  # layer 1 neuron 6
    h1_7 = sigmoid(-2.8152463 * x0 + 0.6391355 * x1 + 1.4938952)  # layer 1 neuron 7
    h1_8 = sigmoid(-1.2026144 * x0 + 2.6421423 * x1 - 1.4995772)  # layer 1 neuron 8
    h1_9 = sigmoid(-0.23316157 * x0 + 2.7601225 * x1 + 1.4189944)  # layer 1 neuron 9
    h1_10 = sigmoid(0.96959096 * x0 - 1.0532783 * x1 - 1.5376366)  # layer 1 neuron 10
    h1_11 = sigmoid(1.9791874 * x0 - 2.1512294 * x1 - 1.5275532)  # layer 1 neuron 11
    h1_12 = sigmoid(-2.3592849 * x0 - 1.6416007 * x1 + 1.56187)  # layer 1 neuron 12
    h1_13 = sigmoid(-0.25783294 * x0 + 1.4873906 * x1 - 1.9329643)  # layer 1 neuron 13
    h1_14 = sigmoid(-1.667592 * x0 - 2.2394047 * x1 - 1.4998761)  # layer 1 neuron 14
    h1_15 = sigmoid(-2.2797174 * x0 + 1.2407162 * x1 - 1.3802462)  # layer 1 neuron 15
    return -0.694553 * h1_0 + 0.34779182 * h1_1 + 0.26815227 * h1_2 - 0.2601531 * h1_3 - 0.64969623 * h1_4 + 0.6498866 * h1_5 + 0.48593345 * h1_6 + 0.514059 * h1_7 - 0.58795494 * h1_8 + 0.6888405 * h1_9 + 0.50183743 * h1_10 - 0.58268493 * h1_11 + 0.6683755 * h1_12 + 0.40906543 * h1_13 - 0.6714155 * h1_14 - 0.68263006 * h1_15 - 0.19284391
