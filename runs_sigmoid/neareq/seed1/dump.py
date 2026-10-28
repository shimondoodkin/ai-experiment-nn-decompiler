# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-1.8325301 * x0 + 0.37399748 * x1 - 0.9222973)  # layer 1 neuron 0
    h1_1 = sigmoid(-10.264247 * x0 + 10.244508 * x1 - 2.202513)  # layer 1 neuron 1
    h1_2 = sigmoid(-6.1676908 * x0 + 6.2344403 * x1 + 1.8394408)  # layer 1 neuron 2
    h1_3 = sigmoid(-11.394746 * x0 + 11.374804 * x1 - 2.3789897)  # layer 1 neuron 3
    h1_4 = sigmoid(-7.186394 * x0 + 7.168465 * x1 - 1.816756)  # layer 1 neuron 4
    h1_5 = sigmoid(10.800418 * x0 - 10.833183 * x1 - 2.3336902)  # layer 1 neuron 5
    h1_6 = sigmoid(11.521255 * x0 - 11.549395 * x1 - 2.4432764)  # layer 1 neuron 6
    h1_7 = sigmoid(0.7019879 * x0 - 0.62826186 * x1 + 2.4517581)  # layer 1 neuron 7
    return 0.48466286 * h1_0 - 8.489141 * h1_1 + 4.465596 * h1_2 - 9.370281 * h1_3 - 5.2992163 * h1_4 - 9.283582 * h1_5 - 9.997082 * h1_6 + 3.4543 * h1_7 + 3.1285763
