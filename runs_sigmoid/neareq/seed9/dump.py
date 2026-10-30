# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-0.33347517 * x0 + 0.22444178 * x1 + 3.5194395)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.45841488 * x0 + 0.5334192 * x1 + 1.4925995)  # layer 1 neuron 1
    h1_2 = sigmoid(0.36388057 * x0 - 0.19119705 * x1 - 2.607877)  # layer 1 neuron 2
    h1_3 = sigmoid(-13.527775 * x0 + 13.52902 * x1 - 2.6185646)  # layer 1 neuron 3
    h1_4 = sigmoid(-11.019744 * x0 + 11.026737 * x1 + 2.3644586)  # layer 1 neuron 4
    h1_5 = sigmoid(0.6542334 * x0 + 0.84546506 * x1 + 5.18384)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.9159713 * x0 - 0.49032104 * x1 + 4.9614224)  # layer 1 neuron 6
    h1_7 = sigmoid(-11.159713 * x0 + 11.166814 * x1 + 2.3864093)  # layer 1 neuron 7
    return -1.9997023 * h1_0 - 1.8846822 * h1_1 + 0.49347168 * h1_2 - 13.838913 * h1_3 + 8.327643 * h1_4 - 2.4042273 * h1_5 - 2.1246114 * h1_6 + 7.6900096 * h1_7 - 0.71303034
