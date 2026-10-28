# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = sigmoid(12.95493 * x0 - 6.4181976)  # layer 1 neuron 0
    h1_1 = sigmoid(-10.43261 * x0 + 5.130853)  # layer 1 neuron 1
    h1_2 = sigmoid(-13.306154 * x0 - 6.695987)  # layer 1 neuron 2
    h1_3 = sigmoid(11.634445 * x0 + 5.8530364)  # layer 1 neuron 3
    return -11.078198 * h1_0 + 3.0278273 * h1_1 - 9.545952 * h1_2 + 4.409145 * h1_3 - 0.28958982
