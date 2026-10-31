# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(0.12840268 * x0 + 4.560615 * x1 - 8.199461)  # layer 1 neuron 0
    h1_1 = sigmoid(-4.0641055 * x0 - 2.3865604 * x1 - 8.488708)  # layer 1 neuron 1
    h1_2 = sigmoid(0.92876333 * x0 + 2.4956489 * x1 - 4.434684)  # layer 1 neuron 2
    h1_3 = sigmoid(0.05266533 * x0 - 0.025051028 * x1 - 1.2582273)  # layer 1 neuron 3
    h1_4 = sigmoid(1.5103494 * x0 - 3.2687922 * x1 - 6.1614885)  # layer 1 neuron 4
    h1_5 = sigmoid(-3.066173 * x0 - 0.83843225 * x1 - 5.3060956)  # layer 1 neuron 5
    h1_6 = sigmoid(-4.053963 * x0 + 2.2586155 * x1 + 8.493024)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.23983037 * x0 + 0.063540496 * x1 - 0.5731119)  # layer 1 neuron 7
    h1_8 = sigmoid(3.314013 * x0 + 3.6631455 * x1 - 8.662461)  # layer 1 neuron 8
    h1_9 = sigmoid(-4.428288 * x0 + 1.3207512 * x1 - 8.152605)  # layer 1 neuron 9
    h1_10 = sigmoid(4.5620728 * x0 + 0.78384304 * x1 - 8.246042)  # layer 1 neuron 10
    h1_11 = sigmoid(-2.8940406 * x0 + 3.5997796 * x1 - 8.098725)  # layer 1 neuron 11
    h1_12 = sigmoid(3.2996407 * x0 - 1.556768 * x1 - 6.419952)  # layer 1 neuron 12
    h1_13 = sigmoid(0.68435603 * x0 - 4.6609597 * x1 - 8.551302)  # layer 1 neuron 13
    h1_14 = sigmoid(-0.10397988 * x0 + 0.017922131 * x1 - 0.35761133)  # layer 1 neuron 14
    h1_15 = sigmoid(2.0728555 * x0 + 4.583634 * x1 + 8.915614)  # layer 1 neuron 15
    return -12.394402 * h1_0 - 12.306965 * h1_1 - 3.1052577 * h1_2 + 0.2076816 * h1_3 - 6.8339825 * h1_4 - 4.3691845 * h1_5 + 7.8995323 * h1_6 + 0.075801805 * h1_7 - 11.941002 * h1_8 - 11.680278 * h1_9 - 11.8110695 * h1_10 - 12.299683 * h1_11 - 6.633249 * h1_12 - 11.203834 * h1_13 + 0.13133219 * h1_14 + 8.673813 * h1_15 + 0.44388002
