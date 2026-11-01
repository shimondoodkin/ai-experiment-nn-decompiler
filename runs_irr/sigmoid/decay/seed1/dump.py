# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-1, 3]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-0.8358407 * x0 + 0.88960004)  # layer 1 neuron 0
    h1_1 = sigmoid(-1.6708902 * x0 - 0.6837812)  # layer 1 neuron 1
    h1_2 = sigmoid(-0.87080616 * x0 + 0.7327382)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.8333056 * x0 + 0.7398383)  # layer 1 neuron 3
    h1_4 = sigmoid(-3.535988 * x0 - 3.4655018)  # layer 1 neuron 4
    h1_5 = sigmoid(-1.1983438 * x0 - 0.45418638)  # layer 1 neuron 5
    h1_6 = sigmoid(-1.9139422 * x0 - 1.3401915)  # layer 1 neuron 6
    h1_7 = sigmoid(-1.0859107 * x0 - 0.14116855)  # layer 1 neuron 7
    return 0.7016472 * h1_0 + 0.94289285 * h1_1 + 1.0780284 * h1_2 + 0.7660089 * h1_3 + 1.020655 * h1_4 + 0.4419953 * h1_5 + 0.922835 * h1_6 + 0.57085335 * h1_7 + 0.277976
