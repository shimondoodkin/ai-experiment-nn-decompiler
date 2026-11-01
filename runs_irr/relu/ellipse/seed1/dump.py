# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = relu(1.0306978 * x0 - 4.2449617 * x1 - 1.0008848)  # layer 1 neuron 0
    h1_1 = relu(0.6780246 * x0 + 5.3779626 * x1 - 1.8844025)  # layer 1 neuron 1
    h1_2 = relu(-3.3637238 * x0 + 3.5659673 * x1 - 0.7839614)  # layer 1 neuron 2
    h1_3 = relu(-1.6467015 * x0 + 4.3892217 * x1 - 1.476804)  # layer 1 neuron 3
    h1_4 = relu(0.5525511 * x0 + 1.1125162 * x1 + 4.858137)  # layer 1 neuron 4
    h1_5 = relu(1.9325027 * x0 + 4.76246 * x1 - 2.153231)  # layer 1 neuron 5
    h1_6 = relu(0.42796734 * x0 + 0.6019861 * x1 + 4.589535)  # layer 1 neuron 6
    h1_7 = relu(0.001913525 * x0 - 0.010191 * x1 - 0.3820634)  # layer 1 neuron 7
    h1_8 = relu(0.10567373 * x0 - 3.5630221 * x1 - 1.11577)  # layer 1 neuron 8
    h1_9 = relu(3.2638638 * x0 + 1.1066182 * x1 - 2.0511668)  # layer 1 neuron 9
    h1_10 = relu(0.7118556 * x0 - 3.3381484 * x1 - 0.66384155)  # layer 1 neuron 10
    h1_11 = relu(2.889362 * x0 - 3.081175 * x1 - 3.995253)  # layer 1 neuron 11
    h1_12 = relu(-1.5259248 * x0 - 5.2080946 * x1 - 1.7093807)  # layer 1 neuron 12
    h1_13 = relu(0.41431051 * x0 + 0.7908961 * x1 + 4.782943)  # layer 1 neuron 13
    h1_14 = relu(-4.365142 * x0 + 0.3625458 * x1 - 1.5386657)  # layer 1 neuron 14
    h1_15 = relu(-0.109352656 * x0 + 0.40627587 * x1 + 1.2671258)  # layer 1 neuron 15
    return -2.7027214 * h1_0 - 5.695705 * h1_1 - 3.4200666 * h1_2 - 4.4392433 * h1_3 + 2.9768343 * h1_4 - 4.9300523 * h1_5 + 3.2818856 * h1_6 + 0.14727655 * h1_7 - 1.9581834 * h1_8 - 3.2219512 * h1_9 - 1.6291639 * h1_10 - 4.8608694 * h1_11 - 6.166774 * h1_12 + 2.750276 * h1_13 - 3.1198711 * h1_14 + 0.4919757 * h1_15 + 1.1883724
