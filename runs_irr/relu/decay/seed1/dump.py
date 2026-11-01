# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-1, 3]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(0.13991325 * x0 + 1.3053523)  # layer 1 neuron 0
    h1_1 = relu(-1.6164408 * x0 - 0.35009834)  # layer 1 neuron 1
    h1_2 = relu(-0.4723221 * x0 + 0.79464525)  # layer 1 neuron 2
    h1_3 = relu(0.16644357 * x0 + 1.1464363)  # layer 1 neuron 3
    h1_4 = relu(-1.1474266 * x0 + 0.7467837)  # layer 1 neuron 4
    h1_5 = relu(0.9453335 * x0 + 0.19507821)  # layer 1 neuron 5
    h1_6 = relu(-0.082121246 * x0 - 0.19249131)  # layer 1 neuron 6
    h1_7 = relu(0.5608176 * x0 + 0.11534811)  # layer 1 neuron 7
    return 0.66786665 * h1_0 + 0.8456641 * h1_1 + 0.79080236 * h1_2 + 0.6886159 * h1_3 + 0.44539183 * h1_4 - 0.56824124 * h1_5 + 0.06613518 * h1_6 - 0.25428158 * h1_7 + 0.5254568
