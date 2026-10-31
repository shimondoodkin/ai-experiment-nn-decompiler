# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(1.3923504 * x0 - 0.3075836)  # layer 1 neuron 0
    h1_1 = relu(-1.1365556 * x0 - 1.2506902)  # layer 1 neuron 1
    h1_2 = relu(-1.0221094 * x0 - 0.22327587)  # layer 1 neuron 2
    h1_3 = relu(1.1821759 * x0 - 0.7904091)  # layer 1 neuron 3
    h1_4 = relu(-1.6537173 * x0 - 1.089274)  # layer 1 neuron 4
    h1_5 = relu(0.9843397 * x0 - 1.0955)  # layer 1 neuron 5
    h1_6 = relu(-0.8348506 * x0 - 1.2980945)  # layer 1 neuron 6
    h1_7 = relu(1.0224231 * x0 - 1.5920001)  # layer 1 neuron 7
    return 0.6385939 * h1_0 + 0.79070234 * h1_1 + 0.85501325 * h1_2 + 0.7542424 * h1_3 + 0.53458554 * h1_4 + 0.899239 * h1_5 + 1.0724889 * h1_6 + 0.8676259 * h1_7 + 0.015702443
