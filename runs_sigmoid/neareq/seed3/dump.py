# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-10.172236 * x0 + 10.182916 * x1 + 2.1321692)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.47127149 * x0 - 0.1747331 * x1 + 3.8183677)  # layer 1 neuron 1
    h1_2 = sigmoid(-0.27824083 * x0 + 0.049835548 * x1 + 3.598747)  # layer 1 neuron 2
    h1_3 = sigmoid(-10.726231 * x0 + 10.734161 * x1 + 2.2183373)  # layer 1 neuron 3
    h1_4 = sigmoid(-0.034101043 * x0 + 0.28647664 * x1 + 3.022868)  # layer 1 neuron 4
    h1_5 = sigmoid(-11.141266 * x0 + 11.117323 * x1 - 2.2133014)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.35331345 * x0 - 0.041835718 * x1 + 3.711267)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.40658635 * x0 - 0.51572174 * x1 - 3.3767052)  # layer 1 neuron 7
    return 8.896664 * h1_0 - 2.4128716 * h1_1 - 2.1298506 * h1_2 + 10.579943 * h1_3 - 2.0974789 * h1_4 - 18.226831 * h1_5 - 2.2201657 * h1_6 + 1.0689062 * h1_7 - 1.4543873
