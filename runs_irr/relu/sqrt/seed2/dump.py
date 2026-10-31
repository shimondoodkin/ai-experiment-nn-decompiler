# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(0.5652518 * x0 + 0.3158782)  # layer 1 neuron 0
    h1_1 = relu(-0.23797429 * x0 - 0.8853446)  # layer 1 neuron 1
    h1_2 = relu(0.13071 * x0 + 0.0017984554)  # layer 1 neuron 2
    h1_3 = relu(0.26506138 * x0 + 0.1718815)  # layer 1 neuron 3
    h1_4 = relu(0.40265375 * x0 - 0.6772428)  # layer 1 neuron 4
    h1_5 = relu(0.49857503 * x0 + 0.6773505)  # layer 1 neuron 5
    h1_6 = relu(-0.19523054 * x0 - 0.031651802)  # layer 1 neuron 6
    h1_7 = relu(-1.2473084 * x0 + 0.51900464)  # layer 1 neuron 7
    return 0.42075104 * h1_0 - 0.20857231 * h1_1 + 0.09809352 * h1_2 + 0.5885587 * h1_3 - 0.50168467 * h1_4 + 0.19252384 * h1_5 + 0.051107444 * h1_6 - 0.58933526 * h1_7 + 0.11655519
