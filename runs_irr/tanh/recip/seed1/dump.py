# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(1.0603207 * x0 + 1.3294913)  # layer 1 neuron 0
    h1_1 = tanh(-1.0665507 * x0 - 1.4699317)  # layer 1 neuron 1
    h1_2 = tanh(-0.09263142 * x0 + 0.6595609)  # layer 1 neuron 2
    h1_3 = tanh(0.84392816 * x0 + 1.0787147)  # layer 1 neuron 3
    h1_4 = tanh(-1.629322 * x0 - 0.7896506)  # layer 1 neuron 4
    h1_5 = tanh(2.68601 * x0 - 0.984565)  # layer 1 neuron 5
    h1_6 = tanh(-0.0069315885 * x0 - 0.8529525)  # layer 1 neuron 6
    h1_7 = tanh(0.9374267 * x0 - 0.7044269)  # layer 1 neuron 7
    return 0.34915835 * h1_0 - 0.31111592 * h1_1 + 0.9272115 * h1_2 + 0.34180447 * h1_3 - 0.2896097 * h1_4 - 1.1988087 * h1_5 - 0.33385745 * h1_6 - 0.6402057 * h1_7 + 0.29288945
