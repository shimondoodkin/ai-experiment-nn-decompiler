# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.1, 5]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(0.72364986 * x0 - 0.070895195)  # layer 1 neuron 0
    h1_1 = relu(-0.23797429 * x0 - 0.8853446)  # layer 1 neuron 1
    h1_2 = relu(0.22359803 * x0 - 0.021582657)  # layer 1 neuron 2
    h1_3 = relu(-0.1791192 * x0 - 0.066108674)  # layer 1 neuron 3
    h1_4 = relu(0.46323577 * x0 - 1.1341513)  # layer 1 neuron 4
    h1_5 = relu(0.7550957 * x0 - 0.07557729)  # layer 1 neuron 5
    h1_6 = relu(-0.86116976 * x0 + 0.9095043)  # layer 1 neuron 6
    h1_7 = relu(-2.297982 * x0 + 0.85774666)  # layer 1 neuron 7
    return 0.53636926 * h1_0 - 0.20857231 * h1_1 + 0.14196928 * h1_2 + 0.22995548 * h1_3 - 0.68305874 * h1_4 + 0.23095874 * h1_5 - 1.0303606 * h1_6 - 1.3365223 * h1_7 - 0.4487619
