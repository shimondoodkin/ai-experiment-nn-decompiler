# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0, 200]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(0.48298818 * x0 - 0.26447073)  # layer 1 neuron 0
    h1_1 = relu(-0.23797429 * x0 - 0.8853446)  # layer 1 neuron 1
    h1_2 = relu(0.08023032 * x0 - 1.7352921)  # layer 1 neuron 2
    h1_3 = relu(0.11441374 * x0 - 2.338938)  # layer 1 neuron 3
    h1_4 = relu(0.19610749 * x0 - 0.18457599)  # layer 1 neuron 4
    h1_5 = relu(0.35634863 * x0 - 0.028470363)  # layer 1 neuron 5
    h1_6 = relu(-0.022080746 * x0 + 4.607337)  # layer 1 neuron 6
    h1_7 = relu(-0.05077642 * x0 + 4.045834)  # layer 1 neuron 7
    return 0.39947325 * h1_0 - 0.20857231 * h1_1 + 0.12130789 * h1_2 + 0.6281135 * h1_3 - 0.10892576 * h1_4 + 0.20019028 * h1_5 - 3.468607 * h1_6 + 3.9647968 * h1_7 - 0.17204256
