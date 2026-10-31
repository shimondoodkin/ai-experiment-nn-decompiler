# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = tanh(0.030752182 * x0 + 1.3063483)  # layer 1 neuron 0
    h1_1 = tanh(-0.030370729 * x0 - 1.3449345)  # layer 1 neuron 1
    h1_2 = tanh(0.035335325 * x0 + 0.99377817)  # layer 1 neuron 2
    h1_3 = tanh(-0.8240908 * x0 - 1.5172845)  # layer 1 neuron 3
    h1_4 = tanh(-0.87305576 * x0 + 1.5779971)  # layer 1 neuron 4
    h1_5 = tanh(-0.031309824 * x0 - 1.2761953)  # layer 1 neuron 5
    h1_6 = tanh(-0.03474916 * x0 - 1.0219247)  # layer 1 neuron 6
    h1_7 = tanh(-0.036464922 * x0 - 0.94594175)  # layer 1 neuron 7
    return 1.4319919 * h1_0 - 1.4084435 * h1_1 + 1.2698387 * h1_2 + 4.24856 * h1_3 - 3.7623935 * h1_4 - 1.3405845 * h1_5 - 1.4070268 * h1_6 - 1.5721523 * h1_7 + 0.5145114
