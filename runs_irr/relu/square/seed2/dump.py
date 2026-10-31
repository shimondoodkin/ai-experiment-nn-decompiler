# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(1.1363467 * x0 - 0.9499765)  # layer 1 neuron 0
    h1_1 = relu(-0.23797429 * x0 - 0.8853446)  # layer 1 neuron 1
    h1_2 = relu(1.1396632 * x0 - 1.6259981)  # layer 1 neuron 2
    h1_3 = relu(-0.8139604 * x0 - 0.6863046)  # layer 1 neuron 3
    h1_4 = relu(0.15933335 * x0 - 0.48090324)  # layer 1 neuron 4
    h1_5 = relu(1.7114754 * x0 + 0.47300565)  # layer 1 neuron 5
    h1_6 = relu(-0.9714322 * x0 - 1.3819113)  # layer 1 neuron 6
    h1_7 = relu(-1.5099874 * x0 + 0.40735248)  # layer 1 neuron 7
    return 1.0254779 * h1_0 - 0.20857231 * h1_1 + 1.0197287 * h1_2 + 1.411795 * h1_3 - 0.12329065 * h1_4 + 0.6421177 * h1_5 + 1.1932168 * h1_6 + 0.7329579 * h1_7 - 0.58232695
