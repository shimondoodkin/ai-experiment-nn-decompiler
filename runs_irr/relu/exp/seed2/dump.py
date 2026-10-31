# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: regression: the returned value
def net(x0):
    h1_0 = relu(1.2392946 * x0 - 0.43504527)  # layer 1 neuron 0
    h1_1 = relu(-0.23797429 * x0 - 0.8853446)  # layer 1 neuron 1
    h1_2 = relu(1.4869899 * x0 - 2.3218477)  # layer 1 neuron 2
    h1_3 = relu(0.9745938 * x0 - 0.99909174)  # layer 1 neuron 3
    h1_4 = relu(0.1582815 * x0 - 0.47972798)  # layer 1 neuron 4
    h1_5 = relu(1.2276301 * x0 + 0.68755144)  # layer 1 neuron 5
    h1_6 = relu(-0.29512456 * x0 + 0.5911828)  # layer 1 neuron 6
    h1_7 = relu(0.4446606 * x0 + 0.92660517)  # layer 1 neuron 7
    return 0.88198656 * h1_0 - 0.20857231 * h1_1 + 1.5493966 * h1_2 + 1.6979808 * h1_3 - 0.12198506 * h1_4 + 0.5109522 * h1_5 - 0.33453166 * h1_6 + 0.4415071 * h1_7 + 0.46426988
