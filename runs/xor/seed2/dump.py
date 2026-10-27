# dumped from a neural network: 2 inputs -> 4 -> 1 outputs
# inputs: x0: bool [0, 1], x1: bool [0, 1]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = relu(-0.1083648 * x0 + 0.015899912 * x1 + 0.26979038)  # layer 1 neuron 0
    h1_1 = relu(0.19390899 * x0 - 0.036117494 * x1 - 0.6260332)  # layer 1 neuron 1
    h1_2 = relu(-0.35209712 * x0 + 0.21288416 * x1 - 0.41265488)  # layer 1 neuron 2
    h1_3 = relu(-0.08127111 * x0 - 0.57166946 * x1 - 0.2532823)  # layer 1 neuron 3
    return 0.000004500076 * h1_0 + 0.40884846 * h1_1 - 0.30302483 * h1_2 - 0.09293491 * h1_3 - 0.000001158528
