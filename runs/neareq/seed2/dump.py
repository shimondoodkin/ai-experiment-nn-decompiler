# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = relu(3.4995902 * x0 - 3.5024092 * x1 - 0.357402)  # layer 1 neuron 0
    h1_1 = relu(0.13386506 * x0 + 0.023924395 * x1 - 0.47717747)  # layer 1 neuron 1
    h1_2 = relu(0.9971862 * x0 - 0.07908688 * x1 + 1.4944394)  # layer 1 neuron 2
    h1_3 = relu(1.0029399 * x0 - 1.3183084 * x1 + 2.7881167)  # layer 1 neuron 3
    h1_4 = relu(3.4161203 * x0 - 3.4211483 * x1 - 0.3481083)  # layer 1 neuron 4
    h1_5 = relu(3.1720812 * x0 - 3.1756787 * x1 - 0.33471)  # layer 1 neuron 5
    h1_6 = relu(-3.9738536 * x0 + 4.000095 * x1 - 0.35588154)  # layer 1 neuron 6
    h1_7 = relu(-4.09036 * x0 + 4.1099324 * x1 - 0.37514275)  # layer 1 neuron 7
    return -9.06185 * h1_0 + 0.033111986 * h1_1 + 1.0269562 * h1_2 + 2.2614715 * h1_3 - 9.315151 * h1_4 - 10.154277 * h1_5 - 8.680973 * h1_6 - 10.320841 * h1_7 + 1.3724438
