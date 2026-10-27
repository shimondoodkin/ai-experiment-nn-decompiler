# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = relu(3.4741066 * x0 - 3.4732788 * x1 - 0.3514232)  # layer 1 neuron 0
    h1_1 = relu(-4.277891 * x0 + 4.272618 * x1 - 0.40335733)  # layer 1 neuron 1
    h1_2 = relu(-3.8758993 * x0 + 3.8697417 * x1 - 0.38936132)  # layer 1 neuron 2
    h1_3 = relu(-3.9580228 * x0 + 3.952065 * x1 - 0.40076992)  # layer 1 neuron 3
    h1_4 = relu(3.2914214 * x0 - 3.2902577 * x1 - 0.33740592)  # layer 1 neuron 4
    h1_5 = relu(0.55146116 * x0 - 0.54878926 * x1 + 2.6810098)  # layer 1 neuron 5
    h1_6 = relu(3.6162682 * x0 - 3.6150029 * x1 - 0.3641312)  # layer 1 neuron 6
    h1_7 = relu(-0.051554263 * x0 - 0.06365895 * x1 - 0.3050039)  # layer 1 neuron 7
    return -5.871051 * h1_0 - 7.1336207 * h1_1 - 4.15549 * h1_2 - 8.786397 * h1_3 - 9.063077 * h1_4 + 2.6034918 * h1_5 - 11.070301 * h1_6 + 0.33536497 * h1_7 + 1.5788696
