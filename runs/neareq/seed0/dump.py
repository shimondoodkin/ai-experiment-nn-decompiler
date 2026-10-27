# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = relu(-3.7059202 * x0 + 3.7096167 * x1 - 0.26511204)  # layer 1 neuron 0
    h1_1 = relu(-1.691698 * x0 + 1.3458985 * x1 + 1.62702)  # layer 1 neuron 1
    h1_2 = relu(-0.13582842 * x0 + 0.050595496 * x1 - 0.6135267)  # layer 1 neuron 2
    h1_3 = relu(-3.2976904 * x0 + 3.3075483 * x1 - 0.24522409)  # layer 1 neuron 3
    h1_4 = relu(-3.5263665 * x0 + 3.5303345 * x1 - 0.2563369)  # layer 1 neuron 4
    h1_5 = relu(5.160562 * x0 - 5.1673737 * x1 - 0.26749894)  # layer 1 neuron 5
    h1_6 = relu(-1.4279348 * x0 + 0.96105784 * x1 + 1.458404)  # layer 1 neuron 6
    h1_7 = relu(-1.8917894 * x0 + 2.1996834 * x1 + 1.7307748)  # layer 1 neuron 7
    return -9.288122 * h1_0 + 1.5636504 * h1_1 + 0.19832896 * h1_2 - 9.440439 * h1_3 - 9.7136135 * h1_4 - 10.776145 * h1_5 + 1.2490262 * h1_6 + 3.661764 * h1_7 + 0.0032594635
