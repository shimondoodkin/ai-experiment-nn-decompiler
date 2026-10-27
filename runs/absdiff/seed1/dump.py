# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = relu(0.44902217 * x0 - 0.4487421 * x1 - 0.0054444824)  # layer 1 neuron 0
    h1_1 = relu(-0.78232116 * x0 + 0.78240037 * x1 - 0.012242669)  # layer 1 neuron 1
    h1_2 = relu(-0.934785 * x0 + 0.9347525 * x1 - 0.023237117)  # layer 1 neuron 2
    h1_3 = relu(-0.6592352 * x0 + 0.65915895 * x1 - 0.013655376)  # layer 1 neuron 3
    h1_4 = relu(-0.0066279355 * x0 + 0.017997997 * x1 - 0.36617988)  # layer 1 neuron 4
    h1_5 = relu(0.57698005 * x0 - 0.5770887 * x1 - 0.012984348)  # layer 1 neuron 5
    h1_6 = relu(0.77456987 * x0 - 0.77455217 * x1 - 0.01622216)  # layer 1 neuron 6
    h1_7 = relu(0.008500496 * x0 - 0.0036041962 * x1 - 0.36505806)  # layer 1 neuron 7
    return 0.2625966 * h1_0 + 0.41048792 * h1_1 + 0.39479586 * h1_2 + 0.47001666 * h1_3 - 0.24836557 * h1_4 + 0.7760664 * h1_5 + 0.56073534 * h1_6 + 0.27534148 * h1_7 + 0.020548318
