# dumped from a neural network: 1 inputs -> 4 -> 1 outputs
# inputs: x0: float [-2, 2]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0):
    h1_0 = relu(0.42576495 * x0 - 1.0304978)  # layer 1 neuron 0
    h1_1 = relu(-8.027302 * x0 - 3.1986792)  # layer 1 neuron 1
    h1_2 = relu(-0.060116112 * x0 - 0.33587638)  # layer 1 neuron 2
    h1_3 = relu(8.0049095 * x0 - 3.0928082)  # layer 1 neuron 3
    return -0.017874625 * h1_0 - 6.301536 * h1_1 + 0.022655476 * h1_2 - 6.26673 * h1_3 + 5.5222554
