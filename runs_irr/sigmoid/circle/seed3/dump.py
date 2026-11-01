# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(-4.0475736 * x0 - 1.8084296 * x1 - 7.5682325)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.7770314 * x0 + 3.2965722 * x1 + 5.6111174)  # layer 1 neuron 1
    h1_2 = sigmoid(3.8015168 * x0 - 2.4302242 * x1 - 7.827055)  # layer 1 neuron 2
    h1_3 = sigmoid(4.3993983 * x0 - 0.20879018 * x1 - 7.6460247)  # layer 1 neuron 3
    h1_4 = sigmoid(2.5697253 * x0 + 3.390157 * x1 - 7.355537)  # layer 1 neuron 4
    h1_5 = sigmoid(0.12951757 * x0 + 2.0704777 * x1 - 3.1837997)  # layer 1 neuron 5
    h1_6 = sigmoid(-2.382271 * x0 + 3.4182737 * x1 + 7.117173)  # layer 1 neuron 6
    h1_7 = sigmoid(-4.4827623 * x0 + 0.2766603 * x1 - 7.7446156)  # layer 1 neuron 7
    h1_8 = sigmoid(-3.6565504 * x0 + 2.2978668 * x1 - 7.586228)  # layer 1 neuron 8
    h1_9 = sigmoid(3.0011227 * x0 + 2.9327445 * x1 + 7.259095)  # layer 1 neuron 9
    h1_10 = sigmoid(-3.7175605 * x0 - 2.1973104 * x1 + 7.57186)  # layer 1 neuron 10
    h1_11 = sigmoid(0.83746195 * x0 - 4.133986 * x1 - 7.3788915)  # layer 1 neuron 11
    h1_12 = sigmoid(-3.8163958 * x0 - 1.5636011 * x1 + 7.2417154)  # layer 1 neuron 12
    h1_13 = sigmoid(-2.1748104 * x0 + 3.9841368 * x1 - 7.895813)  # layer 1 neuron 13
    h1_14 = sigmoid(-1.5891173 * x0 - 4.018875 * x1 - 7.304275)  # layer 1 neuron 14
    h1_15 = sigmoid(0.30721253 * x0 + 4.217401 * x1 - 7.496564)  # layer 1 neuron 15
    return -10.552555 * h1_0 + 2.8198295 * h1_1 - 11.324681 * h1_2 - 10.783138 * h1_3 - 10.593536 * h1_4 - 2.9320877 * h1_5 + 4.4144397 * h1_6 - 10.231677 * h1_7 - 10.282677 * h1_8 + 5.270599 * h1_9 + 4.542904 * h1_10 - 10.700606 * h1_11 + 4.822475 * h1_12 - 10.938293 * h1_13 - 10.886815 * h1_14 - 10.664311 * h1_15 - 1.6629634
