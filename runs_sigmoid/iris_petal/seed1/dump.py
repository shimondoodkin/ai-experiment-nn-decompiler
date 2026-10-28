# dumped from a neural network: 2 inputs -> 8 -> 3 outputs
# inputs: x0: float [1, 6.9], x1: float [0.1, 2.5]
# output: argmax over 3 outputs: the returned class index
def net(x0, x1):
    h1_0 = sigmoid(0.7549059 * x0 + 0.6395562 * x1 - 3.7435865)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.82442474 * x0 - 0.5011599 * x1 + 4.5566664)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.3280365 * x0 - 3.5211847 * x1 + 5.4725266)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.8734833 * x0 - 0.39364457 * x1 + 4.729426)  # layer 1 neuron 3
    h1_4 = sigmoid(1.4273874 * x0 + 2.5845761 * x1 - 5.1536922)  # layer 1 neuron 4
    h1_5 = sigmoid(0.77133584 * x0 + 1.146667 * x1 - 5.564307)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.60902447 * x0 - 1.3228148 * x1 + 4.8048167)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.71939653 * x0 - 0.82630813 * x1 + 3.402403)  # layer 1 neuron 7
    o0 = -3.6526325 * h1_0 + 2.3165064 * h1_1 + 3.1150281 * h1_2 + 2.386679 * h1_3 - 5.229155 * h1_4 - 2.6821926 * h1_5 + 2.1870697 * h1_6 + 3.0222208 * h1_7 - 0.34014016  # output 0
    o1 = -0.778316 * h1_0 + 2.4847753 * h1_1 - 6.076408 * h1_2 + 2.705798 * h1_3 + 1.8738666 * h1_4 - 3.8170388 * h1_5 + 3.5334003 * h1_6 + 0.39490387 * h1_7 - 0.5270266  # output 1
    o2 = 3.9806092 * h1_0 - 3.9905887 * h1_1 - 0.9188162 * h1_2 - 3.7627861 * h1_3 + 2.1525452 * h1_4 + 5.024216 * h1_5 - 3.7207181 * h1_6 - 3.1355655 * h1_7 + 0.043042753  # output 2
    return argmax(o0, o1, o2)
