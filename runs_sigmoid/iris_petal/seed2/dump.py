# dumped from a neural network: 2 inputs -> 8 -> 3 outputs
# inputs: x0: float [1, 6.9], x1: float [0.1, 2.5]
# output: argmax over 3 outputs: the returned class index
def net(x0, x1):
    h1_0 = sigmoid(-0.84715724 * x0 - 0.8084469 * x1 + 5.4324207)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.74336237 * x0 - 0.7039953 * x1 + 4.520175)  # layer 1 neuron 1
    h1_2 = sigmoid(0.8109278 * x0 + 3.7180254 * x1 - 4.6540127)  # layer 1 neuron 2
    h1_3 = sigmoid(-1.1383586 * x0 - 3.3893106 * x1 + 5.1617327)  # layer 1 neuron 3
    h1_4 = sigmoid(1.680434 * x0 + 2.552029 * x1 - 6.041179)  # layer 1 neuron 4
    h1_5 = sigmoid(0.8267112 * x0 + 0.82492334 * x1 - 5.3719106)  # layer 1 neuron 5
    h1_6 = sigmoid(0.70741904 * x0 + 0.9668199 * x1 - 4.8473077)  # layer 1 neuron 6
    h1_7 = sigmoid(0.75871783 * x0 + 1.0459445 * x1 - 5.413136)  # layer 1 neuron 7
    o0 = 2.4306648 * h1_0 + 3.0377429 * h1_1 - 2.9805567 * h1_2 + 2.5517929 * h1_3 - 3.831646 * h1_4 - 2.169625 * h1_5 - 2.3304102 * h1_6 - 1.6322078 * h1_7 + 1.6498932  # output 0
    o1 = 3.064072 * h1_0 + 2.9105508 * h1_1 + 1.9104679 * h1_2 - 5.3547773 * h1_3 + 2.779952 * h1_4 - 2.9832118 * h1_5 - 2.5431976 * h1_6 - 3.2075133 * h1_7 - 0.6533506  # output 1
    o2 = -4.1293664 * h1_0 - 4.537461 * h1_1 + 1.0265776 * h1_2 - 1.2210994 * h1_3 + 0.2721263 * h1_4 + 4.063673 * h1_5 + 3.7486837 * h1_6 + 3.4156926 * h1_7 - 0.8782417  # output 2
    return argmax(o0, o1, o2)
