# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(-1.9002496 * x0 + 0.35516945 * x1 - 4.95011)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.6003895 * x0 - 1.1876833 * x1 - 3.1551)  # layer 1 neuron 1
    h1_2 = sigmoid(0.55124426 * x0 - 1.2754723 * x1 - 3.3186407)  # layer 1 neuron 2
    h1_3 = sigmoid(2.4467022 * x0 + 0.15896057 * x1 - 2.281302)  # layer 1 neuron 3
    h1_4 = sigmoid(0.039839443 * x0 + 3.118136 * x1 - 1.9440912)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.19654003 * x0 + 0.9086084 * x1 - 2.5969725)  # layer 1 neuron 5
    h1_6 = sigmoid(-2.3372712 * x0 + 0.37815642 * x1 - 2.1718638)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.13173836 * x0 - 0.6518959 * x1 - 0.44183227)  # layer 1 neuron 7
    h1_8 = sigmoid(0.14117345 * x0 + 1.4011763 * x1 - 4.1213317)  # layer 1 neuron 8
    h1_9 = sigmoid(2.3822255 * x0 + 1.3313328 * x1 - 1.9828248)  # layer 1 neuron 9
    h1_10 = sigmoid(1.323229 * x0 - 2.1686845 * x1 - 1.7248331)  # layer 1 neuron 10
    h1_11 = sigmoid(2.1009908 * x0 + 0.31237328 * x1 - 5.4113593)  # layer 1 neuron 11
    h1_12 = sigmoid(-0.01754814 * x0 - 0.11448232 * x1 + 0.20027602)  # layer 1 neuron 12
    h1_13 = sigmoid(-2.680485 * x0 + 1.823558 * x1 - 2.1262555)  # layer 1 neuron 13
    h1_14 = sigmoid(-1.3959846 * x0 - 2.1128883 * x1 - 1.7265438)  # layer 1 neuron 14
    h1_15 = sigmoid(0.050148763 * x0 + 1.8665044 * x1 - 3.502957)  # layer 1 neuron 15
    return 1.4030745 * h1_0 + 1.4958715 * h1_1 + 1.4072115 * h1_2 + 0.74251896 * h1_3 + 0.63782895 * h1_4 + 0.9445927 * h1_5 + 0.8103372 * h1_6 + 0.0057395203 * h1_7 + 0.70888233 * h1_8 + 0.61481017 * h1_9 + 0.79268193 * h1_10 + 1.3640252 * h1_11 - 0.067667164 * h1_12 + 0.4162148 * h1_13 + 0.80159885 * h1_14 + 0.6942913 * h1_15 - 0.46157187
