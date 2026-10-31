# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: binary: class 1 if the returned value > 0, else class 0
def net(x0, x1):
    h1_0 = sigmoid(4.226964 * x0 - 2.7682724 * x1 - 8.973224)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.03988297 * x0 - 0.039530333 * x1 + 1.7253036)  # layer 1 neuron 1
    h1_2 = sigmoid(4.5750866 * x0 + 1.2771949 * x1 - 8.499113)  # layer 1 neuron 2
    h1_3 = sigmoid(2.1640625 * x0 - 4.3360395 * x1 - 8.560533)  # layer 1 neuron 3
    h1_4 = sigmoid(4.4843507 * x0 - 1.8575772 * x1 + 8.6395)  # layer 1 neuron 4
    h1_5 = sigmoid(2.8757253 * x0 + 2.3405945 * x1 - 6.4551177)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.18300422 * x0 + 4.8389883 * x1 - 8.528423)  # layer 1 neuron 6
    h1_7 = sigmoid(2.2957246 * x0 + 4.1180806 * x1 - 8.449522)  # layer 1 neuron 7
    h1_8 = sigmoid(-1.8297838 * x0 - 4.2878976 * x1 - 8.233515)  # layer 1 neuron 8
    h1_9 = sigmoid(-2.8604052 * x0 + 3.9133906 * x1 - 8.681395)  # layer 1 neuron 9
    h1_10 = sigmoid(-4.9680486 * x0 - 0.50501925 * x1 - 8.824213)  # layer 1 neuron 10
    h1_11 = sigmoid(0.055352427 * x0 + 4.5646048 * x1 + 8.013204)  # layer 1 neuron 11
    h1_12 = sigmoid(-2.8832548 * x0 + 1.3606155 * x1 - 5.2410965)  # layer 1 neuron 12
    h1_13 = sigmoid(-0.04308222 * x0 - 0.044697028 * x1 + 1.4315155)  # layer 1 neuron 13
    h1_14 = sigmoid(-4.09773 * x0 - 2.974679 * x1 - 8.822608)  # layer 1 neuron 14
    h1_15 = sigmoid(-4.5680165 * x0 + 0.66022164 * x1 + 8.166615)  # layer 1 neuron 15
    return -9.349465 * h1_0 - 0.65627223 * h1_1 - 9.018883 * h1_2 - 10.270848 * h1_3 + 6.047729 * h1_4 - 6.0434694 * h1_5 - 9.923604 * h1_6 - 9.9696455 * h1_7 - 9.880346 * h1_8 - 10.472488 * h1_9 - 9.141608 * h1_10 + 5.929013 * h1_11 - 4.0538154 * h1_12 - 0.8825413 * h1_13 - 9.428269 * h1_14 + 6.1378016 * h1_15 - 0.4697963
