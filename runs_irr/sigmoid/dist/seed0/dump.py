# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(1.338402 * x0 + 1.5051011 * x1 - 1.7216322)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.25255445 * x0 - 1.6286551 * x1 - 3.9664757)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.3425697 * x0 - 0.28416446 * x1 - 3.6019197)  # layer 1 neuron 2
    h1_3 = sigmoid(-0.8788884 * x0 + 1.6208663 * x1 - 1.634442)  # layer 1 neuron 3
    h1_4 = sigmoid(1.6095008 * x0 - 0.2557965 * x1 - 3.8343306)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.05361581 * x0 - 0.24785255 * x1 + 0.48045433)  # layer 1 neuron 5
    h1_6 = sigmoid(0.14616793 * x0 + 0.33014044 * x1 - 0.08011884)  # layer 1 neuron 6
    h1_7 = sigmoid(-3.3667593 * x0 + 0.37411827 * x1 + 2.2262666)  # layer 1 neuron 7
    h1_8 = sigmoid(1.9755992 * x0 - 1.177191 * x1 - 1.847091)  # layer 1 neuron 8
    h1_9 = sigmoid(-1.9476123 * x0 - 0.675243 * x1 - 1.4517537)  # layer 1 neuron 9
    h1_10 = sigmoid(0.21454698 * x0 + 0.66899 * x1 - 0.7276985)  # layer 1 neuron 10
    h1_11 = sigmoid(-0.27310735 * x0 + 1.7666255 * x1 - 4.4654875)  # layer 1 neuron 11
    h1_12 = sigmoid(-0.5015628 * x0 - 2.263171 * x1 - 1.5699258)  # layer 1 neuron 12
    h1_13 = sigmoid(-0.05277953 * x0 - 0.7031981 * x1 - 2.5242953)  # layer 1 neuron 13
    h1_14 = sigmoid(0.4421044 * x0 - 2.9238472 * x1 + 2.0514846)  # layer 1 neuron 14
    h1_15 = sigmoid(0.0073068636 * x0 - 0.043216962 * x1 + 0.24696128)  # layer 1 neuron 15
    return 0.90578526 * h1_0 + 1.2104527 * h1_1 + 1.919547 * h1_2 + 1.1024939 * h1_3 + 1.9521927 * h1_4 - 0.022555223 * h1_5 + 0.1482534 * h1_6 - 0.5192598 * h1_7 + 0.75408983 * h1_8 + 1.0873893 * h1_9 + 0.15714356 * h1_10 + 1.6022446 * h1_11 + 0.9382147 * h1_12 + 0.9884326 * h1_13 - 0.45752278 * h1_14 + 0.01085147 * h1_15 + 0.09725985
