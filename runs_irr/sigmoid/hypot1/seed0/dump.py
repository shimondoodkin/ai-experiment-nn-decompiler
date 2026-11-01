# dumped from a neural network: 2 inputs -> 16 -> 1 outputs
# inputs: x0: float [-2, 2], x1: float [-2, 2]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(-0.15835306 * x0 + 1.3346913 * x1 - 2.6093683)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.29444826 * x0 - 1.9841287 * x1 - 3.5783312)  # layer 1 neuron 1
    h1_2 = sigmoid(-1.5528334 * x0 - 0.054349337 * x1 - 2.359362)  # layer 1 neuron 2
    h1_3 = sigmoid(0.27974242 * x0 + 1.4879214 * x1 - 2.0621226)  # layer 1 neuron 3
    h1_4 = sigmoid(1.9617058 * x0 + 0.16960378 * x1 - 3.2434027)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.17306565 * x0 + 0.021330344 * x1 + 0.5938289)  # layer 1 neuron 5
    h1_6 = sigmoid(-0.26180843 * x0 - 0.031166019 * x1 + 0.27025995)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.33971778 * x0 + 0.30511647 * x1 + 0.21066031)  # layer 1 neuron 7
    h1_8 = sigmoid(1.7060608 * x0 + 1.0923434 * x1 - 1.1231352)  # layer 1 neuron 8
    h1_9 = sigmoid(0.13993406 * x0 - 0.007218141 * x1 - 0.5474236)  # layer 1 neuron 9
    h1_10 = sigmoid(-0.21827243 * x0 + 0.7070259 * x1 + 0.047140297)  # layer 1 neuron 10
    h1_11 = sigmoid(-1.1474994 * x0 + 1.4732864 * x1 - 2.0796747)  # layer 1 neuron 11
    h1_12 = sigmoid(-0.75930345 * x0 - 1.9107319 * x1 - 1.4939833)  # layer 1 neuron 12
    h1_13 = sigmoid(1.1007509 * x0 - 1.4587042 * x1 - 2.040037)  # layer 1 neuron 13
    h1_14 = sigmoid(-2.9028373 * x0 - 0.24972521 * x1 + 1.496045)  # layer 1 neuron 14
    h1_15 = sigmoid(-0.2311766 * x0 - 0.011217961 * x1 + 0.4022038)  # layer 1 neuron 15
    return 0.62047356 * h1_0 + 0.94440335 * h1_1 + 1.5158104 * h1_2 + 1.0457752 * h1_3 + 1.3519261 * h1_4 + 0.13646632 * h1_5 + 0.26424226 * h1_6 - 0.0464681 * h1_7 + 0.22448027 * h1_8 - 0.05261967 * h1_9 - 0.052300505 * h1_10 + 0.66316974 * h1_11 + 0.6464509 * h1_12 + 0.74303067 * h1_13 - 0.23072964 * h1_14 + 0.17222853 * h1_15 + 0.2417272
