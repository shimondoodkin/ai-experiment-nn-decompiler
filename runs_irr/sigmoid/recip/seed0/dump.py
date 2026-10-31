# dumped from a neural network: 1 inputs -> 8 -> 1 outputs
# inputs: x0: float [0.5, 4]
# output: regression: the returned value
def net(x0):
    h1_0 = sigmoid(-1.3396066 * x0 + 0.44495443)  # layer 1 neuron 0
    h1_1 = sigmoid(-0.19061977 * x0 + 0.7653571)  # layer 1 neuron 1
    h1_2 = sigmoid(-4.2219696 * x0 + 0.67134404)  # layer 1 neuron 2
    h1_3 = sigmoid(-6.330622 * x0 + 1.8502626)  # layer 1 neuron 3
    h1_4 = sigmoid(-3.5819921 * x0 + 0.5623914)  # layer 1 neuron 4
    h1_5 = sigmoid(-0.35206825 * x0 - 0.03855779)  # layer 1 neuron 5
    h1_6 = sigmoid(1.8969584 * x0 - 0.81798315)  # layer 1 neuron 6
    h1_7 = sigmoid(1.0297385 * x0 + 0.42698815)  # layer 1 neuron 7
    return 1.0799745 * h1_0 + 0.35679173 * h1_1 + 0.8946181 * h1_2 + 1.2964704 * h1_3 + 1.4553668 * h1_4 + 0.6967301 * h1_5 - 0.54160553 * h1_6 - 0.006209854 * h1_7 + 0.47697556
