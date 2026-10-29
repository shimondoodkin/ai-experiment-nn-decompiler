# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = sigmoid(1.5608935 * x0 - 1.5811212 * x1 - 5.6978836)  # layer 1 neuron 0
    h1_1 = sigmoid(-1.6999716 * x0 - 1.4470838 * x1 + 1.0432267)  # layer 1 neuron 1
    h1_2 = sigmoid(-2.2282135 * x0 - 1.8396368 * x1 + 0.3947982)  # layer 1 neuron 2
    h1_3 = sigmoid(1.7599554 * x0 + 1.9408717 * x1 + 1.062507)  # layer 1 neuron 3
    h1_4 = sigmoid(1.2617681 * x0 - 1.2578588 * x1 + 1.5280614)  # layer 1 neuron 4
    h1_5 = sigmoid(2.485814 * x0 - 2.508978 * x1 - 2.8118944)  # layer 1 neuron 5
    h1_6 = sigmoid(-1.8447071 * x0 - 1.6160655 * x1 + 0.8440864)  # layer 1 neuron 6
    h1_7 = sigmoid(-0.8145771 * x0 + 0.8111943 * x1 - 2.0779521)  # layer 1 neuron 7
    return 3.3133814 * h1_0 - 0.009013219 * h1_1 + 0.16399612 * h1_2 + 0.2512382 * h1_3 - 0.55989116 * h1_4 + 2.4631565 * h1_5 + 0.0653747 * h1_6 + 5.2271266 * h1_7 - 0.19855295
