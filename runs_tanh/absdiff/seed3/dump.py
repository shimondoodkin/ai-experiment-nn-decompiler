# dumped from a neural network: 2 inputs -> 8 -> 1 outputs
# inputs: x0: float [-3, 3], x1: float [-3, 3]
# output: regression: the returned value
def net(x0, x1):
    h1_0 = tanh(-0.10464291 * x0 - 0.10404467 * x1 - 1.3817129)  # layer 1 neuron 0
    h1_1 = tanh(-0.74761504 * x0 - 0.24674715 * x1 + 0.84667194)  # layer 1 neuron 1
    h1_2 = tanh(-0.0746237 * x0 - 0.0436356 * x1 + 1.5117095)  # layer 1 neuron 2
    h1_3 = tanh(-0.06571806 * x0 - 0.03659369 * x1 + 1.4994149)  # layer 1 neuron 3
    h1_4 = tanh(-0.367295 * x0 + 0.37508446 * x1 + 0.73922557)  # layer 1 neuron 4
    h1_5 = tanh(0.07978503 * x0 + 0.047539804 * x1 - 1.3018534)  # layer 1 neuron 5
    h1_6 = tanh(-0.07411262 * x0 - 0.043419957 * x1 + 1.48351)  # layer 1 neuron 6
    h1_7 = tanh(-0.38194802 * x0 + 0.37900206 * x1 - 0.7639359)  # layer 1 neuron 7
    return -1.4126209 * h1_0 - 0.12742522 * h1_1 + 1.0044427 * h1_2 + 1.0865715 * h1_3 - 3.932427 * h1_4 - 1.3433982 * h1_5 + 0.84948987 * h1_6 + 3.928685 * h1_7 + 0.487147
