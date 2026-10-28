# Smoothed "median of three"-like function, symmetric under swapping x0 <-> x1.
# The 16x16 sigmoid net collapses to: 5 first-layer "ordering" detectors + a linear
# trend, fed into ONE second-layer sigmoid (all 16 layer-2 units were sign-flipped copies).
def net(x0, x1, x2):
    # x0 / x2 vs x1 ordering: "is x0 (resp. x1) above the other two", tilted toward x2
    x0_high = sigmoid(1.6 * x0 - 0.6 * x1 - x2) - 0.5
    x1_high = sigmoid(1.6 * x1 - 0.6 * x0 - x2) - 0.5
    # gate on x1 vs x0 (steep, slope 2), offset +-0.1*(x0 + x1 - 2*x2): a bump around x0 == x1
    near_tie = sigmoid(-1.9 * x0 + 2.1 * x1 - 0.2 * x2) - sigmoid(-2.1 * x0 + 1.9 * x1 + 0.2 * x2)
    # is x2 above the mean of x0, x1
    x2_high = sigmoid(0.64 * (2 * x2 - x0 - x1)) - 0.5
    z = 0.56 * (x0_high + x1_high) + 0.6 * near_tie + x2_high + 0.11 * (x0 + x1) + 0.07 * x2
    return 14 * (sigmoid(z) - 0.5)
