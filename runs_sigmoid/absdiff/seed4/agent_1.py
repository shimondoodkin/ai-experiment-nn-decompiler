# A smoothed |x0 - x1|: the net only looks at the difference d = x0 - x1.
# Three soft steps in d build the "V":
#   rise_pos : turns on for d > 2.4 (right arm),
#   rise_neg : turns on for d < -1  (left arm, inner part),
#   rise_far : turns on for d < -3.6 (left arm, outer part),
# plus a small slope and an offset.
def net(x0, x1):
    d = x0 - x1
    rise_pos = sigmoid(0.7 * (d - 2.4))
    rise_neg = sigmoid(-2.5 * (d + 1))
    rise_far = sigmoid(-1.2 * (d + 3.6))
    return 6 * rise_pos + 2.5 * rise_neg + 4.5 * rise_far + 0.1 * d - 0.9
