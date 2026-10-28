def net(x0):
    # Progressive "bracket" schedule: marginal rate 0.1 below 20, 0.2 from 20 to 80, 0.4 above 80.
    bracket_1 = 0.1 * x0
    bracket_2 = 0.1 * relu(x0 - 20)
    bracket_3 = 0.2 * relu(x0 - 80)
    return bracket_1 + bracket_2 + bracket_3 - 0.2
