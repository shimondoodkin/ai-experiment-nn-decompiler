def net(x0):
    # Progressive "bracket" curve: marginal rate 0.10 from 0 to 20,
    # 0.20 from 20 to 80, and 0.40 above 80.
    band1 = relu(x0)
    band2 = relu(x0 - 20)
    band3 = relu(x0 - 80)
    return 0.1 * band1 + 0.1 * band2 + 0.2 * band3
