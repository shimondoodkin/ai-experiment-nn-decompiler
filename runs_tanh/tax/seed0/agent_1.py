def net(x0):
    # Convex, piecewise-linear "bracket" curve: marginal rate 0.15 up to 50,
    # 0.25 from 50 to 90, 0.40 above 90 (continuous at both breakpoints).
    rate_low = 0.15 * x0
    rate_mid = 0.25 * x0 - 5
    rate_high = 0.4 * x0 - 18.5
    return max(rate_low, rate_mid, rate_high)
