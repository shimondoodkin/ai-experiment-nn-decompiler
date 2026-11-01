def net(x0):
    return min(exp(x0), (exp((abs((x0 + 0.8612377)) * x0)) + -0.1256381))
