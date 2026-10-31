# Refactored network: a round bump centred at the origin.
# Near formula (does NOT pass the checker on its own, max error ~0.035):
#     bump = exp(-(x0*x0 + x1*x1)) + 0.01
# The network approximates it with ReLU ridges; the program below keeps 35 of the
# 64 ridges (10 were dead, 19 more were removed by backward elimination) and refits
# the output weights (minimax LP) to stay within the snap tolerance of the net.
# h_k are ridge units relu(a*x0 + b*x1 + c); the return is their weighted sum
# plus a small linear term.
def net(x0, x1):
    h0 = relu(-0.239 * x0 + 0.473 * x1 + 0.64)
    h1 = relu(-0.518 * x0 + 0.509 * x1 - 0.808)
    h2 = relu(0.064 * x0 - 0.53 * x1 + 0.63)
    h3 = relu(0.691 * x0 + 0.118 * x1 + 0.03)
    h4 = relu(-0.333 * x0 - 0.536 * x1 - 0.768)
    h5 = relu(-0.099 * x0 - 0.594 * x1 - 0.726)
    h6 = relu(-0.458 * x0 - 0.001 * x1 - 0.546)
    h7 = relu(-0.392 * x0 + 0.733 * x1 - 0.189)
    h8 = relu(0.29 * x0 - 0.509 * x1 - 0.137)
    h9 = relu(0.672 * x0 - 0.278 * x1 + 0.095)
    h10 = relu(0.433 * x0 - 0.122 * x1 + 0.534)
    h11 = relu(-0.266 * x0 + 0.508 * x1 - 0.709)
    h12 = relu(-0.353 * x0 + 0.268 * x1 + 0.64)
    h13 = relu(-0.384 * x0 - 0.125 * x1 - 0.505)
    h14 = relu(-0.694 * x0 - 0.32 * x1 - 0.025)
    h15 = relu(0.558 * x0 - 0.169 * x1 - 0.686)
    h16 = relu(0.479 * x0 + 0.534 * x1 - 0.859)
    h17 = relu(-0.009 * x0 - 0.615 * x1 - 0.081)
    h18 = relu(0.469 * x0 + 0.284 * x1 - 0.696)
    h19 = relu(-0.494 * x0 + 0.285 * x1 - 0.701)
    h20 = relu(-0.52 * x0 - 0.392 * x1 - 0.024)
    h21 = relu(-0.575 * x0 + 0.444 * x1 + 0.006)
    h22 = relu(-0.578 * x0 - 0.418 * x1 - 0.862)
    h23 = relu(0.005 * x0 + 0.54 * x1 - 0.091)
    h24 = relu(-0.083 * x0 - 0.523 * x1 + 0.632)
    h25 = relu(0.678 * x0 + 0.01 * x1 - 0.801)
    h26 = relu(-0.38 * x0 - 0.601 * x1 + 0.086)
    h27 = relu(0.618 * x0 - 0.059 * x1 - 0.115)
    h28 = relu(0.176 * x0 + 0.558 * x1 - 0.078)
    h29 = relu(0.645 * x0 - 0.25 * x1 - 0.16)
    h30 = relu(-0.091 * x0 + 0.555 * x1 + 0.663)
    h31 = relu(0.342 * x0 + 0.648 * x1 - 0.897)
    h32 = relu(-0.422 * x0 - 0.663 * x1 - 0.131)
    h33 = relu(0.584 * x0 - 0.444 * x1 - 0.752)
    h34 = relu(-0.733 * x0 + 0.069 * x1 - 0.112)
    return sum((0.2592 * h0, 0.2118 * h1, 0.3161 * h2, -0.2792 * h3, 0.2539 * h4, 0.2036 * h5, 0.3409 * h6, -0.2714 * h7, -0.3685 * h8, -0.1923 * h9, 0.2055 * h10, 0.2406 * h11, 0.2087 * h12, 0.1912 * h13, -0.2897 * h14, 0.1275 * h15, 0.1426 * h16, -0.3063 * h17, 0.3502 * h18, 0.1815 * h19, -0.2123 * h20, -0.2697 * h21, 0.2922 * h22, -0.2825 * h23, 0.1939 * h24, 0.3092 * h25, -0.2260 * h26, -0.1635 * h27, -0.2278 * h28, -0.1991 * h29, 0.2713 * h30, 0.1770 * h31, -0.1927 * h32, 0.2372 * h33, -0.1420 * h34, 0.1580, -0.0105 * x0, -0.1817 * x1))
