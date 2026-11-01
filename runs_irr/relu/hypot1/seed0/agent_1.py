# The network approximates  y = sqrt(1 + x0*x0 + x1*x1)  (a hyperboloid: 1 at the origin, 3 at the
# corners).  The formula itself misses the net by up to 0.055 (the net's piecewise-linear dips, e.g.
# near (0.5, 0.19)), which is above the 0.041 snap tolerance, so this program keeps the net's form:
# a convex sum of hinges (every output weight is +1) with 10 refitted, snapped creases.
# Original: 16 units; 3 were dead on the box (never positive), 13 live -> refitted down to 10.
def net(x0, x1):
    sw_a = relu(-0.155 * x0 - 0.195 * x1 - 0.03)   # rises towards (-,-)
    sw_b = relu(-0.165 * x0 - 0.215 * x1 + 0.135)  # same direction, crease further out
    sw_c = relu(-0.14 * x0 - 0.14 * x1 - 0.17)     # (-,-) corner
    west = relu(-0.31 * x0 + 0.11 * x1 - 0.085)
    north = relu(-0.1 * x0 + 0.43 * x1 + 0.01)
    north_b = relu(-0.055 * x0 + 0.245 * x1 + 0.22)
    south = relu(0.06 * x0 - 0.225 * x1 + 0.175)
    east_a = relu(0.17 * x0 - 0.035 * x1 + 0.17)
    east_b = relu(0.28 * x0 + 0.105 * x1 - 0.04)
    east_c = relu(0.35 * x0 - 0.12 * x1 - 0.17)
    return 0.275 + sw_a + sw_b + sw_c + west + north + north_b + south + east_a + east_b + east_c
