def net(x0):
    x = x0
    # convex, decreasing: the larger of two lines that cross at x ~= 1.264
    steep = 2.48677 - 1.44732 * x    # wins for x < 1.264
    shallow = 0.87504 - 0.17217 * x  # wins for x > 1.264
    return max(steep, shallow)
