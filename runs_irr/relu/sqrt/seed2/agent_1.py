# Concave piecewise-linear curve with 3 segments (knots at x0 ~ 0.416 and ~ 1.682).
# Its shape follows sqrt(x0) closely for x0 >= 1 (f(1)=0.98, f(2)=1.42, f(3)=1.72, f(4)=2.02),
# but starts at 0.175 instead of 0.
# Snapped variant (PASS snap, max err 0.025): min(1.2*x0 + 0.2, 0.5*x0 + 0.5, 0.3*x0 + 0.8)
def net(x0):
    steep = 1.2377267 * x0 + 0.17533842   # x0 < 0.416
    middle = 0.50264385 * x0 + 0.48120616  # 0.416 < x0 < 1.682
    flat = 0.30063863 * x0 + 0.82096849    # x0 > 1.682
    return min(steep, middle, flat)
