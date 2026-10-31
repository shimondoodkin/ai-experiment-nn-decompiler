# The network computes the product of its two inputs.
# Near the corners of the box (|x0*x1| close to 4) the net under-reaches by about 0.15,
# so the product is clipped at +-3.8 (the net gives about +-3.85 at the corners).
def net(x0, x1):
    product = x0 * x1
    return min(max(product, -3.8), 3.8)
