def net(x0, x1):
    # the network computes the product x0*x1; it saturates slightly toward the
    # corners of the box (|x0*x1| near 4), which the small cubic term models
    p = x0 * x1
    return p - p * p * p / 500
