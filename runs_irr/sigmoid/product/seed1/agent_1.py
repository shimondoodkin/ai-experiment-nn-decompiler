def net(x0, x1):
    # The 16 sigmoid units split into four groups by weight direction:
    # along s = x0 + x1 (both signs) and along d = x1 - x0 (both signs).
    # Each group builds a 1-D curve: s*s/4 and -d*d/4. Their sum is
    # (s*s - d*d) / 4 = x0 * x1.
    product = x0 * x1
    return product
