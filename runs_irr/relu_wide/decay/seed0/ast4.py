# produced by nndc ast (route: structural-snap)
def net(x0):
    return -0.06600640312780276 * x0 + 1.2562390120138407 + 0.15 * min(max(x0 + 0.2, 0), 1.25) + 0.3 * relu(-x0 - 0.75) + 0.1 * relu(-x0 + 1.6) + 0.28 * relu(-x0 + 0.4) + 0.28 * relu(-x0 - 0.4) + 0.1 * relu(-x0 + 2.3) + 0.0151197484400732 * relu(-x0 + 2.5) - 0.0644266890129887 * relu(x0 - 0.4) + 0.3 * relu(-x0 + 1.05) + 0.1 * relu(-x0 + 1.3) - 0.18 * relu(x0 - 0.75) + 0.1 * relu(-x0 + 1.95) + 0.24 * relu(-x0 + 0.05) + 0.35 * relu(-x0 + 0.75) + 0.1 * relu(-x0 + 2.8)
