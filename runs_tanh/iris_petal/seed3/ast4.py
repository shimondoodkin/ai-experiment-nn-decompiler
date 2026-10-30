# produced by nndc ast (route: linear-scores)
def net(x0, x1):
    s1 = 0.17716701872794458 * x0 + 0.6546001319098672 * x1 - 0.9097702857317229
    s2 = 0.49505853371655384 * x0 + x1 - 3.068853268948265
    return argmax(0, s1, s2)
