# produced by nndc ast (route: linear-scores)
def net(x0, x1):
    s1 = 0.2184375045579381 * x0 + 0.5707782722726646 * x1 - 0.9532696005963921
    s2 = 0.6116726519178519 * x0 + x1 - 3.5794586751235533
    return argmax(0, s1, s2)
