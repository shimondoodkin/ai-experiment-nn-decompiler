# produced by nndc ast (route: linear-scores)
def net(x0, x1):
    s1 = 0.15915054529134104 * x0 + 0.6699688368198784 * x1 - 0.870667825882796
    s2 = 0.4552761517773256 * x0 + x1 - 2.8738100963388846
    return argmax(0, s1, s2)
