# produced by nndc ast (route: linear-scores)
def net(x0, x1):
    s1 = 0.27626143577568874 * x0 + 0.5909463553345534 * x1 - 1.0991641331616555
    s2 = 0.6405733940315114 * x0 + x1 - 3.5763701619959662
    return argmax(0, s1, s2)
