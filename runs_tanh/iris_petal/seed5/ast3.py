# produced by nndc ast (route: linear-scores)
def net(x0, x1):
    s1 = 0.20123761401672363 * x0 + 0.5628026046973759 * x1 - 0.8974209274645693
    s2 = 0.622198612522958 * x0 + x1 - 3.7094104974569952
    return argmax(0, s1, s2)
