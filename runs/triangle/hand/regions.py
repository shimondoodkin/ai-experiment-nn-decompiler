def net(x0):
    if x0 - 2.0 > 0:
        if x0 - 3.0 > 0:
            if x0 - 4.0 > 0:
                return 0.0
            else:
                return -x0 + 4.0
        else:
            return x0 - 2.0
    else:
        return 0.0
