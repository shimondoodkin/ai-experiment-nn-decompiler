def net(x0, x1):
    return (79.0 + (-9.0 * abs((abs(x0) * min(127.0, x1)))) + min((-37.0 * abs(x1)), (abs(x0) * min(-37.0, x1))))
