def net(x0, x1):
    return (126.0 + x0 + ((x0 * x0) * (-31.0 + abs((x0 * x1)))) + ((x1 * x1) * (-31.0 + abs((x0 * x1)))))
