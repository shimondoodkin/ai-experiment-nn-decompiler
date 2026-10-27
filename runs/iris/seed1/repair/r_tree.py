def net(x0, x1, x2, x3):
    if x3 <= 0.800000011920929:
        return 0
    else:
        if x3 <= 1.6500000357627869:
            if x2 <= 4.950000047683716:
                return 1
            else:
                return 2
        else:
            return 2
