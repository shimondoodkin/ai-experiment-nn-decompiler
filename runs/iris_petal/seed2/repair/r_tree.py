def net(x0, x1):
    if x1 <= 0.800000011920929:
        return 0
    else:
        if x1 <= 1.6500000357627869:
            if x0 <= 4.950000047683716:
                return 1
            else:
                return 2
        else:
            return 2
