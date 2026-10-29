def net(x0, x1):
    # two straight decision lines split the (x0, x1) box into three bands
    low = x0 + 3.3 * x1 < 4.9      # small x0 and small x1 -> class 0
    high = x0 + 1.1 * x1 > 6.8     # large x0 and large x1 -> class 2
    if low:
        return 0
    if high:
        return 2
    return 1
