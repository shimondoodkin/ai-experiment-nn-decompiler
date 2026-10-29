def net(x0, x1):
    # Truth table of the net (raw score):
    #   x0=0,x1=0 -> -11.5   x0=1,x1=0 -> +11.4
    #   x0=0,x1=1 -> ~-4e-5  x0=1,x1=1 -> ~+7e-5   (x1=1 rows sit on the boundary)
    # The decision (score > 0) is exactly x0; x1 is irrelevant to the decision.
    return x0
