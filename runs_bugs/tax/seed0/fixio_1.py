def net(x0):
    # Progressive marginal-rate schedule (like income-tax brackets):
    # 10% on the part up to 20, 20% on the part between 20 and 80, 40% on the part above 80.
    low = min(x0, 20)
    mid = min(max(x0 - 20, 0), 60)
    high = max(x0 - 80, 0)
    return 0.1 * low + 0.2 * mid + 0.4 * high
