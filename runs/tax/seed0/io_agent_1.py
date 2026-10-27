def net(x0):
    income = x0
    tax = 0.2 * relu(income - 10) + 0.2 * relu(income - 80)
    return tax
