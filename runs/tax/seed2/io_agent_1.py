def net(x0):
    income = x0
    tax = 0.12 * income + 0.08 * relu(income - 20) + 0.2 * relu(income - 80)
    return tax
