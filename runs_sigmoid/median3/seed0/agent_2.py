def net(x0, x1, x2):
    # soft comparisons ("is this input above that blend of the others?")
    x1_above = sigmoid(1.5 * x1 - 0.9 * x0 - 0.6 * x2)
    x0_above = sigmoid(1.3 * x0 - 0.7 * x1 - 0.7 * x2)
    x1_gt_x2 = sigmoid(2.5 * x1 - 2.2 * x2 - 0.2 * x0)
    x2_below = sigmoid(0.9 * x0 + 0.5 * x1 - 1.5 * x2)
    x2_gt_x1 = sigmoid(2.6 * x2 - 2.4 * x1 - 0.4 * x0)
    # linear trend + soft-comparison terms and their pairwise ANDs
    trend = 0.3 * x0 + 0.36 * (x1 + x2) - 1.62
    single = 0.72 * x1_above + 3.32 * x0_above + 1.34 * x1_gt_x2 - 5.0 * x2_below
    pairs = (1.45 * x1_above * x0_above + 1.44 * x1_above * x2_gt_x1 - 3.0 * x0_above * x1_gt_x2
             + 1.41 * x0_above * x2_below + 3.0 * x1_gt_x2 * x2_below + 1.46 * x2_below * x2_gt_x1)
    return trend + single + pairs
