# Notes

## What I changed
- Result: `net(x0, x1) = abs(x0 - x1)`. Check: `PASS (snap)`, max deviation 0.03 against a snap tolerance of 0.12.
- Neurons 0, 2, 4, 5 and 7 all have weights of the form (a, -a) on (x0, x1). So each one is relu(+-a*(x0-x1) + tiny bias), a function of d = x0 - x1 only.
  - The slope for d>0 is 0.140*0.512 + 0.825*1.127 = 1.001, so it is 1.
  - The slope for d<0 is 0.953*0.745 + 0.028*0.034 + 0.445*0.652 = 1.001, so it is also 1.
  - The biases are close to 0. These neurons therefore add up to |d|.
- Neurons 1, 3 and 6 depend on x1 only, and their output weights are small (|w| < 0.13).
  - Neurons 1 and 6 have kinks near x1 = 0.5 with opposite slopes (-0.087 and +0.085), so they almost cancel.
  - Neuron 3 only becomes active when x1 > 2.46, which adds at most about 0.01.
  - Together with the output bias of -0.026, they leave a residual of no more than about 0.03. I treated this as training noise and dropped it.

## Heuristics
- Group hidden neurons by the direction of their input weights. Weights like (a, -a) mean the neuron only sees a difference x0 - x1, and (0, a) means it only sees one input.
- Two neurons on the same direction with opposite signs, relu(k*d) and relu(-k*d), make an abs-like V shape. Add up (output weight * input slope) on each side. If both sides come to about 1, the group is abs(d). If they differ, it is a*relu(d) + b*relu(-d).
- Several neurons can share one kink. Sum their effective slopes instead of snapping each neuron on its own.
- Neurons with small output weights, or with kinks at the edge of the input range, are often noise. Pairs with nearby kinks and opposite slopes largely cancel. Remove them and use the checker to confirm that the leftover error is within tolerance.
- Snap biases that are tiny compared with the weights (relative bias < 0.1) to 0, so the kinks sit exactly at 0.
- Try the simplest candidate formula early. The checker's report of the maximum deviation tells you whether the remaining terms matter.
