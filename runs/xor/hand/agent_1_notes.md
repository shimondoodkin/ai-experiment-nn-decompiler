# Notes

## What the network does
Two bool inputs, hidden layer `h0 = relu(x0+x1)`, `h1 = relu(x0+x1-1)`, output `h0 - 2*h1 - 0.5`.
With s = x0 + x1 in {0,1,2}: h0 = s, h1 = max(s-1, 0), so the output is -0.5, +0.5, -0.5.
That is XOR, with a constant margin of 0.5.

## What I changed
- Replaced the two ReLUs with `abs(x0 - x1) - 0.5`. On {0,1}^2, `abs(x0 - x1)` equals
  `s - 2*relu(s-1)`, so the score matches the network exactly (PASS exact, exhaustive over 4 inputs),
  not only the class.
- The weights were already snapped (1, -1, -2, -0.5), so no snapping was needed.

## Heuristics
- With bool inputs and few of them, list every input and the raw score; the truth table often
  names the function (here XOR) straight away.
- If a hidden neuron sees only a sum s of inputs, rewrite the network as a piecewise-linear
  function of s. `relu(s) - 2*relu(s-1)` is a "tent" shape, which is `1 - abs(s-1)`, and on bools
  is `abs(x0 - x1)`.
- When the score takes only two values (±c), write it as `indicator - c` with a closed-form
  indicator (abs, min, max) so the result matches exactly, not just in sign.
- A bias of -0.5 on a 0/1-valued expression is the usual "half-step" threshold: the output is
  `expr > 0.5`, i.e. `expr == 1` for integers.
