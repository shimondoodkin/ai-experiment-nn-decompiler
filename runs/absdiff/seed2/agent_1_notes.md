# Notes

Result: `net(x0, x1) = abs(x0 - x1)`, which gets PASS (snap). The largest deviation is 0.017 near (-3, -3), well inside the snap tolerance of 0.12.

## What I changed and why
- Neurons 0, 1, 2 have weights of roughly (+c, -c) and bias about 0, so each is relu(c*d) with d = x0 - x1.
  Their output weights add up to 0.661*0.379 + 0.712*0.721 + 0.628*0.506 = 1.082, giving 1.082*relu(d).
- Neurons 3 and 7 are both relu(-c*d) with bias about 0. Together they give 0.723*0.336 + 1.001*0.674 = 0.918, so 0.918*relu(-d).
- Neuron 5, relu(0.132*d + 0.809), is always active over the input box: its smallest pre-activation is 0.809 - 0.132*6 > 0.
  That makes it a linear term: -0.618*(0.132 d + 0.809) = -0.082 d - 0.500. The constant -0.500 cancels the output bias of +0.495.
- When you add them up: for d > 0 the slope is 1.082 - 0.082 = 1.0, and for d < 0 it is -0.918 - 0.082 = -1.0. So the output is |d|.
- Neuron 4 (output weight 0.022, only active when x1 < -2.47) and neuron 6 (output weight -0.002) add less than 0.02 anywhere. I dropped both as noise.

## Heuristics
- Group hidden neurons by the direction of their input weights. Neurons with proportional weight vectors and the same bias (about 0 here) combine into one ReLU whose output weight is the sum of w_in*w_out.
- Find neurons whose pre-activation never changes sign over the input bounds. Check the min and max over the box corners. Replace them with a linear term (or 0 if always off), then fold their constants into the output bias.
- relu(d) and relu(-d) with slopes a and b, plus a linear term of slope s, become |d| when a+s = 1 and b-s = 1. Look for abs, or relu, hidden behind an always-on "linear correction" neuron.
- Find each neuron's maximum contribution, |w_out| * max relu, over the input box. Drop it if that is far below the snap tolerance.
- Try the simplest closed form first (abs, max, min of simple linear combinations) and verify it with the checker before tuning residual constants.
