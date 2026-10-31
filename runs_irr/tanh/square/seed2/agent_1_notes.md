# Notes

## What the network computes

`net(x0) ≈ x0²` on [-2, 2]. Values from the dump: 0.017 at 0, 0.24 at ±0.5, 0.97 at ±1, 2.27 at ±1.5,
3.90 to 3.93 at ±2. The net falls short of the parabola near the ends of the box (about −0.10 at
x0 = 2 and −0.07 at x0 = −2). Inside the box it wobbles by about ±0.035 around x0², and it is slightly
asymmetric: its odd part is at most 0.015.

## What I changed

1. **Six near-constant units.** Units 0–5 have slopes of about 0.03, so over |x0| ≤ 2 their
   pre-activations move by less than 0.1. Their weighted sum is 6.670 + 0.077·x0 (max deviation from
   that line 0.006, and 7e-7 from a cubic). Together with the output bias they are just a constant
   plus a tiny linear tilt, which is where the asymmetry comes from.
2. **Two mirrored units do the real work.** Units 6 and 7 are tanh(0.827·x0 + 1.524) and
   tanh(−0.866·x0 + 1.573), with output weights −4.21 and −3.81. Snapped to one shared slope, offset
   and weight they form an even bowl:
   `7.3 − 4·(tanh(1.55 + 0.85·x0) + tanh(1.55 − 0.85·x0))`. That passes the check with a max error of
   0.028. I kept it as `scratch/tanhpair.py`.
3. **Formula.** `x0*x0` by itself fails, with a max error of 0.104 at x0 = 2, against a snap tolerance
   of 0.078. The edge loss is smooth, like a saturation, so I added an `x0⁶` term and a small offset:
   `x0² − x0⁶/1000 + 0.01`. It passes as `PASS (snap)` with a max error of 0.050. A least-squares fit
   of the net's even part in x0² gives about 1.006·x0² − 0.0021·x0⁴ − 0.008. The interior wobble of
   about ±0.03 limits any pure-polynomial snap. I kept the formula as `agent.py` because it says what
   the function is. The tanh pair is the closer, more literal alternative.

## Heuristics

- **Near-constant units:** a unit with a tiny slope (|w|·input range much less than 1) is a
  constant, or at most a small linear term, over the box. Fold all such units into one
  constant-plus-line, and check the line's slope before dropping it.
- **Mirrored pairs:** two units with opposite-sign slopes of similar size, similar offsets and
  same-sign output weights form an even bump or bowl. Snap them to a shared |slope|, offset and
  weight: `c − b·(tanh(o + kx) + tanh(o − kx))`. For a 1-input regression, a bowl shape like this
  usually means x², and a bump means something like 1/(1+x²) or exp(−x²).
- **Sample before modelling:** evaluate the net at integer and half-integer inputs and compare it
  with obvious candidates (x², |x|, x³, …). Here the values were close to 0.25, 1, 2.25 and 4.
- **When the plain formula just misses the tolerance:** look at the residual. A smooth loss that
  grows toward the box edges is tanh saturation. A higher even power with a round denominator
  (x⁶/1000) fixes it without any fitted-looking constants. An interior wobble the size of the
  residual means the net itself is noisy, so stop snapping further.
- **Tolerance:** snap tolerance is 2% of the output range. Compute it (here 0.078) before you choose
  between the bare formula and a corrected one.
