# What I changed and why

The decompiled program is a three-band convex piecewise-linear ("tax bracket") curve:
zero below 3.5, then marginal rates 0.15 / 0.19 / 0.38 with kinks at 3.5, 23 and 63.5.

The failing cases are all in x0 ~ 71..169: the network rises too steeply too early (its top
band starts at 63.5), overshooting the label by up to ~2.4. Reading the labels directly shows a
clean bracket rule that fits all 2000 cases exactly (max error ~1e-14):

- marginal rate 0.10 from 0 to 20
- marginal rate 0.20 from 20 to 80
- marginal rate 0.40 above 80

i.e. `0.1*x0 + 0.1*relu(x0-20) + 0.2*relu(x0-80)`.

The network had learned a smeared approximation of this: the first kink at 3.5 instead of 0 (it
compensated with a higher 0.15 rate), the second at 23 instead of 20, and most importantly the top
kink at 63.5 instead of 80 with rate 0.38 instead of 0.40. Only that last error exceeds the
tolerance, but simply moving the top kink to 80 is not enough (max error 1.44 > tolerance 1.24),
because the lower bands are also slightly off. So I replaced the three bands with the exact
breakpoints (0, 20, 80) and rates (0.1, 0.2, 0.4) - same structure, same length.

Score: right on 2000/2000; fixes all 989 network failures; breaks 0 of the 1011 points the network
already got right (they are reproduced exactly rather than approximately).
