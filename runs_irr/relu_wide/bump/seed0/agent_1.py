# A round bump, close to exp(-(x0*x0 + x1*x1)): peak about 1.03 at the origin, about 0 for x0*x0 + x1*x1 > 4.
# Written as: peak + small tilt - "cone" ramps (start near the origin, make it fall)
#             + "floor" ramps (start at distance ~1-1.5 from the origin, cancel the fall so it flattens at 0).
# Each ramp relu(n . x - d) is zero at the origin; ramps are sorted by direction angle.
def net(x0, x1):
    c0 = relu(0.74 * x0 - 0.09)  # dir 0 deg, starts at 0.12
    c1 = relu(0.52 * x0 + 0.34 * x1 - 0.02)  # dir 33 deg, starts at 0.03
    c2 = relu(0.46 * x0 + 0.31 * x1 - 0.14)  # dir 34 deg, starts at 0.25
    c3 = relu(0.31 * x0 + 0.59 * x1 - 0.02)  # dir 62 deg, starts at 0.03
    c4 = relu(0.06 * x0 + 0.52 * x1 - 0.02)  # dir 83 deg, starts at 0.04
    c5 = relu(0.04 * x0 + 0.46 * x1 - 0.16)  # dir 85 deg, starts at 0.35
    c6 = relu(-0.23 * x0 + 0.46 * x1 - 0.13)  # dir 117 deg, starts at 0.25
    c7 = relu(-0.34 * x0 + 0.42 * x1 - 0.03)  # dir 129 deg, starts at 0.06
    c8 = relu(-0.42 * x0 + 0.26 * x1 - 0.09)  # dir 148 deg, starts at 0.18
    c9 = relu(-0.48 * x0 - 0.1)  # dir 180 deg, starts at 0.21
    c10 = relu(-0.84 * x0 - 0.26 * x1)  # dir 197 deg, starts at -0.00
    c11 = relu(-0.42 * x0 - 0.48 * x1 - 0.02)  # dir 229 deg, starts at 0.03
    c12 = relu(-0.09 * x0 - 0.69 * x1 - 0.14)  # dir 263 deg, starts at 0.20
    c13 = relu(0.04 * x0 - 0.67 * x1 - 0.12)  # dir 273 deg, starts at 0.18
    c14 = relu(0.29 * x0 - 0.71 * x1 - 0.07)  # dir 292 deg, starts at 0.09
    c15 = relu(0.61 * x0 - 0.38 * x1 - 0.09)  # dir 328 deg, starts at 0.13
    c16 = relu(0.6 * x0 - 0.15 * x1)  # dir 346 deg, starts at -0.00
    f0 = relu(0.54 * x0 + 0.11 * x1 - 0.65)  # dir 12 deg, starts at 1.18
    f1 = relu(0.59 * x0 + 0.33 * x1 - 0.8)  # dir 29 deg, starts at 1.18
    f2 = relu(0.28 * x0 + 0.42 * x1 - 0.54)  # dir 56 deg, starts at 1.07
    f3 = relu(0.36 * x0 + 0.56 * x1 - 0.96)  # dir 57 deg, starts at 1.44
    f4 = relu(0.02 * x0 + 0.41 * x1 - 0.58)  # dir 87 deg, starts at 1.41
    f5 = relu(0.02 * x0 + 0.42 * x1 - 0.44)  # dir 87 deg, starts at 1.05
    f6 = relu(-0.14 * x0 + 0.46 * x1 - 0.53)  # dir 107 deg, starts at 1.10
    f7 = relu(-0.3 * x0 + 0.49 * x1 - 0.71)  # dir 121 deg, starts at 1.24
    f8 = relu(-0.69 * x0 + 0.49 * x1 - 0.88)  # dir 145 deg, starts at 1.04
    f9 = relu(-0.48 * x0 + 0.34 * x1 - 0.86)  # dir 145 deg, starts at 1.46
    f10 = relu(-0.75 * x0 + 0.12 * x1 - 0.92)  # dir 171 deg, starts at 1.21
    f11 = relu(-0.48 * x0 - 0.1 * x1 - 0.58)  # dir 192 deg, starts at 1.18
    f12 = relu(-0.38 * x0 - 0.23 * x1 - 0.55)  # dir 211 deg, starts at 1.24
    f13 = relu(-0.34 * x0 - 0.52 * x1 - 0.89)  # dir 237 deg, starts at 1.43
    f14 = relu(-0.36 * x0 - 0.56 * x1 - 0.68)  # dir 237 deg, starts at 1.02
    f15 = relu(-0.04 * x0 - 0.91 * x1 - 1.1)  # dir 267 deg, starts at 1.21
    f16 = relu(0.23 * x0 - 0.49 * x1 - 0.51)  # dir 295 deg, starts at 0.94
    f17 = relu(0.34 * x0 - 0.62 * x1 - 0.96)  # dir 299 deg, starts at 1.36
    f18 = relu(0.38 * x0 - 0.26 * x1 - 0.43)  # dir 326 deg, starts at 0.93
    f19 = relu(0.53 * x0 - 0.36 * x1 - 0.87)  # dir 326 deg, starts at 1.36
    f20 = relu(0.54 * x0 - 0.08 * x1 - 0.67)  # dir 352 deg, starts at 1.23
    cone = sum(0.206 * c0, 0.374 * c1, 0.117 * c2, 0.307 * c3, 0.231 * c4, 0.255 * c5, 0.225 * c6, 0.301 * c7, 0.195 * c8, 0.26 * c9, 0.184 * c10, 0.2 * c11, 0.172 * c12, 0.147 * c13, 0.266 * c14, 0.282 * c15, 0.268 * c16)
    floor = sum(0.214 * f0, 0.241 * f1, 0.282 * f2, 0.202 * f3, 0.178 * f4, 0.286 * f5, 0.193 * f6, 0.264 * f7, 0.14 * f8, 0.207 * f9, 0.23 * f10, 0.233 * f11, 0.365 * f12, 0.208 * f13, 0.192 * f14, 0.216 * f15, 0.15 * f16, 0.224 * f17, 0.19 * f18, 0.206 * f19, 0.313 * f20)
    tilt = 0.155 * x0 + 0.067 * x1
    return 1.039 + tilt - cone + floor
