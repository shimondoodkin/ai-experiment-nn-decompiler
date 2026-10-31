# A round bump centred on the origin, close to exp(-(x0*x0 + x1*x1)) (peak 1 at (0, 0),
# about 0.37 at radius 1, about 0 beyond radius 2). The bump itself does not pass the
# check: the network ripples by up to 0.04 around it. So it is written as the network
# builds it: steep sigmoid walls (slope about 3) around a disc of radius about 0.55,
# in several directions, plus three broad terms that shape the shoulders.
# Name pattern: dir<angle of the wall normal in degrees>; m = minus.
def net(x0, x1):
    # walls the output climbs as it moves into the disc (weight about +0.45)
    in_dir27 = sigmoid(3 * x0 + 1.5 * x1 + 1.75)
    in_dir64 = sigmoid(1.35 * x0 + 2.75 * x1 + 1.75)
    in_dirm51 = sigmoid(2 * x0 - 2.5 * x1 + 1.75)
    in_dirm174 = sigmoid(-2.75 * x0 - 0.3 * x1 + 1.5)
    # walls the output drops as it moves out of the disc (weight about -0.48)
    out_dirm32 = sigmoid(2.75 * x0 - 1.75 * x1 - 1.75)
    out_dirm72 = sigmoid(x0 - 3.1 * x1 - 1.75)
    out_dir45 = sigmoid(2.25 * x0 + 2.25 * x1 - 1.5)
    out_dir86 = sigmoid(0.25 * x0 + 3.5 * x1 - 2)
    out_dir167 = sigmoid(-3.35 * x0 + 0.75 * x1 - 1.75)
    # broad, gentle terms
    wide_x0_low = sigmoid(-x0 + 0.1 * x1 + 1.25)
    wide_dir121 = sigmoid(-0.75 * x0 + 1.25 * x1 - 2.75)
    wide_dirm129 = sigmoid(-x0 - 1.25 * x1 - 3)
    walls_in = 0.45 * in_dir27 + 0.48 * in_dir64 + 0.45 * in_dirm51 + 0.59 * in_dirm174
    walls_out = 0.5 * out_dirm32 + 0.49 * out_dirm72 + 0.48 * out_dir45 + 0.476 * out_dir86 + 0.46 * out_dir167
    broad = 0.436 * wide_dir121 + 0.435 * wide_dirm129 - 0.7 * wide_x0_low
    return 0.192 + walls_in - walls_out + broad
