import numpy as np


###############################################################################
# Compute moving_average. credit to:
# https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
###############################################################################
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:]/n


def taper(x, sigma=0.01):
    # Very crude taper for the periodic bc case
    nx = len(x.flatten())
    x0 = np.linspace(0, 1, nx)
    xc_left = sigma / 2
    xc_right = 1 - (sigma / 2)
    y = np.cumsum(
        np.exp(-((x0 - xc_left) ** 2) / sigma ** 2)
        - np.exp(-((x0 - xc_right) ** 2) / sigma ** 2),
        axis=0,
    )
    y = y / y[int(nx / 2)]

    return y
