import numpy as np
from scipy import optimize
from math import atan2

def leastsq_circle(x, y):
    """
    fit a circle using least squares.
    :param x: 1D numpy array
    :param y: 1D numpy array
    :return:
    """

    def calc_radius(x, y, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f(c, x, y):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        ri = calc_radius(x, y, *c)
        return ri - ri.mean()

    x_mean, y_mean = np.mean(x), np.mean(y)
    (xc, yc), ier = optimize.leastsq(f, (x_mean, y_mean), args=(x, y))
    ri = calc_radius(x, y, *(xc, yc))
    r = ri.mean()
    residue = np.sum((ri - r) ** 2)
    return xc, yc, r, residue

def calc_sign(x1, y1, x2, y2, xc, yc):
    theta1 = atan2(y1 - yc, x1 - xc)
    theta2 = atan2(y2 - yc, x2 - xc)
    return np.sign(theta1 - theta2)