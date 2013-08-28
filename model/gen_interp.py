"""
Wrapper for scipy's interpolation module.
"""
# Based on work by John Stachurski
# Date: August 2009
# Corresponds to: Listing 6.4
import numpy as np
from scipy import interp
from scipy.interpolate.polyint import PchipInterpolator
from scipy.interpolate import interp1d, pchip
import matplotlib.pyplot as plt


class LinInterp(object):
    "Provides linear interpolation in one dimension."

    def __init__(self, X, Y, metadata=None):
        """Parameters: X and Y are sequences or arrays
        containing the (x,y) interpolation points.

        kind : type of interpolation. one of {"linear", "pchip"}
        """
        self.X = X
        self.Y = Y
        if metadata:
            self.metadata = metadata

    def __call__(self, z):
        """Parameters: z is a number, sequence or array.
        This method makes an instance f of LinInterp callable,
        so f(z) returns the interpolation value(s) at z.
        """
        return interp(z, self.X, self.Y)

    def __add__(self, other):
        assert (self.X == other.X).all()
        return self.Y + other.Y

    def __sub__(self, other):
        assert (self.X == other.X).all()
        return self.Y - other.Y

    def plot(self, **kwargs):
        return plt.plot(self.X, self.Y, **kwargs)

    def inverse(self):
        return LinInterp(self.Y, self.X)

    def __mul__(self, other):
        """Elementwise Multiplication"""
        return LinInterp(self.X, other.Y * self.Y)


class Interp(PchipInterpolator, object):
    """Provides an interpolation in one dimension.
    Mostly just compatability.
    """

    def __init__(self, X, Y, kind='linear'):
        """Parameters: X and Y are sequences or arrays
        containing the (x,y) interpolation points.

        kind : type of interpolation. one of {"linear", "pchip"}
        """
        self.X = X
        self.Y = Y
        self.kind = kind
        # This will change the methods available fyi.  Be careful w/ linear.
        if self.kind == 'pchip':
            PchipInterpolator.__init__(self, X, Y)

    def __call__(self, z):
        """Parameters: z is a number, sequence or array.
        This method makes an instance f of LinInterp callable,
        so f(z) returns the interpolation value(s) at z.
        """
        if self.kind == 'pchip':
            return pchip(self.X, self.Y)(z)
        else:
            return interp1d(self.X, self.Y, kind=self.kind,
                            bounds_error=False)(z)

    def __add__(self, other):
        assert (self.X == other.X).all()
        return self.Y + other.Y

    def __sub__(self, other):
        assert (self.X == other.X).all()
        return self.Y - other.Y

    def __mul__(self, other):
        """Elementwise Multiplication"""
        return Interp(self.X, other.Y * self.Y, kind=self.kind)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        return ax.plot(self.X, self.Y, **kwargs)

    def inverse(self, kind=None):
        if kind is None:
            kind = self.kind
        if (np.diff(self.Y) < 0).all():
            return Interp(self.Y, self.X, kind=kind)
        elif (np.diff(self.Y) > -.001).all():
            return Interp(np.sort(self.Y), self.X, kind=kind)
        else:
            raise ValueError("Cannot resort Y to invert.")
