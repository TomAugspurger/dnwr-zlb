"""
Origin: QEwP by John Stachurski and Thomas J. Sargent
Date:   5/2013
File:   ecdf.py

Implements the empirical distribution function.

"""
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class ecdf:

    def __init__(self, observations, metadata=None):
        """
        Optionally pass a dictionary of metadata,
        say the parameters used to generate the ecdf.
        """
        if not is_mono(observations):
            observations = np.sort(observations)
        if metadata:
            self.meta = metadata
        self.observations = np.asarray(observations).reshape(-1, 1)
        self.n = len(self.observations)

    def __call__(self, x):
        """Handles vectors for x"""
        return (self.observations <= x).sum(0) / self.n

    def plot(self, ax=None, a=None, b=None, **kwargs):

        # === choose reasonable interval if [a, b] not specified === #
        if not a:
            a = self.observations[0] - self.observations.std()
        if not b:
            b = self.observations[-1] + self.observations.std()

        # === generate plot === #
        x_vals = np.linspace(a, b, num=self.n)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(x_vals, self(x_vals.T), **kwargs)
        plt.show()
        return ax

    def get_range(self):
        return self.observations.min(), self.observations.max()

    def hist(self, **kwargs):
        try:
            ax = kwargs.pop('ax')
        except KeyError:
            fig, ax = plt.subplots()
        df = pd.DataFrame(self.observations)
        ax = df.hist(ax=ax, **kwargs)
        return ax


def is_mono(x):
    if (np.diff(x) >= 0).all():
        return True
    else:
        return False
