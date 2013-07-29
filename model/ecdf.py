# Filename: ecdf.py
# Author: John Stachurski
# Date: December 2008
# Corresponds to: Listing 6.3

import matplotlib.pyplot as plt
import numpy as np


class ECDF:

    def __init__(self, observations):
        self.observations = observations

    def __call__(self, x):
        counter = 0.0
        for obs in self.observations:
            if obs <= x:
                counter += 1
        return counter / len(self.observations)


def ecdf(obs, x):
    "Stop Writing Classes."
    counter = 0
    for ob in obs:
        if ob <= x:
            counter += 1
    return counter / len(obs)


X = np.random.randn(1000)
U = np.random.uniform(size=1000)

ecdfsu = [ecdf(U, quantile) for quantile in np.linspace(0, 1, 100)]
ecdfs = [ecdf(X, quantile) for quantile in np.linspace(-3, 3, 1000)]

plt.plot(np.linspace(-3, 3, 1000), ecdfs)
plt.plot(np.linspace(0, 1, 100), ecdfsu)
