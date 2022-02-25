import random

import numpy as np
from numba import njit


@njit
def numba_randint(a: int, b: int):
    return random.randint(a, b)


@njit
def numba_rand():
    return random.random()


@njit
def numba_rand_shape_2d(x, y):
    return np.random.rand(x, y)


@njit
def numba_array(ar):
    return np.array(ar)


@njit
def numba_compare(ar1, ar2):
    return ar1 < ar2
