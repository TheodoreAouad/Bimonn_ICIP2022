import numpy as np
from skimage.morphology import disk


def identity(size: int) -> np.ndarray:
    selem = np.zeros((size, size))
    selem[size//2, size//2] = 1
    return selem


def vstick(size: int) -> np.ndarray:
    selem = np.zeros((size, size))
    selem[:, size//2] = 1
    return selem


def hstick(size: int) -> np.ndarray:
    selem = np.zeros((size, size))
    selem[size//2, :] = 1   
    return selem


def diagonal_cross(size: int) -> np.ndarray:
    selem = np.zeros((size, size))
    selem[np.arange(size), size - np.arange(1, size + 1)] = 1
    selem[np.arange(size), np.arange(size)] = 1
    return selem


def straight_cross(size: int) -> np.ndarray:
    selem = np.zeros((size, size))
    selem[size//2, :] = 1
    selem[:, size//2] = 1
    return selem


def square(size: int, rect_size: int = None) -> np.ndarray:
    if rect_size is None:
        rect_size = size - 2
    selem = np.zeros((size, size))
    margin = (size - rect_size) // 2
    selem[margin: -margin, margin: -margin] = 1
    return selem


def dcross(*args, **kwargs):
    return diagonal_cross(*args, **kwargs)


def scross(*args, **kwargs):
    return straight_cross(*args, **kwargs)
