from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw
from numba import njit
from general.numba_utils import numba_randint, numba_rand, numba_rand_shape_2d
from general.utils import set_borders_to


@njit
def numba_straight_rect(width, height):
    return np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])


@njit
def numba_rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


@njit
def numba_transform_rect(rect: np.ndarray, R: np.ndarray, offset: np.ndarray):
    return np.dot(rect, R) + offset


@njit
def numba_correspondance(ar: np.ndarray) -> np.ndarray:
    return 1 - ar


@njit
def numba_invert_proba(ar: np.ndarray, p_invert: float) -> np.ndarray:
    if numba_rand() < p_invert:
        return numba_correspondance(ar)
    return ar


def get_rect(x, y, width, height, angle):
    rect = numba_straight_rect(width, height)
    theta = (np.pi / 180.0) * angle
    R = numba_rotation_matrix(theta)
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect


def draw_poly(draw, poly, fill_value=1):
    draw.polygon([tuple(p) for p in poly], fill=fill_value)


def draw_ellipse(draw, center, radius, fill_value=1):
    bbox = (center[0] - radius[0], center[1] - radius[1], center[0] + radius[0], center[1] + radius[1])
    draw.ellipse(bbox, fill=fill_value)


def get_random_rotated_diskorect(
    size: Tuple, n_shapes: int = 30, max_shape: Tuple[int] = (15, 15), p_invert: float = 0.5,
        border=(4, 4), n_holes: int = 15, max_shape_holes: Tuple[int] = (5, 5), noise_proba=0.05, **kwargs
):
    diskorect = np.zeros(size)
    img = Image.fromarray(diskorect)
    draw = ImageDraw.Draw(img)

    def draw_shape(max_shape, fill_value):
        x = numba_randint(0, size[0] - 2)
        y = numba_randint(0, size[0] - 2)

        if np.random.rand() < .5:
            W = numba_randint(1, max_shape[0])
            L = numba_randint(1, max_shape[1])

            angle = numba_rand() * 45
            draw_poly(draw, get_rect(x, y, W, L, angle), fill_value=fill_value)

        else:
            rx = numba_randint(1, max_shape[0]//2)
            ry = numba_randint(1, max_shape[1]//2)
            draw_ellipse(draw, np.array([x, y]), (rx, ry), fill_value=fill_value)

    for _ in range(n_shapes):
        draw_shape(max_shape=max_shape, fill_value=1)

    for _ in range(n_holes):
        draw_shape(max_shape=max_shape_holes, fill_value=0)

    diskorect = np.asarray(img) + 0
    diskorect[numba_rand_shape_2d(*diskorect.shape) < noise_proba] = 1
    diskorect = numba_invert_proba(diskorect, p_invert)

    diskorect = set_borders_to(diskorect, border, value=0)
    return diskorect


def get_random_diskorect_channels(size: Tuple, squeeze: bool = False, *args, **kwargs):
    """Applies diskorect to multiple channels.

    Args:
        size (Tuple): (W, L, H)
        squeeze (bool, optional): If True, squeeze the output: if H = 1, returns size (W, L). Defaults to False.

    Raises:
        ValueError: size must be of len 2 or 3, either (W, L) or (W, L, H) with H number of channels.

    Returns:
        np.ndarray: size (W, L) or (W, L, H)
    """
    if len(size) == 3:
        W, L, H = size
    elif len(size) == 2:
        W, L = size
        H = 1
    else:
        raise ValueError(f"size argument must have 3 or 2 values, not f{len(size)}.")

    final_img = np.zeros((W, L, H))
    for chan in range(H):
        final_img[..., chan] = get_random_rotated_diskorect((W, L), *args, **kwargs)

    if squeeze:
        return np.squeeze(final_img)
    return final_img
