import numpy as np
from matplotlib.patches import Arc


def get_radius_union_intersection(height, prop_arc):
    return height * (1 + prop_arc) * 1.5


def _plot_union_intersection_on_ax(op, ax, center, width=1, height=1, prop_arc=.3, draw_circle=True, **kwargs):
    kwargs['color'] = kwargs.get('color', 'k')

    center = np.array(center)

    segm1 = np.array([[center[0] - width / 2, center[0] - width / 2], [center[1] - height / 2, center[1] + height / 2]])
    segm2 = segm1 + np.array([[width], [0]])

    if op == 'intersection':
        arc = Arc(center + np.array([0, height / 2]), width, height=prop_arc * height, theta1=0, theta2=180, **kwargs)
    elif op == 'union':
        arc = Arc(center - np.array([0, height / 2]), width, height=prop_arc * height, theta1=180, theta2=0, **kwargs)

    ax.plot(*segm1, **kwargs)
    ax.plot(*segm2, **kwargs)

    ax.add_patch(arc, )

    if draw_circle:
        circle_center = np.array([center[0], center[1]])
        radius = get_radius_union_intersection(height, prop_arc)
        ax.add_patch(Arc(circle_center, width=radius, height=radius, theta1=0, theta2=360, **kwargs))

    return ax


def plot_union_on_ax(*args, **kwargs):
    return _plot_union_intersection_on_ax('union', *args, **kwargs)


def plot_intersection_on_ax(*args, **kwargs):
    return _plot_union_intersection_on_ax('intersection', *args, **kwargs)


def plot_erosion_on_ax(ax, center, radius=1, **kwargs):
    ax.plot([center[0] - radius / 2, center[0] + radius / 2], [center[1], center[1]], **kwargs)
    ax.add_patch(Arc(center, radius, radius, theta1=0, theta2=360, **kwargs))
    return ax


def plot_dilation_on_ax(ax, center, radius=1, **kwargs):
    ax.plot([center[0] - radius / 2, center[0] + radius / 2], [center[1], center[1]], **kwargs)
    ax.plot([center[0], center[0]], [center[1] - radius / 2, center[1] + radius / 2], **kwargs)
    ax.add_patch(Arc(center, radius, radius, theta1=0, theta2=360, **kwargs))
    return ax
