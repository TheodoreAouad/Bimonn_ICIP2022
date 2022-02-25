from typing import Tuple, Any

import numpy as np

import matplotlib.pyplot as plt

from .element import Element


class Canva:

    def __init__(
        self,
        elements=dict(),
        xlim=None,
        ylim=None,
        axis='off',
        lim_mode='adaptable',
        **kwargs
    ):
        self.fig, self.ax = plt.subplots(1, 1, **kwargs)
        self.ax.axis(axis)
        if xlim is None:
            xlim = self.ax.get_xlim()
        if ylim is None:
            ylim = self.ax.get_ylim()

        self.elements = elements
        self.lim_mode = lim_mode
        self.xmin, self.xmax, self.ymin, self.ymax = None, None, None, None

        self.set_xlim(xlim)
        self.set_ylim(ylim)


        if self.lim_mode == 'adaptable':
            self.ax.set_xlim(auto=True)
            self.ax.set_ylim(auto=True)

    @property
    def xlim(self):
        return self.xmin, self.xmax

    @property
    def ylim(self):
        return self.ymin, self.ymax

    def set_xlim(self, left: float = None, right: float = None):
        if isinstance(left, tuple):
            left, right = left

        if left is not None:
            self.xmin = left
            self.ax.set_xlim(left=left)

        if right is not None:
            self.xmax = right
            self.ax.set_xlim(right=right)

    def set_ylim(self, bottom: float = None, top: float = None):
        if isinstance(bottom, tuple):
            bottom, top = bottom

        if bottom is not None:
            self.ymin = bottom
            self.ax.set_ylim(bottom=bottom)

        if top is not None:
            self.ymax = top
            self.ax.set_ylim(top=top)

    def set_lims(self, xlim: Tuple[float], ylim: Tuple[float]):
        self.set_xlim(xlim)
        self.set_ylim(ylim)
        return xlim, ylim

    def add_element(self, element: Element, key: Any = None, *args, **kwargs):
        if key is None:
            key = 0
            while key in self.elements.keys():
                key = np.random.randint(0, 9999999)

        self.elements[key] = element
        element.add_to_canva(self, *args, **kwargs)

        if self.lim_mode == "adaptable":
            self.adapt_lims_to_element(element.shape, element.xy_coords_botleft)

        return self

    def adapt_lims_to_element(self, shape: Tuple[float, float], xy_coords_botleft: Tuple[float, float]):
        new_xmin = xy_coords_botleft[0]
        if self.xmin > new_xmin:
            self.set_xlim(left=new_xmin)

        new_xmax = new_xmin + shape[0]
        if self.xmax < new_xmax:
            self.set_xlim(right=new_xmax)

        new_ymin = xy_coords_botleft[1]
        if self.ymin > new_ymin:
            self.set_ylim(bottom=new_ymin)

        new_ymax = new_ymin + shape[1]
        if self.ymax < new_ymax:
            self.set_ylim(top=new_ymax)

    def show(self):
        self.fig.show()
        return self.fig

    def draw_bounding_boxes(self):
        for elt in self.elements.values():
            elt.draw_bounding_box_on_ax(self.ax)

    def save_fig(self, savepath: str):
        self.fig.savefig(savepath)
