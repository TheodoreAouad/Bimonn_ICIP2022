from typing import Tuple

import numpy as np
from matplotlib.patches import Rectangle


class Element:

    def __init__(
        self,
        shape=None,
        xy_coords_botleft=None,
        xy_coords_mean=None,
        key=None,
    ):
        if xy_coords_botleft is not None and xy_coords_mean is not None:
            raise ValueError("choose either xy coords botleft or mean.")

        self.key = key
        self._shape = np.array(shape)
        self._xy_coords_botleft = None

        if xy_coords_botleft is not None:
            self.set_xy_coords_botleft(xy_coords_botleft)
        if xy_coords_mean is not None:
            self.set_xy_coords_mean(xy_coords_mean)

    def translate(self, vector: np.ndarray):
        self.set_xy_coords_botleft(self.xy_coords_botleft + vector)

    def set_shape(self, new_shape):
        self._shape = np.array(new_shape)
        return self

    @property
    def shape(self):
        return self._shape

    @property
    def xy_coords_mean(self):
        return self.xy_coords_botleft + self.shape / 2

    @property
    def barycentre(self):
        return self.xy_coords_mean

    @property
    def xy_coords_topleft(self):
        return self.xy_coords_botleft + np.array([0, self.shape[1]])

    @property
    def xy_coords_topright(self):
        return self.xy_coords_botleft + self.shape

    @property
    def xy_coords_botright(self):
        return self.xy_coords_botleft + np.array([self.shape[0], 0])

    @property
    def xy_coords_midright(self):
        return self.barycentre + np.array([self.shape[0] / 2, 0])

    @property
    def xy_coords_midleft(self):
        return self.barycentre - np.array([self.shape[0] / 2, 0])

    @property
    def xy_coords_midtop(self):
        return self.barycentre + np.array([0, self.shape[1] / 2])

    @property
    def xy_coords_midbot(self):
        return self.barycentre - np.array([0, self.shape[1] / 2])

    @property
    def xy_coords_midbottom(self):
        return self.xy_coords_midbot

    @property
    def xy_coords_botleft(self):
        return self._xy_coords_botleft

    def is_inside_element(self, coords):
        coords = np.array(coords)
        return (self.xy_coords_botleft <= coords <= self.xy_coords_topright).all()

    def set_xy_coords_mean(self, new_coords: Tuple):
        assert self.shape is not None, "Must give shape to give coords mean. Else give coords botleft and mean."
        new_coords = np.array(new_coords)
        self._xy_coords_botleft = new_coords - self.shape / 2

    def set_xy_coords_botleft(self, new_coords: Tuple):
        self._xy_coords_botleft = np.array(new_coords)

    def add_to_canva(self, canva: "Canva"):
        pass

    def draw_bounding_box_on_ax(self, ax, **kwargs):
        kwargs['color'] = kwargs.get('color', 'k')
        ax.add_patch(Rectangle(self.xy_coords_botleft, self.shape[0], self.shape[1], fill=False, **kwargs))


class ElementGrouper(Element):

    def __init__(self, elements=dict(), *args, **kwargs):
        self.elements = dict()
        super().__init__(shape=None, xy_coords_botleft=None, *args, **kwargs)
        for key, element in elements.items():
            self.add_element(element, key=key)
        # self.xy_coords_botleft = np.array([0, 0])  # TODO: adapt this to the elements
        # self.shape = np.array([0, 0])  # TODO: adapt this to the elements

    def translate(self, vector: np.ndarray):
        # self.translate(vector)
        for element in self.elements.values():
            element.translate(vector)
        return self

    def set_xy_coords_mean(self, new_coords: Tuple):
        return self.translate(new_coords - self.xy_coords_mean)

    def set_xy_coords_botleft(self, new_coords: Tuple):
        return self.translate(new_coords - self.xy_coords_botleft)

    def set_xy_coords_midleft(self, new_coords: Tuple):
        return self.translate(new_coords - self.xy_coords_midleft)


    def add_element(self, element: Element, key=None):
        if key is None:
            key = element.key

        if key is None:
            key = 0
            while key in self.elements.keys():
                key = np.random.randint(0, 9999999)

        self.elements[key] = element
        return self

    def add_to_canva(self, canva: "Canva"):
        for key, element in self.elements.items():
            if element in canva.elements.values():
                continue

            if key in canva.elements.keys():
                idx = 0
                while f'{key}_{idx}' in canva.elements.keys():
                    idx += 1
                canva.add_element(element, key=f'{key}_{idx}')
            else:
                canva.add_element(element, key=key)

            canva.add_element(element, key)
        return self

    def __len__(self):
        return len(self.elements)

    @property
    def xy_coords_botleft(self):
        if len(self.elements) == 0:
            return np.array([0, 0])
        all_coords = np.stack([elt.xy_coords_botleft for elt in self.elements.values()], axis=0)
        return all_coords.min(0)

    @property
    def xy_coords_topright(self):
        if len(self.elements) == 0:
            return np.array([0, 0])
        all_coords = np.stack([elt.xy_coords_topright for elt in self.elements.values()])
        return all_coords.max(0)

    @property
    def shape(self):
        return self.xy_coords_topright - self.xy_coords_botleft
