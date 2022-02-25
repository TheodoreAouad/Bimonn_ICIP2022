import numpy as np

from .element import Element
from .plot_symbols import plot_union_on_ax, plot_intersection_on_ax, plot_erosion_on_ax, plot_dilation_on_ax, get_radius_union_intersection


class ElementSymbolUnionIntersection(Element):

    def __init__(self, op, xy_coords_mean, width=1, height=1, prop_arc=.3, draw_circle=True, imshow_kwargs={}, **kwargs):
        if draw_circle:
            radius = get_radius_union_intersection(height, prop_arc)
            shape = (radius, radius)
        else:
            shape = (width, height)

        super().__init__(np.array(shape), xy_coords_mean=xy_coords_mean, **kwargs)
        self.prop_arc = prop_arc
        self.draw_circle = draw_circle
        self.imshow_kwargs = imshow_kwargs
        self.width = width
        self.height = height

        if op == "intersection":
            self.plot_op_on_ax = plot_intersection_on_ax
        elif op == "union":
            self.plot_op_on_ax = plot_union_on_ax

    def add_to_canva(self, canva: "Canva"):
        return self.plot_op_on_ax(
            canva.ax, self.xy_coords_mean, width=self.width, height=self.height, prop_arc=self.prop_arc,
            draw_circle=self.draw_circle, **self.imshow_kwargs
        )


class ElementSymbolUnion(ElementSymbolUnionIntersection):

    def __init__(self, *args, **kwargs):
        super().__init__(op="union", *args, **kwargs)



class ElementSymbolIntersection(ElementSymbolUnionIntersection):

    def __init__(self, *args, **kwargs):
        super().__init__(op="intersection", *args, **kwargs)



class ElementSymbolDilation(Element):

    def __init__(self, xy_coords_mean, radius=1, imshow_kwargs={}, **kwargs):
        super().__init__((radius, radius), xy_coords_mean=xy_coords_mean, **kwargs)
        self.radius = radius
        self.imshow_kwargs = imshow_kwargs
        self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')

    def add_to_canva(self, canva: "Canva"):
        return plot_dilation_on_ax(
            canva.ax, self.xy_coords_mean, radius=self.radius, **self.imshow_kwargs
        )


class ElementSymbolErosion(Element):

    def __init__(self, xy_coords_mean, radius=1, imshow_kwargs={}, **kwargs):
        super().__init__((radius, radius), xy_coords_mean=xy_coords_mean, **kwargs)
        self.radius = radius
        self.imshow_kwargs = imshow_kwargs
        self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')

    def add_to_canva(self, canva: "Canva"):
        return plot_erosion_on_ax(
            canva.ax, self.xy_coords_mean, radius=self.radius, **self.imshow_kwargs
        )
