from typing import Dict

from matplotlib.patches import Circle

from .element import Element


class ElementCircle(Element):

    def __init__(self, radius: float, imshow_kwargs: Dict = {}, *args, **kwargs):
        super().__init__(shape=(2*radius, 2*radius), *args, **kwargs)
        self.radius = radius

        self.imshow_kwargs = imshow_kwargs
        self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')
        self.imshow_kwargs['fill'] = self.imshow_kwargs.get('fill', False)

    def add_to_canva(self, canva: "Canva"):
        canva.ax.add_patch(Circle(xy=self.xy_coords_mean, radius=self.radius, **self.imshow_kwargs))
