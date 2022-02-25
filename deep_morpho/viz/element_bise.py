import numpy as np
from matplotlib.patches import Polygon

from general.nn.viz import ElementGrouper, ElementArrow, ElementImage, ElementSymbolDilation, ElementSymbolErosion, ElementCircle


MAX_WIDTH_COEF = 1


class ElementBiseWeightsChan(ElementImage):

    def __init__(self, model, chout=0, *args, **kwargs):
        self.model = model
        self.chout = chout
        super().__init__(image=None, *args, **kwargs)

        self.imshow_kwargs['vmin'] = self.imshow_kwargs.get('vmin', 0)
        self.imshow_kwargs['vmax'] = self.imshow_kwargs.get('vmax', 1)
        self.imshow_kwargs['cmap'] = self.imshow_kwargs.get('cmap', 'gray')

    @property
    def image(self):
        return self.model._normalized_weight[self.chout, 0].detach().cpu().numpy()


class ElementBiseSelemChan(ElementGrouper):
    operation_elements_dict = {'dilation': ElementSymbolDilation, "erosion": ElementSymbolErosion}

    def __init__(self, model, chout=0, v1=0, v2=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.chout = chout
        self.v1 = v1
        self.v2 = v2
        self.kernel_shape = self.model.weight.shape[-2:]

        selem, operation = self.model.find_selem_and_operation_chan(self.chout, self.v1, self.v2)

        if selem is None:
            self.selem_element = ElementCircle(radius=self.kernel_shape[-1] / 2, **kwargs)
            self.operation_element = None

        else:
            self.selem_element = ElementImage(selem, imshow_kwargs={"interpolation": "nearest"}, **kwargs)
            self.operation_element = self.operation_elements_dict[operation](
                radius=2, xy_coords_mean=self.selem_element.xy_coords_mean + np.array([0, self.kernel_shape[-1] / 2 + 2])
            )
            self.add_element(self.operation_element, key="operation")
        self.add_element(self.selem_element, key="selem")

    # def add_to_canva(self, canva: "Canva"):
    #     self.selem_element.add_to_canva(canva)
    #     if self.operation_element is not None:
    #         self.operation_element.add_to_canva(canva)


# class ElementBise(Element):

#     def __init__(self, model, input_elements, imshow_kwargs={}, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.model = model
#         self.input_elements = input_elements
#         self.imshow_kwargs = imshow_kwargs

#         self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')

#     def add_to_canva(self, canva: "Canva"):
#         canva.ax.add_patch(Polygon(np.stack([
#             self.xy_coords_botleft, self.xy_coords_topleft, self.xy_coords_midright
#         ]), closed=True, fill=False, **self.imshow_kwargs))
#         self.link_input_lui(canva)


#     def link_input_lui(self, canva, max_width_coef=MAX_WIDTH_COEF):
#         coefs = self.model.positive_weight[0].detach().cpu().numpy()
#         coefs = coefs / coefs.max()

#         for elt_idx, elt in enumerate(self.input_elements):
#             canva.add_element(ElementArrow.link_elements(
#                 elt, self, width=coefs[elt_idx] * max_width_coef,
#             ))
