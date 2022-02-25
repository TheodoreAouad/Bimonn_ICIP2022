import numpy as np
from matplotlib.patches import Polygon

from general.nn.viz import Element, ElementArrow, ElementGrouper, ElementSymbolIntersection, ElementSymbolUnion


MAX_WIDTH_COEF = 1
OPERATION_FACTOR = .3


class ElementLuiCoefs(Element):

    def __init__(self, model, input_elements, imshow_kwargs={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.input_elements = input_elements
        self.imshow_kwargs = imshow_kwargs
        self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')

        self.widths = self.get_widths()
        # self.widths = np.zeros(100)

    def add_to_canva(self, canva: "Canva"):
        canva.ax.add_patch(Polygon(np.stack([
            self.xy_coords_botleft, self.xy_coords_topleft, self.xy_coords_midright
        ]), closed=True, fill=False, **self.imshow_kwargs))
        self.link_input_lui(canva)

    def get_widths(self, max_width_coef=MAX_WIDTH_COEF):
        coefs = self.model.positive_weight[0].detach().cpu().numpy()
        coefs = coefs / coefs.max()

        return coefs * max_width_coef

    def link_input_lui(self, canva,):
        for elt_idx, elt in enumerate(self.input_elements):
            canva.add_element(ElementArrow.link_elements(
                elt, self, width=self.widths[elt_idx],
            ))



class ElementLui(ElementGrouper):
    operation_element_dicts = {'intersection': ElementSymbolIntersection, 'union': ElementSymbolUnion}

    def __init__(self, model, input_elements, shape, imshow_kwargs={}, v1=None, v2=None, mode='weight', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.v1 = v1
        self.v2 = v2
        self.input_elements = input_elements
        self.imshow_kwargs = imshow_kwargs
        self.imshow_kwargs['color'] = self.imshow_kwargs.get('color', 'k')
        self.mode = mode

        self.element_lui_operation = None

        self.element_lui_coefs = ElementLuiCoefs(model, input_elements, imshow_kwargs, shape=shape, *args, **kwargs)
        self.add_element(self.element_lui_coefs, key="coefs")

        C, operation = self.model.find_set_and_operation_chan(0, v1, v2)
        if operation is not None:
            shape = self.element_lui_coefs.shape * OPERATION_FACTOR
            self.element_lui_operation = self.operation_element_dicts[operation](
                width=shape[0], height=shape[1],
                xy_coords_mean=self.element_lui_coefs.xy_coords_mean + np.array([0, self.element_lui_coefs.shape[-1] / 2 + 2])
            )
            self.add_element(self.element_lui_operation, key="operation")

            if self.mode == 'selem':
                self.element_lui_coefs.widths = C.astype(int)
