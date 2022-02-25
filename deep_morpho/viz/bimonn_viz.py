import pathlib
import numpy as np

from general.nn.viz import Canva, ElementArrow, ElementImage, ElementGrouper, ElementCircle
from deep_morpho.models import BiMoNN
from general.nn.viz import (
    ElementSymbolDilation, ElementSymbolErosion, ElementSymbolIntersection, ElementSymbolUnion
)
from general.nn.viz.element import Element
from .element_lui import ElementLui
from .element_bise import ElementBiseSelemChan, ElementBiseWeightsChan


LUI_HORIZONTAL_FACTOR = 3
LUI_RADIUS_FACTOR = 1
INPUT_RADIUS_FACTOR = .1

NEXT_LAYER_HORIZONTAL_FACTOR = 1
FIRST_LAYER_HORIZONTAL_FACTOR = 1 / INPUT_RADIUS_FACTOR * .5


class BimonnVizualiser:
    operation_elements = {
        'union': ElementSymbolUnion,
        'intersection': ElementSymbolIntersection,
        'erosion': ElementSymbolErosion,
        'dilation': ElementSymbolDilation,
    }

    def __init__(self, model: BiMoNN, mode="weight"):
        self.model = model
        self.canva = None
        self.layer_groups = []

        assert mode in ['weight', 'selem'], "mode must be either 'weight' or 'selem'."
        self.mode = mode

        self.box_height = self._infer_box_height()

    def _infer_box_height(self):
        return 2 * (
            np.array(self.model.in_channels) *
            np.array(self.model.out_channels) *
            self.max_selem_shape
        ).max()

    @property
    def max_selem_shape(self):
        return np.array(self.model.kernel_size).max(1)

    def draw(self, **kwargs):
        self.canva = Canva(**kwargs)
        cursor = 0
        prev_elements = []
        for layer_idx in range(-1, len(self)):
            group, cur_elements = self.get_layer_group(layer_idx)
            group.set_xy_coords_midleft(np.array([cursor, 0]))

            self.canva.add_element(group, key=f'layer_{layer_idx}')

            for chin, elt in enumerate(prev_elements):
                for chout in range(self.model.out_channels[layer_idx]):
                    self.canva.add_element(ElementArrow.link_elements(
                        elt, group.elements[f"group_layer_{layer_idx}_chout_{chout}"].elements[f"selem_layer_{layer_idx}_chout_{chout}_chin_{chin}"]
                    ))

            if layer_idx == -1:
                cursor += group.shape[0] * (1 + FIRST_LAYER_HORIZONTAL_FACTOR)
            else:
                cursor += group.shape[0] * (1 + NEXT_LAYER_HORIZONTAL_FACTOR)

            prev_elements = cur_elements

        return self.canva


    def get_input_layer(self):
        layer_group = ElementGrouper()
        n_elts = self.model.in_channels[0]
        coords = np.linspace(0, self.box_height, 2*n_elts + 1)[1::2]

        for elt_idx, coord in enumerate(coords):
            layer_group.add_element(ElementCircle(
                xy_coords_mean=np.array([0, coord]),
                radius=INPUT_RADIUS_FACTOR * self.box_height / (2 * n_elts),
                imshow_kwargs={"fill": True},
            ), key=f"input_chan_{elt_idx}")

        return layer_group, [layer_group.elements[f"input_chan_{elt_idx}"] for elt_idx in range(n_elts)]


    def _get_height_group(self, coords_group, n_groups):
        if n_groups > 1:
            return (coords_group[1] - coords_group[0])*.7
        return self.box_height

    def _get_coords_selem(self, height_group, n_per_group):
        if n_per_group == 1:
            return np.zeros(1)
        return np.linspace(0, height_group, 2*n_per_group + 1)[1::2]

    def add_bise_to_group(self, group, layer_idx, chout, chin, coord_selem) -> Element:
        """ Add bise to the group. Returns the element to link to the LUI.
        """
        # selem = self.model.layers[layer_idx].normalized_weights[chout, chin].detach().cpu().numpy()
        key_selem = f"selem_layer_{layer_idx}_chout_{chout}_chin_{chin}"

        # cur_elt = ElementImage(
        #     selem,
        #     imshow_kwargs={"cmap": "gray", "interpolation": "nearest", "vmin": 0, "vmax": 1},
        #     xy_coords_mean=(0, coord_selem)
        # )
        bise_model = self.model.layers[layer_idx].bises[chin]

        if self.mode == "weight":
            cur_elt = ElementBiseWeightsChan(model=bise_model, chout=chout, xy_coords_mean=np.array([0, coord_selem]))
            lui_link_elt = cur_elt

        elif self.mode == "selem":
            cur_elt = ElementBiseSelemChan(model=bise_model, chout=chout, xy_coords_mean=np.array([0, coord_selem]))
            lui_link_elt = cur_elt.elements['selem']

        group.add_element(cur_elt, key=key_selem)

        return lui_link_elt

    def get_layer_group(self, layer_idx):
        if layer_idx == -1:
            return self.get_input_layer()

        layer_group = ElementGrouper()

        n_groups = self.model.out_channels[layer_idx]
        n_per_group = self.model.in_channels[layer_idx]

        coords_group = np.linspace(0, self.box_height, 2*n_groups + 1)[1::2]
        height_group = self._get_height_group(coords_group, n_groups)

        for chout, coord_group in enumerate(coords_group):

            coords_selem = self._get_coords_selem(height_group, n_per_group)
            subgroup = ElementGrouper()

            input_lui_elements = []

            # add bises
            for coord_selem_idx, chin in enumerate(range(n_per_group)):
                coord_selem = coords_selem[coord_selem_idx]
                link_lui_elt = self.add_bise_to_group(subgroup, layer_idx, chout, chin, coord_selem)
                input_lui_elements.append(link_lui_elt)

            # add lui layers
            shape = LUI_RADIUS_FACTOR * np.array([self.max_selem_shape[layer_idx], self.max_selem_shape[layer_idx]])
            subgroup.add_element(ElementLui(
                model=self.model.layers[layer_idx].luis[chout],
                input_elements=input_lui_elements,
                xy_coords_mean=(LUI_HORIZONTAL_FACTOR * self.max_selem_shape[layer_idx], (coords_selem).mean()),
                shape=shape,
                imshow_kwargs={"color": "k"},
                mode=self.mode,
            ), key=f"lui_layer_{layer_idx}_chout_{chout}")

            subgroup.set_xy_coords_mean(np.array([0, coord_group]))
            layer_group.add_element(
                subgroup,
                key=f"group_layer_{layer_idx}_chout_{chout}"
            )

        return layer_group, [
            layer_group.elements[f"group_layer_{layer_idx}_chout_{chout}"].elements[f"lui_layer_{layer_idx}_chout_{chout}"]
            for chout in range(n_groups)]

    def __len__(self):
        return len(self.model)

    def save_fig(self, savepath: str, **kwargs):
        pathlib.Path(savepath).parent.mkdir(exist_ok=True, parents=True)
        self.draw(**kwargs)
        self.canva.fig.savefig(savepath)

    def get_fig(self, **kwargs):
        self.draw(**kwargs)
        return self.canva.fig
