import numpy as np

from general.nn.viz import Canva, ElementArrow, ElementImage, ElementGrouper, ElementCircle
from general.nn.viz import (
    ElementSymbolDilation, ElementSymbolErosion, ElementSymbolIntersection, ElementSymbolUnion
)


LUI_HORIZONTAL_FACTOR = 3
LUI_RADIUS_FACTOR = .5
INPUT_RADIUS_FACTOR = .4

NEXT_LAYER_HORIZONTAL_FACTOR = .5
FIRST_LAYER_HORIZONTAL_FACTOR = 1.3


class MorpOperationsVizualiser:
    operation_elements = {
        'union': ElementSymbolUnion,
        'intersection': ElementSymbolIntersection,
        'erosion': ElementSymbolErosion,
        'dilation': ElementSymbolDilation,
    }

    def __init__(self, morp_operations: "ParallelMorpOperations", ):
        self.morp_operations = morp_operations
        self.canva = None
        self.layer_groups = []

        self.box_height = self._infer_box_height()

    def _infer_box_height(self):
        return 2 * (
            np.array(self.morp_operations.in_channels) *
            np.array(self.morp_operations.out_channels) *
            self.morp_operations.max_selem_shape
        ).max()

    def draw(self, **kwargs):
        self.canva = Canva(**kwargs)
        cursor = 0
        prev_elements = []
        for layer_idx in range(-1, len(self)):
        # for layer_idx in range(len(self)):
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


    def get_layer_group(self, layer_idx):
        if layer_idx == -1:
            return self.get_input_layer()

        layer_group = ElementGrouper()

        n_groups = self.model.out_channels[layer_idx]
        n_per_group = self.model.in_channels[layer_idx]

        coords_group = np.linspace(0, self.box_height, 2*n_groups + 1)[1::2]

        if n_groups > 1:
            height_group = (coords_group[1] - coords_group[0]) * .9
        else:
            height_group = self.box_height

        for group_idx, coord_group in enumerate(coords_group):
            selem_chans = self.model.selem_args[layer_idx][group_idx][-1]
            selem_chans = range(n_per_group) if selem_chans == "all" else selem_chans

            coords_selem = np.linspace(0, height_group, 2*len(selem_chans) + 1)[1::2]
            subgroup = ElementGrouper()

            # add aggregations
            aggreg = self.model.operation_names[layer_idx][group_idx][-1]
            width, height = LUI_RADIUS_FACTOR * np.array(self.model.max_selem_shape[layer_idx])
            subgroup.add_element(self.operation_elements[aggreg](
                xy_coords_mean=(LUI_HORIZONTAL_FACTOR * self.model.max_selem_shape[layer_idx][1], (coords_selem).mean()),
                width=width, height=height,
                imshow_kwargs={"color": "k"}
            ), key=f"aggregation_layer_{layer_idx}_chout_{group_idx}")

            # add selem operations
            for coord_selem_idx, selem_idx in enumerate(selem_chans):
                coord_selem = coords_selem[coord_selem_idx]
                selem = self.model.selems[layer_idx][group_idx][selem_idx]
                op_name = self.model.operation_names[layer_idx][group_idx][selem_idx]
                key_selem = f"selem_layer_{layer_idx}_chout_{group_idx}_chin_{selem_idx}"

                subgroup.add_element(ElementImage(
                    selem,
                    imshow_kwargs={"cmap": "gray", "interpolation": "nearest"},
                    xy_coords_mean=(0, coord_selem)
                ), key=key_selem)

                subgroup.add_element(self.operation_elements[op_name](
                    xy_coords_mean=(0, coord_selem + selem.shape[1] / 2 + 2),
                    radius=2,
                    imshow_kwargs={"color": "k"}
                ), key=f"operation_layer_{layer_idx}_chout_{group_idx}_chin_{selem_idx}")

                # add connections between selem and aggregation
                subgroup.add_element(ElementArrow.link_elements(
                    subgroup.elements[key_selem],
                    subgroup.elements[f"aggregation_layer_{layer_idx}_chout_{group_idx}"]
                ))

            subgroup.set_xy_coords_mean(np.array([0, coord_group]))
            layer_group.add_element(
                subgroup,
                key=f"group_layer_{layer_idx}_chout_{group_idx}"
            )

        return layer_group, [
            layer_group.elements[f"group_layer_{layer_idx}_chout_{group_idx}"].elements[f"aggregation_layer_{layer_idx}_chout_{group_idx}"]
            for group_idx in range(n_groups)]



    @property
    def model(self):
        return self.morp_operations

    def __len__(self):
        return len(self.morp_operations)
