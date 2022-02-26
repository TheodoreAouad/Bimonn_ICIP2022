import pathlib
from PIL import Image
from os.path import join
import os

import numpy as np

from general.utils import list_dir_joined
from deep_morpho.models import LightningBiMoNN


PATH_OUT = "weights_png/sandbox"
pathlib.Path(PATH_OUT).mkdir(exist_ok=True, parents=True)


def save_img(ar, savepath):
    ar = np.uint8(ar * 255)
    img = Image.fromarray(ar).resize((50, 50), resample=Image.NEAREST)
    img.save(savepath)


version_dict = {"version_0": "disk", "version_1": "stick", "version_2": "cross"}

paths_tb = {
    'diskorect': "deep_morpho/results/ICIP_2022/sandbox/4/diskorect",
    'mnist': "deep_morpho/results/ICIP_2022/sandbox/5/mnist",
    'inverted_mnist': "deep_morpho/results/ICIP_2022/sandbox/5/inverted_mnist",
}

# DILATION
for dataset in ['diskorect', 'mnist', 'inverted_mnist']:
    for op in ['dilation', 'erosion']:
        # pathlib.Path(join(PATH_OUT, dataset, op)).mkdir(exist_ok=True, parents=True)
        for tb_version in version_dict.keys():
            figname = f"{version_dict[tb_version]}.png"

            tb_path = join(paths_tb[dataset], op, "bisel", tb_version)
            path_weights = os.listdir(join(tb_path, "checkpoints"))[0]

            model = LightningBiMoNN.load_from_checkpoint(join(tb_path, 'checkpoints', path_weights))

            save_img(
                model.model.layer1.normalized_weight.detach().cpu()[0, 0].numpy(),
                join(PATH_OUT, f"{dataset}_{op}_{figname}")
            )


    for op in ['opening', 'closing']:
        # pathlib.Path(join(PATH_OUT, dataset, op)).mkdir(exist_ok=True, parents=True)
        for tb_version in version_dict.keys():
            figname1 = f"{version_dict[tb_version]}1.png"
            figname2 = f"{version_dict[tb_version]}2.png"

            tb_path = join(paths_tb[dataset], op, "bisel", tb_version)
            path_weights = os.listdir(join(tb_path, "checkpoints"))[0]

            model = LightningBiMoNN.load_from_checkpoint(join(tb_path, 'checkpoints', path_weights))

            save_img(
                model.model.layer1.normalized_weight.detach().cpu()[0, 0].numpy(),
                join(PATH_OUT, f"{dataset}_{op}_{figname1}")
            )
            save_img(
                model.model.layer2.normalized_weight.detach().cpu()[0, 0].numpy(),
                join(PATH_OUT, f"{dataset}_{op}_{figname2}")
            )
