import pathlib
from os.path import join
import os

import numpy as np
import pandas as pd

from general.utils import list_dir_joined, load_json
from deep_morpho.models import LightningBiMoNN


PATH_OUT = "deep_morpho/metric_results"
pathlib.Path(PATH_OUT).mkdir(exist_ok=True, parents=True)


version_dict = {"version_0": "disk", "version_1": "stick", "version_2": "cross"}

paths_tb = {
    'diskorect': "deep_morpho/results/ICIP_2022/sandbox/4/diskorect",
    'mnist': "deep_morpho/results/ICIP_2022/sandbox/5/mnist",
    'inverted_mnist': "deep_morpho/results/ICIP_2022/sandbox/5/inverted_mnist",
}


all_dfs = []

# DILATION
for dataset in ['diskorect', 'mnist', 'inverted_mnist']:
    for op in ['dilation', 'erosion', 'opening', 'closing']:
        # pathlib.Path(join(PATH_OUT, dataset, op)).mkdir(exist_ok=True, parents=True)
        for tb_version in version_dict.keys():
            figname = f"{version_dict[tb_version]}.png"

            tb_path = join(paths_tb[dataset], op, "bisel", tb_version)


            cur_dice = join(tb_path, 'observables', 'CalculateAndLogMetrics', 'metrics.json')
            cur_dice = eval(load_json(cur_dice)['dice'])

            cur_conv = join(tb_path, 'observables', 'ConvergenceMetrics', 'convergence_step.json')
            cur_conv = load_json(cur_conv)

            binary_selem = join(tb_path, 'observables', 'ConvergenceBinary', 'convergence_step.json')
            binary_selem = load_json(binary_selem)['bisel']

            all_dfs.append(pd.DataFrame(dict(
                **{
                    "tb_path": [tb_path],
                    "dataset": [dataset],
                    "operation": [op],
                    "selem": [version_dict[tb_version]],
                    "dice": [cur_dice],
                },
                **{f'convergence_dice_{state}': [v['dice']] for state, v in cur_conv.items()},
                **{f'convergence_selem_{eval(layer_idx)[0]}': [v] for layer_idx, v in binary_selem.items()},
            )))


all_dfs = pd.concat(all_dfs).reset_index(drop=True)
all_dfs.to_csv(join(PATH_OUT, "results.csv"), index=False)
