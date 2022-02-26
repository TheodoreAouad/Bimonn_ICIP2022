import numpy as np
import torch.nn as nn
import torch.optim as optim

from deep_morpho.datasets.generate_forms3 import get_random_diskorect_channels
from deep_morpho.loss import MaskedMSELoss, MaskedDiceLoss, MaskedBCELoss
from general.utils import dict_cross
from .args_morp_ops_mnist import morp_operations as morp_operations_mnist
from .args_morp_ops_diskorect import morp_operations as morp_operations_diskorect

loss_dict = {"MaskedMSELoss": MaskedMSELoss, "MaskedDiceLoss": MaskedDiceLoss, "MaskedBCELoss": MaskedBCELoss,}

all_args = {}

all_args['n_try'] = [0]
# all_args['n_try'] = range(1, 11)

all_args['experiment_name'] = [
    'bimonn',
]


# DATA ARGS

all_args['preprocessing'] = [  # for axspa roi
    None,
]
all_args['dataset_path'] = [
    # 'data/deep_morpho/dataset_0',
    'generate',
]
all_args['in_ram'] = [
    # False,
    True,
]
all_args['random_gen_fn'] = [
    get_random_diskorect_channels
]
all_args['random_gen_args'] = [
    {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02}
    # {'size': (50, 50), 'n_shapes': 30, 'max_shape': (15, 15), 'p_invert': 0.5, 'n_holes': 15, 'max_shape_holes': (7, 7)}

]
all_args['n_inputs'] = [
    3_000_000,
    # 100_000,
    # 70000,
]
all_args['train_test_split'] = [(1, 1, 0)]


# TRAINING ARGS
all_args['learning_rate'] = [
    1e-2,
    # 10,
]

# if max_plus, then the loss is MSELoss
all_args['loss'] = [
    # "MaskedBCELoss",
    # nn.BCEWithLogitsLoss(),
    # "MaskedMSELoss",
    "MaskedDiceLoss",
]
all_args['optimizer'] = [
    optim.Adam,
    # optim.SGD
]
all_args['batch_size'] = [256]
all_args['num_workers'] = [
    20,
    # 0,
]
all_args['freq_imgs'] = [500]
all_args['n_epochs'] = [20]


# MODEL ARGS
# all_args['n_atoms'] = [
#     # 'adapt',
#     4,
# ]
all_args['atomic_element'] = [
    "bisel",
]
all_args['force_lui_identity'] = [True]
all_args['kernel_size'] = [
    # 7,
    "adapt",
]
all_args['channels'] = [
    'adapt',
]
all_args['init_weight_mode'] = [
    # "identity",
    # "normal_identity",
    "conv"
]
all_args['activation_P'] = [4]
all_args['constant_activation_P'] = [True]
all_args['constant_P_lui'] = [False]
all_args['constant_weight_P'] = [True]
all_args['threshold_mode'] = [
    # 'arctan',
    # 'sigmoid',
    'tanh',
    # 'erf',
    # "identity",
]
all_args["alpha_init"] = [0]

all_args['share_weights'] = [False]
all_args['do_thresh_penalization'] = [False]
all_args['args_thresh_penalization'] = [{
    'coef': .005,
    'degree': 4,
    'detach_weights': True,
}]
all_args['first_batch_pen'] = [1]


#########################################


all_args['dataset_type'] = ['diskorect']
all_args['morp_operation'] = morp_operations_diskorect
all_args_diskorect = dict_cross(all_args)


all_args['dataset_type'] = ['mnist']
all_args['morp_operation'] = morp_operations_mnist
all_args['mnist_args'] = [
    {"threshold": 30, "size": (50, 50), "invert_input_proba": 0}
]
all_args_mnist = dict_cross(all_args)

all_args['mnist_args'] = [
    {"threshold": 30, "size": (50, 50), "invert_input_proba": 1}
]
all_args_inverted_mnist = dict_cross(all_args)

###########
# DATASET #
###########

all_args = (
    all_args_diskorect +
    all_args_mnist +
    all_args_inverted_mnist +
    []
)


for idx, args in enumerate(all_args):

    # args['kernel_size'] = 'adapt'
    args['n_atoms'] = 'adapt'
    args['experiment_subname'] = f"{args['dataset_type']}/{args['morp_operation'].name}"


    if args["kernel_size"] == "adapt":
        args["kernel_size"] = args["morp_operation"].selems[0][0][0].shape[0]



    if args['channels'] == 'adapt':
        args['channels'] = args['morp_operation'].in_channels + [args['morp_operation'].out_channels[-1]]

    if args["n_atoms"] == 'adapt':
        args['n_atoms'] = len(args['morp_operation'])
        if args['atomic_element'] in ['cobise', 'cobisec']:
            args['n_atoms'] = max(args['n_atoms'] // 2, 1)


    if args['dataset_type'] == "diskorect":
        if args['morp_operation'].name in ['erosion', "dilation"]:
            args['n_inputs'] = 2_000_000
            args['learning_rate'] = 1e-2
        args['n_epochs'] = 1
        args["random_gen_args"] = args["random_gen_args"].copy()
        args["random_gen_args"]["border"] = (args["kernel_size"]//2 + 1, args["kernel_size"]//2 + 1)
        args['random_gen_args']['size'] = args['random_gen_args']['size'] + (args["morp_operation"].in_channels[0],)

    if args['dataset_type'] == "mnist":

        if args['mnist_args']['invert_input_proba'] == 1:
            args['experiment_subname'] = f"inverted_{args['experiment_subname']}"

        args['n_inputs'] = 70_000
        if args['morp_operation'].name in ['erosion', 'dilation']:
            args['n_epochs'] = 30
            args['learning_rate'] = 1e-1

        args['loss'] = "MaskedMSELoss"

        if args['morp_operation'].name == "dilation":
            args['n_epochs'] = 3

    args['loss'] = loss_dict[args['loss']](border=np.array([args['kernel_size'] // 2, args['kernel_size'] // 2]))
