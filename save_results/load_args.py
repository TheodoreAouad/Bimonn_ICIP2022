import re
from typing import Any, Dict


def load_args(path: str) -> Dict:
    all_keys_line = [
        "experiment_name",
        "experiment_subname",
        "name",
        "dataset_type",
        "dataset_path",
        "n_inputs",
        "learning_rate",
        "batch_size",
        "num_workers",
        "freq_imgs",
        "n_epochs",
        "n_atoms",
        "atomic_element",
        "kernel_size",
        "init_weight_identity",
        "activation_P",
        "constant_activation_P",
        "constant_weight_P",
        "threshold_mode",
        "alpha_init",
        "share_weights",
        "loss",
    ]

    with open(path, "r") as f:
        yaml_str = f.read()

    args = {}

    # args['loss'] = parse_yaml_dict_loss(yaml_str)
    args['optimizer'] = parse_yaml_dict_optimizer(yaml_str)
    args['operations'] = parse_yaml_dict_operations(yaml_str)

    for key in all_keys_line:
        args[key] = parse_yaml_dict_key_line(yaml_str, key)

    return args


def regex_find_or_none(regex: str, st: str, group_nb: int = -1):
    exps = re.findall(regex, st)
    if len(exps) == 0:
        return None
    assert len(exps) == 1, exps

    # for multiple parenthesis, we have to select the group. If there is only one group, -1
    if group_nb == -1:
        return exps[0]
    return exps[0][group_nb]


def parse_yaml_dict_key_line(yaml_str: str, key: str) -> Any:
    return regex_find_or_none(f"( |^|\n){key}: ([^\n]+)\n", yaml_str, group_nb=1)


def parse_yaml_dict_loss(yaml_str: str) -> Any:
    return regex_find_or_none(r"\n?loss[^\n]+\.(\w+)\n", yaml_str)


def parse_yaml_dict_optimizer(yaml_str: str) -> Any:
    return regex_find_or_none(r"\n?optimizer[^\n]+\.(\w+)[ \n]", yaml_str)


def parse_yaml_dict_operations(yaml_str: str) -> Any:
    idx0 = yaml_str.find('operations')
    if idx0 == -1:
        return None

    idx = idx0 + len('operations:\n')
    all_ops = []
    while True and idx < len(yaml_str):
        sent = ""
        while yaml_str[idx] == " ":
            idx += 1

        if yaml_str[idx] != "-":
            break

        while yaml_str[idx] != "\n":
            sent += yaml_str[idx]
            idx += 1

        all_ops.append(sent[2:])
        idx += 1

    return all_ops
