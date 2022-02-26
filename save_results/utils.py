from typing import List, Dict

import base64
import io

import matplotlib.pyplot as plt


def plot_to_base64(figure):
    my_stringIObytes = io.BytesIO()
    figure.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    return base64.b64encode(my_stringIObytes.read()).decode()


def plot_to_html(figure):
    if figure is None:
        return ""
    base64_fig = plot_to_base64(figure)
    return f"<img src='data:image/png;base64, {base64_fig}'>"


def detect_identical_values(all_args: List[Dict]) -> (List[str], List[str]):
    """
    Given a list of dicts, gives the keys where the values are the same.
    """

    same_values = []

    for key in all_args[0].keys():
        bad_key = False
        for args in all_args[1:]:
            if args[key] != all_args[0][key]:
                bad_key = True
                break

        if not bad_key:
            same_values.append(key)

    return same_values, list(set(all_args[0].keys()).difference(same_values))


def load_png_as_fig(path: str, **kwargs):
    ar = plt.imread(path)
    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.imshow(ar)
    ax.grid(False)
    ax.axis('off')
    return fig
