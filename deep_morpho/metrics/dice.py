import numpy as np


def dice(y_true, y_pred, threshold=.5, SMOOTH=1e-6,):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    if y_true.ndim == 2:
        y_true = y_true.unsqueeze(0)
    if y_pred.ndim == 2:
        y_pred = y_pred.unsqueeze(0)

    if y_true.ndim == 4:
        return np.stack([dice(y_true[:, k, ...], y_pred[:, k, ...], threshold, SMOOTH) for k in range(y_true.shape[1])], axis=0).mean(0)

    targets = (y_true != 0)
    if threshold is None:
        outputs = y_pred != 0
    else:
        outputs = y_pred > threshold

    intersection = (outputs & targets).float().sum((1, 2))

    return (
        (2*intersection + SMOOTH) / (targets.sum((1, 2)) + outputs.sum((1, 2)) + SMOOTH)
    ).detach().cpu().numpy()


def masked_dice(ytrue, ypred, border, *args, **kwargs):
    if border[0] != 0:
        ypred = ypred[..., border[0]:-border[0], :]
        ytrue = ytrue[..., border[0]:-border[0], :]

    if border[1] != 0:
        ypred = ypred[..., border[1]:-border[1]]
        ytrue = ytrue[..., border[1]:-border[1]]

    return dice(ytrue, ypred, *args, **kwargs)
