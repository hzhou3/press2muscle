import torch

def smooth_loss(y_pred, y_true):
    dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
    dy_true = y_true[:, 1:] - y_true[:, :-1]
    return torch.mean((dy_pred - dy_true) ** 2)
