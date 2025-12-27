import torch
import numpy as np
import pandas as pd
import scipy.signal


def add_gaussian_noise(tensor, mean=0.0, std=0.07):
    """
    Adds Gaussian noise to a tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor (B, T, 8)
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the noise.

    Returns:
        torch.Tensor: Noisy tensor.
    """
    noise = torch.randn_like(tensor) * std + mean
    return tensor + noise


def clean_data_tensor(tensor, thre=0.0002): # 0.0002 or 0.1
    """
    Cleans a tensor by setting values to zero if variance is below a threshold.

    Args:
        tensor (torch.Tensor): Input tensor (can be on GPU).
        thre (float): Variance threshold. If variance < thre, returns all zeros.

    Returns:
        torch.Tensor: Processed tensor, keeping device consistent.
    """
    # Compute variance along the last dimension (per-channel processing)
    var = torch.var(tensor, dim=0, keepdim=True)

    # Mask for low-variance channels
    mask = var >= thre

    # Return tensor where low-variance channels are set to zero
    return tensor * mask.float()

def clean_channels_np(array, threshold=0.1):
    """
    Sets entire channels to zero if the min-max range is below a threshold.

    Args:
        array (np.ndarray): Input array of shape (T, C).
        threshold (float): Minimum required min-max difference to keep values.

    Returns:
        np.ndarray: Modified array with low-variation channels zeroed out.
    """
    # Compute min and max along time dimension (T)
    min_vals = np.min(array, axis=0)  # Shape (C,)
    max_vals = np.max(array, axis=0)  # Shape (C,)

    # Compute min-max difference per channel
    min_max_diff = max_vals - min_vals  # Shape (C,)

    # Create a mask where channels with low variation are zeroed
    mask = (min_max_diff >= threshold).astype(float)  # Shape (C,)

    # Apply mask to zero out low-variation channels
    return array * mask[np.newaxis, :]  # Shape (T, C)


def clean_channels(tensor, threshold=0.2):
    """
    Sets entire channels to zero if the min-max range is below a threshold.

    Args:
        tensor (torch.Tensor): Input tensor of shape (T, C)
        threshold (float): Minimum required min-max difference to keep values.

    Returns:
        torch.Tensor: Modified tensor with low-variation channels zeroed out.
    """    

    # Compute min and max along time dimension (T)
    min_vals, _ = torch.min(tensor, dim=0)  # (C,)
    max_vals, _ = torch.max(tensor, dim=0)  # (C,)

    # Compute min-max range per channel
    min_max_diff = max_vals - min_vals  # (C,)

    # Create a mask where channels with low variation are zeroed
    mask = (min_max_diff >= threshold).float()  # (C,)

    # Apply mask to zero out low-variation channels
    return tensor * mask.unsqueeze(0)  # (T, C)

def smooth_(tensor, window_length=31, polyorder=2):
    """
    Applies Savitzky-Golay filter to each channel in a tensor (T, C).
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (T, C).
        window_length (int): The length of the filter window (must be odd).
        polyorder (int): The order of the polynomial used to fit the samples.
    
    Returns:
        torch.Tensor: Smoothed tensor with the same shape and device.
    """
    device = tensor.device  # Store original device (CPU/GPU)
    tensor_np = tensor.cpu().numpy()  # Convert to NumPy for filtering

    # Apply SG filter to each channel (axis=0 is time dimension)
    smoothed_np = scipy.signal.savgol_filter(tensor_np, window_length=window_length, polyorder=polyorder, axis=0)

    # Convert back to PyTorch tensor and restore device
    smoothed_tensor = torch.tensor(smoothed_np, device=device, dtype=tensor.dtype)
    
    return smoothed_tensor

def align_sequence(y_pred, y_gt, args):
    y_pred = y_pred.reshape(-1, y_gt.shape[1])[(args.INPUT_MOTION_LENGTH-1):]
    min_len = min(y_pred.shape[0], y_gt.shape[0]) 
    y_pred = y_pred[args.INPUT_MOTION_LENGTH:min_len]
    y_gt = y_gt[args.INPUT_MOTION_LENGTH:min_len]
    assert y_pred.shape[0] == y_gt.shape[0]

    return y_pred, y_gt


def process_predictions(pred_mean, pred_var, y_gt, threshold=0.4):
    """
    Process predictions based on variance for tensors with shape (T, C).

    Args:
        pred_mean (torch.Tensor): Predicted mean, shape (T, C).
        pred_var (torch.Tensor): Predicted variance, shape (T, C).
        y_gt (torch.Tensor): Ground truth, shape (T, C).
        threshold (float): Variance threshold to classify unreliable predictions.

    Returns:
        tuple: (version1_output, version2_output)
            - version1_output: Tuple of pred_mean and y_gt for indexes where variance > threshold.
            - version2_output: Tuple of full-length arrays with high-variance values replaced and interpolated.
    """
    # Convert to NumPy for easier processing
    pred_mean_np = pred_mean.detach().cpu().numpy()
    pred_var_np = pred_var.detach().cpu().numpy()
    y_gt_np = y_gt.detach().cpu().numpy()

    # Version 1: Find indices with variance < threshold
    low_var_indices = np.where(pred_var_np < threshold)  # Tuple of arrays (time_indices, feature_indices)
    version1_pred_mean = pred_mean_np[low_var_indices]
    version1_y_gt = y_gt_np[low_var_indices]

    # Version 2: Replace high-variance predictions with NaN and interpolate
    high_var_indices = np.where(pred_var_np > threshold)
    version2_pred_mean = np.copy(pred_mean_np)
    version2_y_gt = np.copy(y_gt_np)
    version2_pred_mean[high_var_indices] = np.nan
    version2_y_gt[high_var_indices] = np.nan

    # Interpolation
    for c in range(pred_mean_np.shape[1]):  # Iterate over features
        # Interpolate pred_mean
        version2_pred_mean[:, c] = pd.Series(version2_pred_mean[:, c]).interpolate(
            method='linear', limit_direction='both'
        ).to_numpy()
        # Interpolate y_gt
        version2_y_gt[:, c] = pd.Series(version2_y_gt[:, c]).interpolate(
            method='linear', limit_direction='both'
        ).to_numpy()

    # Convert results back to PyTorch tensors
    version1_pred_mean = torch.tensor(version1_pred_mean, device=pred_mean.device)
    version1_y_gt = torch.tensor(version1_y_gt, device=pred_mean.device)
    version2_pred_mean = torch.tensor(version2_pred_mean, device=pred_mean.device)
    version2_y_gt = torch.tensor(version2_y_gt, device=pred_mean.device)

    return version1_pred_mean, version1_y_gt, version2_pred_mean, version2_y_gt
