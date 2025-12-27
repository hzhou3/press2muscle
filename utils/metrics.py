import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
from utils.helpers import clean_data_tensor

all_metrics = [
    "mse",
    "rmse",
    "mape",
    # "r2",
    # "cosine_sim",
    "rho",
]


TO_DECOUPLE = False


def evaluate_prediction(args, metrics, pred, gt, fps, filename, prefix='pressure', to_decouple=False):
    global TO_DECOUPLE
    TO_DECOUPLE = to_decouple

    eval_log = {}
    for metric in metrics:
        eval_log[prefix+'_'+metric] = (
            get_metric_function(metric)(
                pred, gt, args.INPUT_MOTION_LENGTH
            ).cpu().numpy()
        )
    torch.cuda.empty_cache()
    return eval_log

def precision(preds, gt):
    """Computes precision, recall, and F1-score."""
    # preds = torch.argmax(preds, dim=1).cpu().numpy()  # Convert logits to class indices
    preds = preds.cpu().numpy() 
    gt = gt.cpu().numpy()  # Convert ground truth to numpy
    precision = precision_score(gt, preds, average='macro', zero_division=0)  # Use 'macro' for class-wise avg
    return torch.tensor(precision)


def recall(preds, gt):
    """Computes precision, recall, and F1-score."""
    # preds = torch.argmax(preds, dim=1).cpu().numpy()  # Convert logits to class indices
    preds = preds.cpu().numpy() 
    gt = gt.cpu().numpy()  # Convert ground truth to numpy
    recall = recall_score(gt, preds, average='macro', zero_division=0)
    return torch.tensor(recall)


def f1(preds, gt):
    """Computes precision, recall, and F1-score."""
    # preds = torch.argmax(preds, dim=1).cpu().numpy()  # Convert logits to class indices
    preds = preds.cpu().numpy() 
    gt = gt.cpu().numpy()  # Convert ground truth to numpy
    f1 = f1_score(gt, preds, average='macro')
    return torch.tensor(f1)



def mse(y_pred, y_gt, T):
    """
    Compute Mean Squared Error (MSE) per channel over cropped segments of length `T`.

    Returns:
        torch.Tensor: Shape (num_windows, 8), MSE for each channel per window.
    """
    N, C = y_pred.shape
    num_windows = N // T
    mse_vals = [
        torch.mean((y_pred[i * T:(i + 1) * T, :] - y_gt[i * T:(i + 1) * T, :]) ** 2, dim=0)
        for i in range(num_windows)
    ]
    ret = torch.stack(mse_vals)  # Shape (num_windows, 8)
    return torch.mean(torch.mean(ret, dim=1)) 

def rmse(y_pred, y_gt, T):
    """
    Compute Root Mean Squared Error (RMSE) averaged over channels and windows.

    Args:
        y_pred (torch.Tensor): Predicted values (N, 8)
        y_gt (torch.Tensor): Ground truth values (N, 8)
        T (int): Segment length for computing metrics.

    Returns:
        torch.Tensor: Single scalar RMSE value.
    """
    N, C = y_pred.shape
    num_windows = N // T
    ret = torch.stack([
        torch.sqrt(torch.mean((y_pred[i * T:(i + 1) * T, :] - y_gt[i * T:(i + 1) * T, :]) ** 2, dim=0))
        for i in range(num_windows)
    ])
    # 
    if TO_DECOUPLE:
        return torch.mean(ret, dim=0)
    else:
        return torch.mean(torch.mean(ret, dim=1)) 



def smape(y_pred, y_gt, T):
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE) per channel over cropped segments of length `T`,
    ensuring that ground truth (y_gt) being zero does not cause issues.

    Args:
        y_pred (torch.Tensor): Predicted values (N, 8).
        y_gt (torch.Tensor): Ground truth values (N, 8).
        T (int): Segment length for computing metrics.

    Returns:
        torch.Tensor: Scalar mean SMAPE across windows and channels.
    """
    N, C = y_pred.shape
    num_windows = N // T
    smape_vals = []

    for i in range(num_windows):
        y_pred_T = y_pred[i * T:(i + 1) * T, :]
        y_gt_T = y_gt[i * T:(i + 1) * T, :]

        # SMAPE formula: |y_pred - y_gt| / (|y_pred| + |y_gt| + epsilon)
        denominator = torch.abs(y_pred_T) + torch.abs(y_gt_T) + 1e-8  # Avoid division by zero
        smape_per_channel = torch.abs(y_pred_T - y_gt_T) / denominator  # Shape (T, 8)

        # Compute mean over time dimension (T)
        smape_per_channel = torch.mean(smape_per_channel, dim=0)  # Shape (8,)

        smape_vals.append(smape_per_channel)

    ret = torch.stack(smape_vals)  # Shape (num_windows, 8)

    # Compute the final mean over all windows and channels
    if TO_DECOUPLE:
        return torch.mean(ret, dim=0) * 100
    else:
        return torch.mean(ret) * 100




def rho(y_pred, y_gt, T):
    """
    Compute Pearson Correlation Coefficient (ρ) per channel over cropped segments of length `T`.
    If both `y_pred_T` and `y_gt_T` are all zeros for any channel, return ρ = 1 for that channel.

    Args:
        y_pred (torch.Tensor): Predicted values (N, 8).
        y_gt (torch.Tensor): Ground truth values (N, 8).
        T (int): Segment length for computing metrics.

    Returns:
        torch.Tensor: Single scalar mean Pearson correlation across windows and channels.
    """
    N, C = y_pred.shape
    num_windows = N // T
    rho_vals = []

    y_pred = clean_data_tensor(y_pred)
    y_gt = clean_data_tensor(y_gt)

    for i in range(num_windows):
        y_pred_T = y_pred[i * T:(i + 1) * T, :]
        y_gt_T = y_gt[i * T:(i + 1) * T, :]

        # Compute mean per channel
        mean_pred_T = torch.mean(y_pred_T, dim=0)
        mean_gt_T = torch.mean(y_gt_T, dim=0)

        # Compute Pearson numerator and denominator
        num = torch.sum((y_pred_T - mean_pred_T) * (y_gt_T - mean_gt_T), dim=0)
        denom = torch.sqrt(
            torch.sum((y_pred_T - mean_pred_T) ** 2, dim=0) *
            torch.sum((y_gt_T - mean_gt_T) ** 2, dim=0)
        )

        # Compute Pearson correlation per channel
        rho_per_channel = num / (denom + 1e-8)  # Avoid division by zero

        # Identify channels where both y_pred_T and y_gt_T are all zeros
        zero_mask = (y_pred_T.abs().sum(dim=0) == 0) & (y_gt_T.abs().sum(dim=0) == 0)

        # Set rho = 1 for those channels
        rho_per_channel = torch.where(zero_mask, torch.ones_like(rho_per_channel), rho_per_channel)

        rho_vals.append(rho_per_channel)  # Shape (8,)

    ret = torch.stack(rho_vals)  # Shape (num_windows, 8)
    # print(ret)

    # return torch.mean(torch.median(ret, dim=0).values)  
    # return torch.mean(ret, dim=0)
    return torch.mean(torch.mean(ret, dim=1))  


def cosine_similarity(y_pred, y_gt, T):
    """
    Compute Cosine Similarity per channel over cropped segments of length `T`.

    Returns:
        torch.Tensor: Shape (num_windows, 8), Cosine similarity for each channel per window.
    """
    N, C = y_pred.shape
    num_windows = N // T
    cosine_vals = []

    for i in range(num_windows):
        y_pred_T = y_pred[i * T:(i + 1) * T, :]
        y_gt_T = y_gt[i * T:(i + 1) * T, :]

        # Compute cosine similarity per channel
        cos_sim = F.cosine_similarity(y_pred_T.T, y_gt_T.T, dim=1)  # Shape (8,)
        cosine_vals.append(cos_sim)

    ret = torch.stack(cosine_vals)  # Shape (num_windows, 8)
    return torch.mean(torch.mean(ret, dim=1))  


# Define the metric functions dictionary
metric_funcs_dict = {
    "mse": mse,
    "rmse": rmse,
    "mape": smape,
    "cosine_sim": cosine_similarity,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "rho": rho,
}



def get_metric_function(metric):
    return metric_funcs_dict[metric]
