import numpy as np
import sklearn.metrics as metrics
import torch

from utils.data_utils import TaskType


def check_softmax(logits):
    if np.any((logits < 0) | (logits > 1)) or (not np.allclose(logits.sum(axis=-1), 1, atol=1e-5)):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # stabilize by subtracting max
        return exps / np.sum(exps, axis=1, keepdims=True)
    else:
        return logits

def get_metrics(preds, gts, y_info, task_type):
    if not isinstance(gts, np.ndarray):
        if gts.dtype == torch.bfloat16:
            gts = gts.to(torch.float32)
        gts = gts.cpu().numpy()

    if not isinstance(preds, np.ndarray):
        if preds.dtype == torch.bfloat16:
            preds = preds.to(torch.float32)
        preds = preds.cpu().numpy()

    if task_type == TaskType.REGRESSION:
        rmse = metrics.mean_squared_error(gts, preds) ** 0.5
        if y_info["policy"] == "mean_std":
            rmse *= y_info["std"]
        return rmse, "RMSE"

    elif task_type == TaskType.BINCLASS:
        # if not softmax, convert to probabilities
        preds = check_softmax(preds)
        accuracy = metrics.accuracy_score(gts, preds.argmax(axis=-1))
        return accuracy, "ACC"

    elif task_type == TaskType.MULTICLASS:
        # if not softmax, convert to probabilities
        preds = check_softmax(preds)
        accuracy = metrics.accuracy_score(gts, preds.argmax(axis=-1))
        return accuracy, "ACC"

    else:
        raise ValueError("Unknown tabular task type")
