from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Computes accuracy, precision, recall, and F1 score.
    Automatically selects 'binary' or 'weighted' averaging.

    Returns:
        dict with keys: accuracy, precision, recall, f1
    """
    unique_labels = set(y_true)
    if len(unique_labels) == 2:
        average = 'binary'
    else:
        average = 'weighted'

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics