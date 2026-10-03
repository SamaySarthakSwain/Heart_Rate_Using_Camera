import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mae(y_true, y_pred):
    """Mean Absolute Error (BPM)"""
    return mean_absolute_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    """Root Mean Squared Error (BPM)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_pearson(y_true, y_pred):
    """Pearson Correlation Coefficient"""
    if len(y_true) < 2:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return r

def calculate_spearman(y_true, y_pred):
    """Spearman Rank Correlation Coefficient"""
    if len(y_true) < 2:
        return 0.0
    rho, _ = spearmanr(y_true, y_pred)
    return rho

def calculate_r2(y_true, y_pred):
    """R-squared (Coefficient of Determination)"""
    return r2_score(y_true, y_pred)

def bland_altman_stats(y_true, y_pred):
    """
    Returns the bias (mean difference) and limits of agreement (95%).
    """
    diff = np.array(y_pred) - np.array(y_true)
    bias = np.mean(diff)
    sd = np.std(diff)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
    return bias, sd, loa_lower, loa_upper

def calculate_failure_rate(y_true, y_pred, threshold=5.0):
    """
    Percentage of predictions that are off by more than `threshold` BPM.
    """
    diff = np.abs(np.array(y_pred) - np.array(y_true))
    failures = np.sum(diff > threshold)
    return (failures / len(y_true)) * 100.0

def evaluate_all(y_true, y_pred, threshold=5.0):
    """
    Returns a dictionary of all standardized evaluation metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    bias, sd, loa_lower, loa_upper = bland_altman_stats(y_true, y_pred)
    
    return {
        "MAE": calculate_mae(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "Pearson": calculate_pearson(y_true, y_pred),
        "Spearman": calculate_spearman(y_true, y_pred),
        "R2": calculate_r2(y_true, y_pred),
        "Bias": bias,
        "SD": sd,
        "LoA_Lower": loa_lower,
        "LoA_Upper": loa_upper,
        "Failure_Rate_5BPM": calculate_failure_rate(y_true, y_pred, threshold)
    }

if __name__ == "__main__":
    # Test with dummy data
    y_true = [70, 72, 75, 80, 85, 78, 73]
    y_pred = [71, 71, 76, 82, 83, 80, 72]
    
    metrics = evaluate_all(y_true, y_pred)
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")
