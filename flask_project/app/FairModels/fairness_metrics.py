from sklearn.metrics import confusion_matrix
import numpy as np

# Define a function to calculate predictive rates
def calculate_predictive_rates_multigroup(y_true, y_pred, A_one_hot):
    A = np.argmax(A_one_hot, axis = 1)
    unique_A = np.unique(A)
    predictive_rates = {}
    for a in unique_A:
        mask_A = A == a
        predictive_rate_A = np.mean(y_pred[mask_A] > 0.5)
        predictive_rates[a] = predictive_rate_A
    return predictive_rates

# Define a function to calculate fairness metrics
def calculate_fairness_metrics_multigroup(y_true, y_pred, A_one_hot):
    A = np.argmax(A_one_hot, axis = 1)
    unique_A = np.unique(A)
    fairness_metrics = {}
    predictive_rates = calculate_predictive_rates_multigroup(y_true, y_pred, A_one_hot)
    for a in unique_A:
        cm_A = confusion_matrix(y_true[A == a], y_pred[A == a] > 0.5)
        if cm_A.sum() == 0 or cm_A[1].sum() == 0:
            continue
        demographic_parity_differences = {}
        demographic_parity_ratios = {}
        for b in unique_A:
            if a != b:
                demographic_parity_differences[b] = predictive_rates[a] - predictive_rates[b]
                demographic_parity_ratios[b] = predictive_rates[a] / (predictive_rates[b] + 1e-7)
        fairness_metrics[a] = (demographic_parity_differences, demographic_parity_ratios)
    return fairness_metrics

# Define a function to calculate confusion matrix
def calculate_confusion_matrix_multigroup(y_true, y_pred, A_one_hot):
    A = np.argmax(A_one_hot, axis = 1)
    unique_A = np.unique(A)
    confusion_matrices = {}
    for a in unique_A:
        cm_A = confusion_matrix(y_true[A == a], y_pred[A == a] > 0.5)
        confusion_matrices[a] = cm_A
    return confusion_matrices