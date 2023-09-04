import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import (
    count,
    selection_rate,
    equalized_odds_difference,
    demographic_parity_difference,
    false_positive_rate,
    false_negative_rate,
    true_positive_rate,
    true_negative_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds, DemographicParity
from sklearn.model_selection import train_test_split
import warnings
from .preprocessing import preprocess_data
import scipy
import tensorflow as tf
import uuid
from app.main.models import ModelResults, FairnessMetrics
from sklearn.linear_model import LogisticRegression
from app import db


def logistic_regression_demographic_parity_reductions(df, target_variable, sensitive_attribute, learning_rate, epsilon, num_epochs, batch_size):

    def compute_error_metric(metric_value, sample_size):
        """Compute standard error of a given metric based on the assumption of normal distribution
        
        Parameters:
            metric_value (float): Value of the metric
            sample_size (int): Sample size used to compute the metric
        
        Returns:
            float: Standard error of the metric
        """
        metric_value = metric_value / sample_size
        return (
            1.96
            * np.sqrt(metric_value * (1 - metric_value)
            / np.sqrt(sample_size))
        )

    def false_positive_error(y_true, y_pred):
        "Compute the standard error for the false positive rate estimate"
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return compute_error_metric(fp, tn + fp)

    def true_positive_error(y_true, y_pred):
        "Compute the standard error for the true positive rate estimate"
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return compute_error_metric(tp, tp + fn)

    def true_negative_error(y_true, y_pred):
        "Compute the standard error for the true negative rate estimate"
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return compute_error_metric(tn, tn + fp)

    def false_negative_error(y_true, y_pred):
        "Compute the standard error for the false negative rate estimate"
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return compute_error_metric(fn, fn + tp)

    def balanced_accuracy_error(y_true, y_pred):
        "Compute the standard error for the balanced accuracy estimate"
        fpr_error, fnr_error = false_positive_error(y_true, y_pred), false_negative_error(y_true, y_pred)
        return np.sqrt(fnr_error ** 2 + fpr_error ** 2) / 2

    fairness_metrics = {
    "count": count,
    "balanced_accuracy": balanced_accuracy_score,
    "balanced_acc_error": balanced_accuracy_error,
    "selection_rate": selection_rate,
    "false_positive_rate": false_positive_rate,
    "false_positive_error": false_positive_error,
    "false_negative_rate": false_negative_rate,
    "false_negative_error": false_negative_error,
    "true_positive_rate": true_positive_rate,
    "true_positive_error": true_positive_error,
    "true_negative_rate": true_negative_rate,
    "true_negative_error": true_negative_error,
}

    warnings.simplefilter("ignore")

    rand_seed = 1234
    np.random.seed(rand_seed)

    # % matplotlib inline

    X_train, y_train, A_train, X_test, y_test, A_test, A_train_original, A_test_original = preprocess_data(df, target_variable, sensitive_attribute)
    # Convert sparse matrices to dense matrices
    X_train = X_train.toarray() if scipy.sparse.issparse(X_train) else X_train
    X_test = X_test.toarray() if scipy.sparse.issparse(X_test) else X_test
    y_train = y_train.toarray() if scipy.sparse.issparse(y_train) else y_train
    y_test = y_test.toarray() if scipy.sparse.issparse(y_test) else y_test
    A_train = A_train.toarray() if scipy.sparse.issparse(A_train) else A_train
    A_test = A_test.toarray() if scipy.sparse.issparse(A_test) else A_test

    print("A test shape", A_test.shape)
    print("X test shape", X_test.shape)
    print("y test shape", y_test.shape)
    # Try resampling the data
    def resample_training_data(X_train, Y_train, A_train):
        """Down-sample the majority class in the training dataset to produce a
        balanced dataset with a 50/50 split in the predictive labels.
        """
        negative_ids = np.where(Y_train == 0)[0]
        positive_ids = np.where(Y_train == 1)[0]
        balanced_ids = np.concatenate([positive_ids, np.random.choice(a=negative_ids, size=len(positive_ids))])

        X_train = X_train[balanced_ids]
        Y_train = Y_train[balanced_ids]
        A_train = A_train[balanced_ids]
        return X_train, Y_train, A_train

    X_train, y_train, A_train = resample_training_data(X_train, y_train, A_train)





    X_train = tf.cast(X_train, tf.float32)
    X_test = tf.cast(X_test, tf.float32)
    y_test = tf.cast(y_test, tf.float32)
    A_test = tf.cast(A_test, tf.float32)


    # lgb_params = {
    #     "objective": "binary",
    #     "metric": "auc",
    #     "learning_rate": learning_rate,
    #     "num_leaves": 10,
    #     "max_depth": 3,
    #     "random_state": rand_seed,
    #     "n_jobs": 1,
    # }

    estimator = Pipeline(
        steps=[
            ("preprocessing", StandardScaler()),
            ("classifier", LogisticRegression()),
        ]
    )

    estimator.fit(X_train, y_train)

    def get_expgrad_models_per_epsilon(
        estimator, epsilon, X_train, y_train, A_train
    ):
        exp_grad_est = ExponentiatedGradient(
            estimator=estimator,
            sample_weight_name="classifier__sample_weight",
            constraints=DemographicParity(difference_bound=epsilon)
        )

        exp_grad_est.fit(X_train, y_train, sensitive_features=A_train)
        predictors = exp_grad_est.predictors_
        return predictors

    epsilons = [0.01, 0.002, 0.003, 0.004, 0.005, 0.6, 0.007, 0.008, 0.009]
    all_models = {}
    for eps in epsilons:
        all_models[eps] = get_expgrad_models_per_epsilon(
            estimator=estimator,
            epsilon=eps,
            X_train=X_train,
            y_train=y_train,
            A_train=A_train,
        )

    for epsilon, models in all_models.items():
        print(f"For epsilon = {epsilon}, ExponentiatedGradient learned {len(models)} inner models")


    def is_pareto_efficient(points):
        """
        Filter a NumPy Matrix to remove rows that are strictly dominated by another
        row in the matrix. Strictly dominated means all the row values are greater than
        the values of another row.
        """
        n, m = points.shape
        is_efficient = np.ones(n, dtype=bool)
        for i, c in enumerate(points):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(points[is_efficient] < c, axis=1)
                is_efficient[i] = True
        return is_efficient


    def filter_dominated_rows(points):
        """Remove rows from a Dataframe that are monotonically dominated by another row.
        """
        pareto_mask = is_pareto_efficient(points.to_numpy())
        return pareto_mask, points.loc[pareto_mask, :]

    def aggregate_predictor_performances(
        predictors, metric, X_test, Y_test, A_test=None
    ):
        """Compute the specified metric for all classifiers in predictors.
        If no sensitive features are present, the metric is computed without disaggregation..
        """
        all_predictions = [predictor.predict(X_test) for predictor in predictors]
        if A_test is not None:
            return [
                metric(Y_test, Y_sweep, sensitive_features=A_test)
                for Y_sweep in all_predictions
            ]
        else:
            return [metric(Y_test, Y_sweep) for Y_sweep in all_predictions]

    def model_performance_sweep(models_dict, X_test, y_test, A_test):
        performances = []
        for eps, models in models_dict.items():
            dem_parity_difference = aggregate_predictor_performances(
                models, demographic_parity_difference, X_test, y_test, A_test
            )
            bal_acc_score = aggregate_predictor_performances(
                models, balanced_accuracy_score, X_test, y_test
            )
            for (i, score) in enumerate(dem_parity_difference):
                performances.append((eps, i, score, (1 - bal_acc_score[i])))
        performances_df = pd.DataFrame.from_records(
            performances,
            columns=["epsilon", "index", "demographic_parity", "balanced_error"],
        )
        return performances_df

    performance_df = model_performance_sweep(all_models, X_test, y_test, A_test)

    print("Performance Dataframe", performance_df)

    performance_subset = performance_df.loc[
        :, ["demographic_parity", "balanced_error"]
    ]

    mask, pareto_subset = filter_dominated_rows(performance_subset)

    performance_df_masked = performance_df.loc[mask, :]

    print("Masked Performance DataFrame:", performance_df_masked)

    loss_values = [] # This should be populated during the training process
    accuracy_values = [] # This should be populated during the training process
    model_accuracy = 0.9 # Replace with actual accuracy calculation on test set
    auc_score = 0.95 # Replace with actual AUC calculation
    fairness_score = 0.85 # Replace with actual fairness score calculation
    
    def store_performance_metrics(epsilon, index, demographic_parity, balanced_error, model_id):
        
        model_results = ModelResults(id=model_id,
                                     model_class='logistic_regression',
                                     fairness_notion='demographic_parity',
                                     learning_rate=learning_rate,
                                     lambda_fairness=epsilon,
                                     batch_size=batch_size,
                                     num_epochs=num_epochs,
                                     loss_values=loss_values,
                                     accuracy_values=accuracy_values,
                                     model_accuracy=balanced_error,
                                     auc_score=auc_score,
                                     fairness_score=demographic_parity)
        db.session.add(model_results)
        db.session.commit()
    
    for _, row in performance_df_masked.iterrows():
        model_id = str(uuid.uuid4())
        store_performance_metrics(row["epsilon"], row["index"], row["demographic_parity"], row["balanced_error"], model_id)
        # Do predictions on the test set
        y_pred_pareto_optimal = all_models[row["epsilon"]][row["index"]].predict(X_test)
        # bal_acc_pareto_optimal = balanced_accuracy_score(y_test, y_pred_pareto_optimal)
        # dem_parity_pareto_optimal = demographic_parity_difference(y_test, y_pred_pareto_optimal, sensitive_features=A_test)
        metric_frame_pareto_optimal = MetricFrame(
            metrics=fairness_metrics,
            y_true=y_test,
            y_pred=y_pred_pareto_optimal,
            sensitive_features=A_test_original
        )

        metrics_by_group = metric_frame_pareto_optimal.by_group



        metrics_data = metrics_by_group.to_dict()

        def convert_tuple_keys_to_str(data):
            """
            Recursively convert tuple keys in a nested dictionary to string representations.
            """
            if isinstance(data, dict):
                new_data = {}
                for key, value in data.items():
                    new_key = str(key) if isinstance(key, tuple) else key
                    new_data[new_key] = convert_tuple_keys_to_str(value)
                return new_data
            else:
                return data

        metrics_data = convert_tuple_keys_to_str(metrics_data)

        fairness_metric = FairnessMetrics(
            id=str(uuid.uuid4()),
            model_results_id=model_id,
            fairness_notion='demographic_parity',
            # group="test",
            epoch=row["epsilon"],
            metrics=metrics_data

        )

        db.session.add(fairness_metric)

    db.session.commit()
      