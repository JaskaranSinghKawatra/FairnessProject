from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd
from app import celery
import tensorflow as tf
import numpy as np
import time

@celery.task(bind=True)
def train_model(self, file_path, target_variable, sensitive_attribute):
    print("Train model task is being executed.")
    start_time = time.time()
    # Load the dataset
    df = pd.read_csv(file_path)

    # Separate the features and target
    X = df.drop(columns=[target_variable, sensitive_attribute]).values
    y = df[target_variable].values
    
    # Get unique values for the sensitive attribute and create a mapping
    unique_values = df[sensitive_attribute].unique()
    mapping_dict = {value: i for i, value in enumerate(unique_values)}
    
    # Apply the mapping to the column and convert to integers
    A = df[sensitive_attribute].map(mapping_dict).values.astype(np.int32)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.2, random_state=42)
    
    # Cast X_train as float for matrix multiplication
    X_train = tf.cast(X_train, tf.float64)

    # Initialize the weights and bias
    w = tf.Variable(tf.random.normal([X.shape[1], 1], dtype = tf.float64))
    b = tf.Variable(tf.zeros([1], dtype = tf.float64))

    # Define the logistic regression model
    def model(X):
        return tf.math.sigmoid(tf.matmul(X, w) + b)

    # Define the fairness penalty
    def fairness_penalty(A, predictions):
        group_0_mask = tf.equal(A, 0)
        group_1_mask = tf.equal(A, 1)
        predictions_0 = tf.boolean_mask(predictions, group_0_mask)
        predictions_1 = tf.boolean_mask(predictions, group_1_mask)
        return tf.abs(tf.reduce_mean(predictions_0) - tf.reduce_mean(predictions_1))

    # Define the custom loss function
    def loss_fn(y_true, y_pred, A, lambda_fairness):
        log_loss = -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        fairness_loss = fairness_penalty(A, y_pred)
        return log_loss + lambda_fairness * fairness_loss

    # Define the optimizer
    optimizer = tf.optimizers.Adam()

    # Train the model
    for _ in range(1000):  # number of training iterations
        with tf.GradientTape() as tape:
            X_train_tf = tf.cast(X_train, tf.float64)
            predictions = model(X_train_tf)
            # predictions = model(X_train)
            loss = loss_fn(y_train, predictions, A_train, lambda_fairness=1.0)  # adjust lambda_fairness as needed
        grads = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(grads, [w, b]))

    # Make predictions
    predictions_test = model(X_test)

    # Calculate the accuracy of the model
    model_accuracy = np.mean((predictions_test.numpy() > 0.5) == y_test)

    # Calculate the AUC score
    auc_score = roc_auc_score(y_test, predictions_test.numpy())

    # Calculate the fairness metric (demographic parity)
    fairness_metric = fairness_penalty(A_test, predictions_test).numpy()

    end_time = time.time()  # End measuring execution time
    execution_time = end_time - start_time
    print("Execution time: {:.2f} seconds".format(execution_time))

    return predictions_test.numpy().tolist(), model_accuracy, auc_score


# @celery.task(bind=True)
# def train_model(self, file_path, target_variable, sensitive_attribute):
#     print("Train model task is being executed.")
#     # Load the dataset
#     df = pd.read_csv(file_path)

#     # Separate the features and target
#     X = df.drop(columns=[target_variable, sensitive_attribute])
#     y = df[target_variable]

#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Initialize and train the model
#     model = LogisticRegression()
#     model.fit(X_train, y_train)

#     # Make predictions
#     predictions = model.predict(X_test)

#     # Calculate the accuracy of the model
#     model_accuracy = model.score(X_test, y_test)

#     # Calculate the AUC score
#     auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

#     return predictions.tolist(), model_accuracy, auc_score










