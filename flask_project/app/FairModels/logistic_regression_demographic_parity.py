import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from .preprocessing import preprocess_data
import scipy
from .fairness_metrics import calculate_predictive_rates_multigroup, calculate_fairness_metrics_multigroup, calculate_confusion_matrix_multigroup

def logistic_regression_demographic_parity(df, target_variable, sensitive_attribute, learning_rate, lambda_fairness):

    X_train, y_train, A_train, X_test, y_test, A_test = preprocess_data(df, target_variable, sensitive_attribute)

      
# Convert sparse matrices to dense matrices
    X_train = X_train.toarray() if scipy.sparse.issparse(X_train) else X_train
    X_test = X_test.toarray() if scipy.sparse.issparse(X_test) else X_test
    y_train = y_train.toarray() if scipy.sparse.issparse(y_train) else y_train
    y_test = y_test.toarray() if scipy.sparse.issparse(y_test) else y_test
    A_train = A_train.toarray() if scipy.sparse.issparse(A_train) else A_train
    A_test = A_test.toarray() if scipy.sparse.issparse(A_test) else A_test



    X_train = tf.cast(X_train, tf.float32)
    X_test = tf.cast(X_test, tf.float32)
    y_test = tf.cast(y_test, tf.float32)
    A_test = tf.cast(A_test, tf.float32)

    # Initialize the weights and bias
    w = tf.Variable(tf.random.normal([X_train.shape[1], 1], dtype = tf.float32))
    b = tf.Variable(tf.zeros([1], dtype = tf.float32))

    # Define the logistic regression model
    def model(X):
        X = tf.cast(X, tf.float32)
        return tf.keras.activations.sigmoid(tf.matmul(X, w) + b)

    # Define the fairness penalty
    def fairness_penalty(A_one_hot, predictions):
        A_one_hot = tf.cast(A_one_hot, tf.float32)
        predictions = tf.cast(predictions, tf.float32)
        group_predictions = tf.matmul(tf.transpose(A_one_hot), predictions)
        group_counts = tf.reduce_sum(A_one_hot, axis=0)
        group_averages = group_predictions / group_counts[:, tf.newaxis]
        # Check for NaNs
        # tf.debugging.check_numerics(group_predictions, message='Debugging: NaN found in group_predictions')
        # tf.debugging.check_numerics(group_counts, message='Debugging: NaN found in group_counts')
        # tf.debugging.check_numerics(group_averages, message='Debugging: NaN found in group_averages')
        max_diff = tf.reduce_max(group_averages) - tf.reduce_min(group_averages)
        return max_diff

        # group_0_mask = tf.equal(A, 0)
        # group_1_mask = tf.equal(A, 1)
        # predictions_0 = tf.boolean_mask(predictions, group_0_mask)
        # predictions_1 = tf.boolean_mask(predictions, group_1_mask)
        # return tf.abs(tf.reduce_mean(predictions_0) - tf.reduce_mean(predictions_1))

    # Define the custom loss function
    def loss_fn(y_true, y_pred, A_one_hot):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        A_one_hot = tf.cast(A_one_hot, tf.float32)
        # lambda_fairness = tf.cast(lambda_fairness, tf.float32)
        epsilon = 1e-7
        log_loss = -tf.reduce_mean(y_true * tf.math.log(y_pred + epsilon) + (1 - y_true) * tf.math.log(1 - y_pred + epsilon))

        # Check for NaNs
        #tf.debugging.check_numerics(log_loss, message='Debugging: NaN found in log_loss')
        fairness_loss = fairness_penalty(A_one_hot, y_pred)
        # Check for NaNs
        #tf.debugging.check_numerics(fairness_loss, message='Debugging: NaN found in fairness_loss')
        return log_loss + lambda_fairness * fairness_loss

    # Define the optimizer
    optimizer = tf.optimizers.Adam(learning_rate)

    # Create a tf.data.Dataset object
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train, A_train))

    # Shuffle and batch the data
    batch_size = 32
    train_data = train_data.shuffle(buffer_size=1024).batch(batch_size)

    # Initialize lists to store the values for each iteration
    loss_values = []
    fairness_values = []
    accuracy_values = []

    with tf.device('/GPU:0'):

    # Train the model
        for _ in range(100):  # number of training iterations
            predictions_list = []
            for batch_x, batch_y, batch_a in train_data:
                with tf.GradientTape() as tape:
                    # X_train_tf = tf.cast(X_train, tf.float64)
                    predictions = model(batch_x)
                    predictions_list.append(predictions)

                    # print(f"Min prediction: {tf.reduce_min(predictions).numpy()}, Max prediction: {tf.reduce_max(predictions).numpy()}")


                    # predictions = model(X_train)
                    loss = loss_fn(batch_y, predictions, batch_a)  # adjust lambda_fairness as needed

                    

                    # Check for NaNs in predictions and loss
                    # tf.debugging.check_numerics(predictions, message='Debugging: NaN found in predictions')
                    # tf.debugging.check_numerics(loss, message='Debugging: NaN found in loss')

            grads = tape.gradient(loss, [w, b])
            optimizer.apply_gradients(zip(grads, [w, b]))

        # Calculate the fairness and accuracy for this iteration
        current_accuracy = np.mean((predictions.numpy() > 0.5) == batch_y)
        current_fairness = fairness_penalty(batch_a, predictions).numpy()

        # After the end of each epoch, concatenate the predictions for all batches
        epoch_predictions = tf.concat(predictions_list, axis=0)

        # At the end of each epoch, calculate the predictive rates and fairness metrics
        group_predictive_rates = calculate_predictive_rates_multigroup(y_train, epoch_predictions, A_train)
        demographic_parity_difference, demographic_parity_ratio = calculate_fairness_metrics_multigroup(y_train, epoch_predictions, A_train)
        predictions_list.clear()
        
        # Append the loss, accuracy, and fairness values for this epoch
        loss_values.append(loss.numpy())
        accuracy_values.append(current_accuracy)
        fairness_values.append(current_fairness)

    # Make predictions
    predictions_test = model(X_test)

    # Calculate the accuracy of the model
    model_accuracy = np.mean((predictions_test.numpy() > 0.5) == y_test)

    # Calculate the AUC score
    auc_score = roc_auc_score(y_test, predictions_test.numpy())

    # Calculate the fairness metric (demographic parity)
    fairness_metric = fairness_penalty(A_test, predictions_test).numpy()

    # end_time = time.time()  # End measuring execution time
    # execution_time = end_time - start_time
    # print("Execution time: {:.2f} seconds".format(execution_time))

    return predictions_test.numpy().tolist(), model_accuracy, auc_score
