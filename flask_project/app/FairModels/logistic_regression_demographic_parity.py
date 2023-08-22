import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from .preprocessing import preprocess_data
import scipy
from .fairness_metrics import calculate_predictive_rates_multigroup, calculate_fairness_metrics_multigroup, calculate_confusion_matrix_multigroup, convert_keys_to_int
from app.main.models import ModelResults, FairnessMetrics
from app import db
import uuid

def logistic_regression_demographic_parity(df, target_variable, sensitive_attribute, learning_rate, lambda_fairness, num_epochs, batch_size):

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
    if np.isnan(w.numpy()).any() or np.isnan(b.numpy()).any():
        print("NaN value encountered in initial weights")
        import pdb; pdb.set_trace()
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

        # Debug: Check if group_counts contains zero
        if tf.reduce_any(tf.equal(group_counts, 0)):
            print("Zero value encountered in group_counts")
            import pdb; pdb.set_trace()

        group_averages = group_predictions / group_counts[:, tf.newaxis]
        if tf.reduce_any(tf.math.is_nan(group_averages)):
            print("NaN value encountered in group_averages")
            import pdb; pdb.set_trace()

        max_diff = tf.reduce_max(group_averages) - tf.reduce_min(group_averages)
        if tf.reduce_any(tf.math.is_nan(max_diff)):
            print("NaN value encountered in max_diff")
            import pdb; pdb.set_trace()

        return max_diff


    # Define the custom loss function

    def loss_fn(y_true, y_pred, A_one_hot):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        A_one_hot = tf.cast(A_one_hot, tf.float32)
        epsilon = 1e-7

        log_y_pred = tf.math.log(y_pred + epsilon)
        if tf.reduce_any(tf.math.is_nan(log_y_pred)):
            print("NaN value encountered in log_y_pred")
            import pdb; pdb.set_trace()

        log_1_minus_y_pred = tf.math.log(1 - y_pred + epsilon)
        if tf.reduce_any(tf.math.is_nan(log_1_minus_y_pred)):
            print("NaN value encountered in log_1_minus_y_pred")
            import pdb; pdb.set_trace()

        log_loss = -tf.reduce_mean(y_true * log_y_pred + (1 - y_true) * log_1_minus_y_pred)
        if tf.reduce_any(tf.math.is_nan(log_loss)):
            print("NaN value encountered in log_loss")
            import pdb; pdb.set_trace()

        fairness_loss = fairness_penalty(A_one_hot, y_pred)
        if tf.reduce_any(tf.math.is_nan(fairness_loss)):
            print("NaN value encountered in fairness_loss")
            import pdb; pdb.set_trace()

        total_loss = log_loss + lambda_fairness * fairness_loss
        if tf.reduce_any(tf.math.is_nan(total_loss)):
            print("NaN value encountered in total_loss")
            import pdb; pdb.set_trace()

        return total_loss

    # Define the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # Create a tf.data.Dataset object
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train, A_train))

    # Shuffle and batch the data
    # batch_size = 32
    train_data = train_data.shuffle(buffer_size=1024).batch(batch_size)

    # Initialize lists to store the values for each iteration
    loss_values = []
    fairness_values = []
    accuracy_values = []
    model_id = str(uuid.uuid4())

    with tf.device('/GPU:0'):

    # Train the model
        for epoch in range(num_epochs):  # number of training iterations
            predictions_list = []
            for batch_x, batch_y, batch_a in train_data:
                with tf.GradientTape() as tape:
                    predictions = model(batch_x)
                    if np.isnan(predictions.numpy()).any():
                        print("NaN value encountered in predictions")
                        import pdb; pdb.set_trace()
                    predictions_list.append(predictions)
                    loss = loss_fn(batch_y, predictions, batch_a)
                    if np.isnan(loss.numpy()).any():
                        print("NaN value encountered in loss")
                        import pdb; pdb.set_trace()

                grads = tape.gradient(loss, [w, b])
                if np.isnan(grads[0].numpy()).any() or np.isnan(grads[1].numpy()).any():
                    print("NaN value encountered in gradients")
                    import pdb; pdb.set_trace()

                optimizer.apply_gradients(zip(grads, [w, b]))
                if np.isnan(w.numpy()).any() or np.isnan(b.numpy()).any():
                    print("NaN value encountered in weights")
                    import pdb; pdb.set_trace()
 
            # Calculate the fairness and accuracy for this iteration
            current_accuracy = np.mean((predictions.numpy() > 0.5) == batch_y)
            current_fairness = fairness_penalty(batch_a, predictions).numpy()

            # After the end of each epoch, concatenate the predictions for all batches
            epoch_predictions = tf.concat(predictions_list, axis=0)

            # At the end of each epoch, calculate the predictive rates and fairness metrics
            group_predictive_rates = calculate_predictive_rates_multigroup(y_train, epoch_predictions, A_train)
            # demographic_parity_difference, demographic_parity_ratio = calculate_fairness_metrics_multigroup(y_train, epoch_predictions, A_train)

            fairness_metrics_dict = calculate_fairness_metrics_multigroup(y_train, epoch_predictions, A_train)
            for group, predictive_rate in group_predictive_rates.items():
                group_metrics = fairness_metrics_dict[int(group)]

                # Create a dictionary of fairness metrics for the current group and epoch
                metrics = {
                            'predictive_rate': predictive_rate,
                            'demographic_parity_difference': group_metrics[0],
                            'demographic_parity_ratio': group_metrics[1]
                        }
                
                metrics = convert_keys_to_int(metrics)
                
                # Create a FairnessMetrics instance and add it to the session
                fairness_metrics = FairnessMetrics(
                                                    id=str(uuid.uuid4()),
                                                    model_results_id=model_id,
                                                    fairness_notion='demographic_parity',
                                                    group=group,
                                                    epoch=epoch,
                                                    metrics=metrics
                )
                db.session.add(fairness_metrics)
            db.session.commit()

            predictions_list.clear()
            
            # Append the loss, accuracy, and fairness values for this epoch
            loss_values.append(loss.numpy())
            accuracy_values.append(current_accuracy)
            fairness_values.append(current_fairness)

    # Make predictions
    predictions_test = model(X_test)

    # Calculate the accuracy of the model
    model_accuracy = np.mean((predictions_test.numpy() > 0.5) == y_test)

    print("Test Predictions NaN: ",np.isnan(predictions_test.numpy()).any())
    print("Test Values NaN",np.isnan(y_test).any())

    # Calculate the AUC score
    auc_score = roc_auc_score(y_test, predictions_test.numpy())

    # Calculate the fairness metric (demographic parity)
    fairness_score = fairness_penalty(A_test, predictions_test).numpy()
    
    # Store the results in a model database

    model_results = ModelResults(id=model_id, 
                                 model_class='logistic_regression', 
                                 fairness_notion='demographic_parity', 
                                 learning_rate=learning_rate, 
                                 lambda_fairness=lambda_fairness, 
                                 batch_size=batch_size, 
                                 num_epochs=num_epochs, 
                                 loss_values=loss_values, 
                                 accuracy_values=accuracy_values, 
                                 model_accuracy=model_accuracy, 
                                 auc_score=auc_score,
                                 fairness_score=fairness_score)
    
    db.session.add(model_results)
    db.session.commit()
    # end_time = time.time()  # End measuring execution time
    # execution_time = end_time - start_time
    # print("Execution time: {:.2f} seconds".format(execution_time))

