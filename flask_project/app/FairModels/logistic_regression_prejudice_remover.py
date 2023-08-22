import numpy as np
import tensorflow as tf
from .preprocessing import preprocess_data
import scipy



# def objective_function(D, w_s, lambda_reg, prob_y_given_s, prob_y):
#     "Compute the objective function to minimize"
#     # First term: -log likelihood
#     log_likelihood = sum(np.log(logistic_regression_model(y_i, x_i, s_i, w_s[s_i])) for y_i, x_i, s_i in D)

#     # L2 regularizer
#     l2_regularizer = 0.5 * lambda_reg * sum(np.linalg.norm(w_s[s])**2 for s in w_s.keys())

#     return -log_likelihood + RPR(D, w_s, prob_y_given_s, prob_y) + l2_regularizer


# TensorFlow Model Structure

class LogisticRegressionWithRPR(tf.Module):
    def __init__(self, feature_dim, sensitive_classes):
        "Initialize the model parameters"
        # Initialize weights for each sensitive class
        self.weights = {}
        for s in sensitive_classes:
            self.weights[s] = tf.Variable(tf.zeros([feature_dim]), name= f'w_{s}')
        
    def predict(self, x, s):
        "Compute the model's prediction for a given input x and sensitive class s"
        return tf.sigmoid(tf.tensordot(x, self.weights[s], axes=1))





def logistic_regression_prejudice_remover(df, target_variable, sensitive_attribute, learning_rate, lambda_fairness, num_epochs, batch_size):

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def logistic_regression_model(y, x, s, w_s):
        return y * sigmoid(np.dot(x, w_s)) + (1-y) * (1 - sigmoid(np.dot(x, w_s)))

    def compute_sample_mean(D, S):
        "Compute sample mean vector x_bar_s for each sensitive feature s"
        print("Computing sample mean vector ...")
        x_bar = {}
        for s in S:
            x_values = [x for _, x, s_ in D if s_ == s]
            x_bar[s] = tf.reduce_mean(x_values, axis=0)
        return x_bar

    def compute_probabilities(D, x_bar, w_s, S):
        "Compute Pr[y_hat|s] and Pr[y_hat]"
        print("Computing probabilities ...")
        # Pr[y_hat|s] = M[y|x_bar_s, s; Theta]
        prob_y_given_s = {s: tf.sigmoid(tf.tensordot(x_bar[s], w_s[s], axes=1)) for s in S}

        # Pr[y_hat] = sum_s Pr[s_hat] M[y|x_bar_s, s; Theta]
        prob_s_hat = {s: len([_ for _, _, s_ in D if s_ == s]) / len(D) for s in S}
        prob_y = sum(prob_s_hat[s] * prob_y_given_s[s] for s in S)

        return prob_y_given_s, prob_y

        
    def RPR(D, w_s, prob_y_given_s, prob_y):
        "Compute the Prejudice Remover Regularizer"
        total = 0
        for y_i, x_i, s_i in D:
            for y in [0, 1]:
                M_value = logistic_regression_model(y, x_i, s_i, w_s[s_i])
                total += M_value * np.log(prob_y_given_s[s_i] / prob_y)
        return total

    def objective_function(model, D, lambda_reg, S):
        print("Computing objective function ...")
        log_likelihood = sum(tf.math.log(model.predict(x_i, s_i)) for y_i, x_i, s_i in D)

        # Compute RPR
        x_bar = compute_sample_mean(D, S)
        prob_y_given_s, prob_y = compute_probabilities(D, x_bar, model.weights, S)
        print("Computing RPR ...")
        rpr_value = sum(model.predict(x_i, s_i) * tf.math.log(prob_y_given_s[s_i] / prob_y) for _, x_i, s_i in D)

        # Compute L2 regularization
        l2_regularizer = 0.5 * lambda_reg * sum(tf.norm(model.weights[s])**2 for s in S)

        print("Objective function computed.")

        return -log_likelihood + rpr_value + l2_regularizer

    X_train, y_train, A_train, X_test, y_test, A_test = preprocess_data(df, target_variable, sensitive_attribute)
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
    feature_dim = X_train.shape[1]
    print("Feature dimension: ", feature_dim)
    sensitive_classes = list(range(A_train.shape[1]))
    print("Sensitive classes: ", sensitive_classes)
    model = LogisticRegressionWithRPR(feature_dim, sensitive_classes)

    # Convert one hot encoded A_train to integer_labels
    D_train = list(zip(y_train, X_train, [np.argmax(a) for a in A_train]))
    
    # Define optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    print("Training Prejudice Remover model ...")
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            loss = objective_function(model, D_train, lambda_fairness, sensitive_classes)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss {loss}')