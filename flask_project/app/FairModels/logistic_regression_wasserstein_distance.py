from absl import logging
import numpy as np
import scipy.sparse as sparse

from wasserstein_fairness import basic_costs
from wassterstein_fairness import optimal_transport

def gradient_smoothed_logistic(dataframes_all_data, dataframes_protected,
    theta, lambda_, beta, alpha, distance='wassestein-2', delta=0.1, baryscore=None):
    """Calculate the parameter gradient for Wasserstein fair logistic regression.
    
    This function uses the Sinkhorn algorithm to compute a smoothed coupling between predictions for separate datasets.
    
    The regression parameters "theta" may have one more entry than there are features in the input data,
    in which case an intercept term will be added to that data prior to calculating the gradient.
    
    Args:
        dataframes_all_data: a 2-tuple whose members are [0] feature data inputs to the regression as a pandas dataframe
            and [1] [0, 1] target outputs of the regression as a pandas dataframe
        dataframes_protected: a collection of feature data inputs (like the first element of 'dataframes_all_data') whose rows
            are limited to only those entries concerning members of protected categories, etc.
        theta: a vector of regression parameters of length N or N+1
        lambda_: A regulariser for computing the Wasserstein coupling.
        beta: Penalisation weight for the Wasserstein fairness loss.
        alpha: Tradeoff penalty weight between regression loss (/gradient) and the wasserstein loss(/gradient):
            'loss = alpha * regression_loss + (1 - alpha) * beta * wasserstein_loss'
        distance: Selects the distribution distance to use for computing fairness gradients and costs. Valid values are 'wasserstein-1' and 'wasserstein-2'.
        delta: delta parameter for keeping the 'wasserstein-1' difference differentiable. (The absolute value function is replaced with the pseudo-Huber loss.)
            (The actual Wasserstein cost still uses the absolute value)
        baryscore: (d, ) array that distributes as the barycenter of distributions.
        
        Returns:
            A 3-tuple with the following members:
            [0]: Wasserstein-fair logistic regression gradient
            [1]: Regression objective cost
            [2]: Wasserstein objective cost, unscaled (i.e not multiplied by beta)
        """
        def get_coupling(outputs_all_data, op):
            "Compute smoothed Wasserstein coupling"
            cost_matrix = (
                (np.reshape(outputs_all_data**2, (-1, 1)) + 
                np.reshape(op**2, (1, -1))) -
                2.0 * np.outer(outputs_all_data, op)
            )
            p = np.ones_like(outputs_all_data) / len(outputs_all_data)
            q = np.ones_like(op) / len(op)
            _, _, coupling = optimal_transport.sinkhorn(cost_matrix, lambda_, p, q)

            return coupling
        
        return _gradient_function_core(
            dataframes_all_data, dataframes_protected, theta, get_coupling,
            beta, alpha, distance, delta, baryscore
        )
        
    def gradient_line_logistic(dataframes_all_data, dataframes_protected,
        theta, beta, alpha,
        distance='wasserstein-2', delta=0.1, baryscore=None):
        """Calculate parameter gradient for Wasserstein-fair logistic regression.
        
        This function uses the hard Wasserstein coupling between predictions for separate datasets.
        
        The regression parameters "theta" may have one more entry than there are features in the input data,
        in which case an intercept term will be added to that data prior to calculating the gradient.
        
        Args:
            dataframes_all_data: a 2-tuple whose members are [0] feature data inputs to the regression as a pandas dataframe
                and [1] [0, 1] target outputs of the regression as a pandas dataframe.
            dataframes_protected: a collection of feature data inputs (like the first element of 'dataframes_all_data') whose rows"""



def _gradient_function_core(dataframes_all_data, dataframes_protected, theta, get_coupling,
                            beta, alpha, distance, delta, baryscore):
    x_all_data, y_all_data = np.array(dataframes_all_data[0]), np.array(dataframes_all_data[1])
    x_protected = [np.array(dp) for dp in dataframes_protected]

    # Run model on all and on protected inputs
    outputs_all_data = basic_costs.predict_prob(x_all_data, theta)
    outputs_protected = [
        basic_costs.predict_prob(xp, theta) for xp in x_protected
    ]

    # Compute the gradient and cost of the logistic regression for all data
    grad_logistic = basic_costs.regression_loss_gradient(
        x_all_data, y_all_data, theta
    )
    cost_logistic = basic_costs.regression_loss(
        outputs_all_data, y_all_data)

        # Compute the gradient and cost of the wasserstein loss for each protected group. Wasserstein comparisons are between 
        # the predictions for the protected group and the predictions for all data
        grad_wasserstein = np.zeros_like(theta)
        cost_wasserstein = 0.0

        if beta > 0.0:
            for xp, op in zip(x_protected, outputs_protected):

                # Compute fairness costs and gradients
                if distance == "wasserstein-2":
                    assert baryscore is None

                    # Compute coupling matrix. This may be sparse.
                    coupling = get_coupling(outputs_all_data, op)

                    # Compute the Wasserstein-2 cost. 
                    cost_wasserstein += np.array(
                        coupling.sum(axis=1)).ravel().dot(outputs_all_data**2)
                    cost_wasserstein += coupling.dot(op**2).sum()
                    cost_wasserstein -= 2.0 * coupling.dot(op).dot(outputs_all_data)

                    # Compute the Wasserstein-2 gradient
                    grad_wasserstein += basic_costs.wasserstein_two_loss_gradient(
                        x_all_data, xp, coupling, theta
                    )

                elif distance == 'wasserstein-1':
                    if baryscore is not None:
                        od = baryscore
                    else:
                        od = outputs_all_data
                    
                    # Compute the coupling matrix. This may be sparse.
                    coupling = get_coupling(od, op)

                    if sparse.issparse(coupling):
                        cost_wasserstein += abs(
                            sparse.spdiags(od, 0, len(od), len(od)).dot(coupling) - 
                            coupling.dot(sparse.spdiags(op, 0, len(op), len(op)))).sum()
                    else:
                        cost_wasserstein += abs(
                            od.reshape((-1, 1)) * coupling -
                            coupling * od.reshape((1,-1))).sum()
                    
                    # Compute the Wasserstein-1 gradient
                    if baryscore is not None:
                        grad_wasserstein += basic_costs.wass_barycenter_loss_gradient(
                            baryscore, xp, coupling, theta, delta
                        )
                    else:
                        grad_wasserstein += basic_costs.wasserstein_one_loss_gradient(
                            x_all_data, xp, coupling, theta, delta
                        )

        # All done. Totalise and return
        logging.debug('Logistic cost: %f', cost_logistic)
        logging.debug('Wasserstein cost: %f', cost_wasserstein)
        return (alpha * grad_logistic + (1.0 - alpha) * beta * grad_wasserstein,
                cost_logistic, cost_wasserstein)
                