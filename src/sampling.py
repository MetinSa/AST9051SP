from typing import Callable

import numpy as np


def metropolis_hastings(
    f: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    theta_0: tuple[float, ...],
    step_sizes: tuple[float, ...],
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic Metropolis-Hastings algorithm.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        Model evaluation function which is called `f(theta)` where `theta` is tuple of
        model parameters.
    y : np.ndarray
        Observed data.
    theta_0 : tuple[float, ...]
        Initial guess for the model parameters.
    step_sizes : tuple[float, ...]
        Step sizes for the model parameters.
    iterations : int
        Number of iterations to run the algorithm.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of arrays with the first array being the trace of the model parameters and
        the second array being the trace of the chi-squared values.
    """
    n_theta = len(theta_0)
    theta_trace = np.zeros((n_theta, iterations - 1))
    chisq_trace = np.zeros(iterations - 1)

    theta_current = theta_0

    accepted_proposals = 0
    for i in range(iterations):
        # For the first iteration we just want to evaluate the model at the initial guess
        if i == 0:
            y_hat = f(theta_0)
            chisq_current = sum((y - y_hat) ** 2)
            continue

        # update the model parameters
        theta_new = theta_current + step_sizes * np.random.randn(n_theta)

        # evaluate the model at the new parameters
        y_hat = f(theta_new)
        chisq_new = sum((y - y_hat) ** 2)
        # compute the log likelihood
        chisq_diff = max(chisq_new - chisq_current, 0)
        log_likelihood = -0.5 * chisq_diff

        # Metropolis-Hastings acceptance criterion
        if log_likelihood > np.log(np.random.rand()):
            # if accepted, update the current parameters and the current chi-squared value
            theta_current = theta_new
            chisq_current = chisq_new
            accepted_proposals += 1

        # store the current parameters and chi-squared value
        theta_trace[:, i - 1] = theta_current
        chisq_trace[i - 1] = chisq_current

    print(f"Acceptance rate: {accepted_proposals / (iterations-1)}")

    return theta_trace, chisq_trace
