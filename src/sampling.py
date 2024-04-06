from typing import Callable, Optional

import numpy as np


def chisq(y: np.ndarray, y_hat: np.ndarray, sigma_y: np.ndarray) -> float:
    """Returns the Chisq."""
    return sum(((y - y_hat) / sigma_y) ** 2)


def loglikelihood(chisq: float) -> float:
    """Return the loglikelihood."""
    return -0.5 * chisq


def metropolis_hastings(
    iterations: int,
    f: Callable[[np.ndarray], np.ndarray],
    theta_0: tuple[float, ...],
    step_sizes: tuple[float, ...],
    y: np.ndarray,
    sigma_y: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic Metropolis-Hastings algorithm.

    Parameters
    ----------
    iterations : int
        Number of iterations to run the algorithm.
    f : Callable[[np.ndarray], np.ndarray]
        Model evaluation function which is called `f(theta)` where `theta` is tuple of
        model parameters.
    theta_0 : tuple[float, ...]
        Initial guess for the model parameters.
    step_sizes : tuple[float, ...]
        Step sizes for the model parameters.
    y : np.ndarray
        Observed data.
    sigma_y : Optional[np.ndarray], optional
        Uncertainty in the observed data, by default None
    cov : Optional[np.ndarray], optional
        Covariance matrix for the model parameters, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of arrays with the first array being the trace of the model parameters and
        the second array being the trace of the chi-squared values.
    """
    n_theta = len(theta_0)
    theta_trace = np.zeros((n_theta, iterations - 1))
    chisq_trace = np.zeros(iterations - 1)

    theta_current = np.asarray(theta_0)

    if sigma_y is None:
        sigma_y = np.ones_like(y)

    if cov is None:
        cov = np.eye(n_theta)

    L = np.linalg.cholesky(cov)
    accepted_proposals = 0
    for i in range(iterations):
        # For the first iteration we just want to evaluate the model at the initial guess
        if i == 0:
            y_hat = f(theta_0)
            chisq_current = chisq(y, y_hat, sigma_y)
            continue

        # update the model parameters
        theta_new = theta_current + (np.random.randn(n_theta) * step_sizes) @ L

        # evaluate the model at the new parameters
        y_hat = f(theta_new)

        # compute the log likelihood
        chisq_new = chisq(y, y_hat, sigma_y)
        chisq_diff = max(chisq_new - chisq_current, 0)
        log_likelihood_ratio = loglikelihood(chisq_diff)

        # Metropolis-Hastings acceptance criterion
        if log_likelihood_ratio > np.log(np.random.rand()):
            # if accepted, update the current parameters and the current chi-squared value
            theta_current = theta_new
            chisq_current = chisq_new
            accepted_proposals += 1
            print(f"Accepted! rate: {accepted_proposals / (i)}", "iteration: ", i)

        # store the current parameters and chi-squared value
        theta_trace[:, i - 1] = theta_current
        chisq_trace[i - 1] = chisq_current

    print(f"Final acceptance rate: {accepted_proposals / (iterations-1)}")

    return theta_trace, chisq_trace


def hamiltonian(
    iterations: int,
    f: Callable[[np.ndarray], np.ndarray],
    gradient: Callable[[np.ndarray], np.ndarray],
    theta_0: tuple[float, ...],
    y: np.ndarray,
    path_length: float,
    step_size: float,
    sigma_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_theta = len(theta_0)
    theta_trace = np.zeros((n_theta, iterations - 1))
    chisq_trace = np.zeros(iterations - 1)

    theta_current = np.array(theta_0)
    accepted_proposals = 0

    for i in range(iterations):
        p0 = np.random.normal(size=n_theta)
        if i == 0:
            K_current = 0.5 * np.dot(p0, p0)
            chisq_current = chisq(y, f(theta_current), sigma_y)
            U_current = -loglikelihood(chisq_current)
            H_current = U_current + K_current
            continue

        q_new, p_new = leapfrog(
            gradient=gradient,
            position=theta_current,
            momentum=p0,
            step_size=step_size,
            path_length=path_length,
        )
        K_new = 0.5 * np.dot(p_new, p_new)
        chisq_new = chisq(y, f(q_new), sigma_y)
        U_new = -loglikelihood(chisq_new)
        H_new = U_new + K_new

        if (H_current - H_new) >= np.log(np.random.rand()):
            accepted_proposals += 1
            theta_current = q_new
            chisq_current = chisq_new
            H_current = H_new
            print(f"Accepted! rate: {accepted_proposals / (i)}", "iteration: ", i)
        theta_trace[:, i - 1] = theta_current
        chisq_trace[i - 1] = chisq_current

    print(f"Final acceptance rate: {accepted_proposals / (iterations-1)}")

    return theta_trace, chisq_trace


def leapfrog(
    gradient: Callable[[np.ndarray], np.ndarray],
    position: np.ndarray,
    momentum: np.ndarray,
    step_size: float,
    path_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Leapfrog integrator for Hamiltonian Monte Carlo.

    Parameters
    ----------
    gradient : Callable[[np.ndarray], np.ndarray]
        Gradient of the potential energy function.
    position : np.ndarray
        Position vector.
    momentum : np.ndarray
        Momentum vector.
    step_size : float
        Step size for the integrator.
    path_length : float
        Length of the integration path.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of the new position and momentum vectors.
    """
    q = position.copy()
    p = momentum.copy()

    p += gradient(q) * (step_size / 2)

    for _ in range(path_length):
        q += p * step_size
        p += gradient(q) * step_size

    q += p * step_size
    p += gradient(q) * (step_size / 2)

    return q, -p
