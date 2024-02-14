import numpy as np

from dirbe_data import DirbeData
from extended_zodipy import FittingModel
import matplotlib.pyplot as plt

np.random.seed(42059)

def main() -> None:
    data = DirbeData.new(range(1, 4), slice(0, 2000))
    model = FittingModel(data=data, model="DIRBE", parallel=False)
    print("n_0 og:", model.get_param_value("n_0"))
    print("gamma og:", model.get_param_value("gamma"))

    thetas, chisqs = metropolis_hastings(model, 200)
    print("n_0:", thetas[0, -1])
    print("gamma:", thetas[1, -1])

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(thetas[0], label="n_0")
    ax[1].plot(thetas[1], label="gamma")
    ax[2].plot(chisqs, label="chisq")

    ax[0].set_ylabel("n_0")
    ax[1].set_ylabel("gamma")
    ax[2].set_ylabel("chisq")

    fig.supxlabel("Iteration")
    plt.tight_layout()
    plt.savefig("mcmc.pdf", dpi=250)
    plt.show()


def metropolis_hastings(
    model: FittingModel, iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    theta_trace = np.zeros((2, iterations - 1))
    chisq_trace = np.zeros(iterations - 1)

    s_zodi_true = model.evaluate()
    s_zodi_observed = s_zodi_true + np.random.randn(s_zodi_true.size)
    plt.plot(s_zodi_observed)
    plt.plot(s_zodi_true)
    plt.show()

    n_0 = model.get_param_value("n_0")
    gamma = model.get_param_value("gamma")

    step_sizes = np.array([0.05 * n_0, 0.05 * gamma])


    chisq_current = 1e30
    theta_current = np.array([n_0, gamma])
    theta_current += theta_current * np.random.randn(2)

    model.update({"n_0": theta_current[0], "gamma": theta_current[1]})

    for i in range(iterations):
        theta_new = theta_current + step_sizes * np.random.randn(2)
        model.update({"n_0": theta_new[0], "gamma": theta_new[1]})
        s_zodi = model.evaluate()
        chisq_new = sum((s_zodi_observed - s_zodi) ** 2)

        if i == 0:
            chisq_current = chisq_new
            continue

        chisq_diff = max(chisq_new - chisq_current, 0)
        ln_P = -0.5 * chisq_diff
        if ln_P > np.log(np.random.rand()):
            theta_current = theta_new
            chisq_current = chisq_new

        theta_trace[:, i - 1] = theta_current
        chisq_trace[i - 1] = chisq_current

    return theta_trace, chisq_trace


def metropolis_hastings_old(
    model: FittingModel, iterations: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_zodi_true = model.evaluate()
    s_zodi_true += np.random.randn(s_zodi_true.size)

    n_0_trace = np.zeros(iterations)
    gamma_trace = np.zeros_like(n_0_trace)
    chisq_trace = np.zeros_like(n_0_trace)

    n_0_current = model.get_param_value("n_0")
    gamma_current = model.get_param_value("gamma")
    n_0_step = 0.1 * n_0_current
    gamma_step = 0.1 * gamma_current

    n_0_current *= np.random.rand()
    gamma_current *= np.random.rand()

    n_0_trace[0] = n_0_current
    gamma_trace[0] = gamma_current

    for i in range(iterations):
        n_0_new = n_0_current + n_0_step * np.random.uniform()
        gamma_new = gamma_current + gamma_step * np.random.uniform()
        theta = {
            "n_0": n_0_new,
            "gamma": gamma_new,
        }
        model.update(theta)
        s_zodi = model.evaluate()

        if i == 0:
            chisq_current = ((s_zodi_true - s_zodi) ** 2).sum()
            likelihood_current = -0.5 * chisq_current
            chisq_trace[i] = chisq_current
            continue

        chisq_new = ((s_zodi_true - s_zodi) ** 2).sum()
        chisq_diff = chisq_new - chisq_current

        likelihood_new = -0.5 * chisq_diff

        # Compute acceptance probability
        alpha = max(0, likelihood_new / likelihood_current)

        accepted = likelihood_new > np.log(np.random.uniform())
        # Accept or reject proposal
        if np.random.rand() < alpha:
            likelihood_current = likelihood_new
            chisq_current = chisq_new
            n_0_current = n_0_new
            gamma_current = gamma_new

        n_0_trace[i] = n_0_current
        gamma_trace[i] = gamma_current
        chisq_trace[i] = chisq_current

    return n_0_trace, gamma_trace, chisq_trace


if __name__ == "__main__":
    main()
