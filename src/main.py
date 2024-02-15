from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np
from dirbe_data import DirbeData

from model import ZodiModel
from sampling import metropolis_hastings

np.random.seed(42059)

FIG_DIR = Path("../figs/")
FIG_DIR.mkdir(exist_ok=True)


def plot_timestream() -> None:
    N = 2000
    data = DirbeData.new(range(1, 2), slice(0, N))
    # get the true signal y
    s_zodi = np.squeeze(data.zodi) if len(data.zodi) == 1 else np.concatenate(data.zodi)

    # add noise to the true signal
    s_zodi_obs = s_zodi + np.random.randn(s_zodi.size)

    plt.plot(s_zodi_obs, label="Observed")
    plt.plot(s_zodi, label="True")
    plt.ylabel("MJy/sr")
    plt.xlabel("Observations")
    plt.legend()
    plt.savefig(FIG_DIR / "timestream.pdf", dpi=250)
    plt.show()
    exit()


def plot_n_0() -> None:
    N = 2000
    data = DirbeData.new(range(1, 2), slice(0, N))
    zodi_model = ZodiModel.from_data(data)

    gamma = zodi_model.get_param_value("gamma")
    n_0s = np.linspace(1e-3, 1e-2, 10)
    cmap = plt.get_cmap("rainbow")
    for i, n_0 in enumerate(n_0s):
        theta = (n_0, gamma)
        s_zodi = zodi_model(theta)
        plt.plot(
            s_zodi, label=f"n_0={n_0:.0e}", color=cmap(np.linspace(0, 1, len(n_0s)))[i]
        )

    plt.xlabel("Observations")
    plt.ylabel("MJy/sr")
    plt.legend()
    plt.savefig(FIG_DIR / "n_0.pdf", dpi=250)
    plt.show()


def plot_gamma() -> None:
    N = 2000
    data = DirbeData.new(range(1, 2), slice(0, N))
    model = ZodiModel.from_data(data)

    n_0 = model.get_param_value("n_0")
    gammas = np.linspace(0.7, 1.3, 10)
    cmap = plt.get_cmap("rainbow")
    for i, gamma in enumerate(gammas):
        theta = (n_0, gamma)
        s_zodi = model(theta)
        plt.plot(
            s_zodi,
            label=rf"$\gamma={gamma:.2f}$",
            color=cmap(np.linspace(0, 1, len(gammas)))[i],
        )

    plt.xlabel("Observations")
    plt.ylabel("MJy/sr")
    plt.legend(loc="upper right")
    plt.savefig(FIG_DIR / "gamma.pdf", dpi=250)
    plt.show()


def grid_chisq() -> None:
    """Plots gridded chisq for gamma and n_0"""
    N = 2000
    # read in the data
    data = DirbeData.new(range(1, 2), slice(0, N))

    # initialize the model f(theta)
    zodi_model = ZodiModel.from_data(data)

    # get the true signal y
    zodi_timestream = (
        np.squeeze(data.zodi) if len(data.zodi) == 1 else np.concatenate(data.zodi)
    )

    # add noise to the true signal
    s_zodi_obs = zodi_timestream + np.random.randn(zodi_timestream.size)

    gamma_0 = zodi_model.get_param_value("gamma")
    n_0 = zodi_model.get_param_value("n_0")

    n_0s = np.linspace(n_0 * 0.5, n_0 * 2, 50)
    gammas = np.linspace(gamma_0 * 0.5, gamma_0 * 2, 50)
    chisqs = np.zeros((n_0s.size, gammas.size))

    for i, n_0 in enumerate(n_0s):
        for j, gamma in enumerate(gammas):
            theta = (n_0, gamma)
            s_zodi = zodi_model(theta)
            chisqs[i, j] = ((s_zodi_obs - s_zodi) ** 2).sum()

    plt.pcolor(n_0s, gammas, chisqs - chisqs.min(), vmin=0, vmax=50 * N, cmap="viridis")
    plt.xlabel(r"$n_0$")
    plt.ylabel(r"$\gamma$")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "grid_chisq.pdf", dpi=250)
    plt.show()



def mhmc(iterations: int, save_figs: bool = False) -> None:
    """Run the Metropolis-Hastings Markov Chain Monte Carlo algorithm and plot the results."""

    n_days = 4
    n_tods = 2000

    # read in the data
    data = DirbeData.new(range(1, n_days), slice(0, n_tods))

    # initialize the model f(theta)
    zodi_model = ZodiModel.from_data(data)

    # get the true signal y
    zodi_timestream = (
        np.squeeze(data.zodi) if len(data.zodi) == 1 else np.concatenate(data.zodi)
    )

    # add noise to the true signal
    zodi_and_noise_timestream = zodi_timestream + np.random.randn(zodi_timestream.size)

    # set the initial guess and step sizes
    theta_0 = (1.13e-7, 0.95)
    step_sizes = (0.001e-07, 0.0005)

    print("n_0 og:", theta_0[0])
    print("gamma og:", theta_0[1])

    theta_trace, chisq_trace = metropolis_hastings(
        f=zodi_model,
        y=zodi_and_noise_timestream,
        theta_0=theta_0,
        step_sizes=step_sizes,
        iterations=iterations,
    )

    print("n_0:", theta_trace[0, -1])
    print("gamma:", theta_trace[1, -1])

    # visualize the data
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(theta_trace[0])
    ax[1].plot(theta_trace[1])
    ax[2].plot(chisq_trace)

    ax[0].set_ylabel(r"$n_0$")
    ax[1].set_ylabel(r"$\gamma$")
    ax[2].set_ylabel(r"$\chi^2$")

    ax[2].set_xlabel("Iteration")
    plt.tight_layout()
    if save_figs:
        plt.savefig(FIG_DIR / "mcmc_new.pdf", dpi=250)

    fig, ax = plt.subplots(3, 1, sharex=True)
    x = np.arange(1001, 10000)
    ax[0].plot(x, theta_trace[0, 1000:])
    ax[1].plot(x, theta_trace[1, 1000:])
    ax[2].plot(x, chisq_trace[1000:])

    ax[0].set_ylabel(r"$n_0$")
    ax[1].set_ylabel(r"$\gamma$")
    ax[2].set_ylabel(r"$\chi^2$")

    ax[2].set_xlabel("Iteration")
    plt.tight_layout()
    if save_figs:
        plt.savefig(FIG_DIR / "mcmc_postburnin_new.pdf", dpi=250)

    corner.corner(theta_trace[:, 1000:].T, labels=[r"$n_0$", r"$\gamma$"])
    plt.tight_layout()
    if save_figs:
        plt.savefig(FIG_DIR / "corner_new.pdf", dpi=250)
    plt.show()


if __name__ == "__main__":
    # plot_timestream()
    # plot_n_0()
    # plot_gamma()
    # fit_n_0()
    # grid_chisq()
    mhmc(iterations=10000, save_figs=False)
