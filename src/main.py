from pathlib import Path
from functools import partial

import corner
import matplotlib.pyplot as plt
import numpy as np

from data_containers import DirbeData, HoggData
from models import (
    ZodiModel,
    linear_model,
    zodi_gradient_loglikelihood,
    linear_regression_gradient_loglikelihood,
)
from sampling import metropolis_hastings, hamiltonian

np.random.seed(42059)

FIG_DIR = Path("../figs/")
FIG_DIR.mkdir(exist_ok=True)
TABLE = Path("../data/hogg-table1.txt")


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


def plot_alpha() -> None:
    N = 2000
    data = DirbeData.new(range(1, 2), slice(0, N))
    model = ZodiModel.from_data(data)

    gamma = model.get_param_value("gamma")
    alphas = np.linspace(1, 3, 10)
    cmap = plt.get_cmap("coolwarm")
    for i, alpha in enumerate(alphas):
        theta = (alpha, gamma)
        s_zodi = model(theta)
        plt.plot(
            s_zodi,
            label=rf"$\alpha={alpha:.2f}$",
            color=cmap(np.linspace(0, 1, len(alphas)))[i],
        )

    plt.xlabel("Observations")
    plt.ylabel("MJy/sr")
    plt.legend(loc="upper right")
    plt.savefig(FIG_DIR / "alpha.pdf", dpi=250)
    plt.show()


def plot_gamma() -> None:
    N = 2000
    data = DirbeData.new(range(1, 2), slice(0, N))
    model = ZodiModel.from_data(data)

    alpha = model.get_param_value("alpha")
    gammas = np.linspace(0.7, 1.3, 10)
    cmap = plt.get_cmap("coolwarm")
    for i, gamma in enumerate(gammas):
        theta = (alpha, gamma)
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

    alpha = zodi_model.get_param_value("alpha")
    gamma = zodi_model.get_param_value("gamma")
    print(alpha, gamma)
    alphas = np.linspace(1, 1.6, 50)
    gammas = np.linspace(0.6, 1.2, 50)
    chisqs = np.zeros((alphas.size, gammas.size))

    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            theta = (alpha, gamma)
            s_zodi = zodi_model(theta)
            chisqs[i, j] = ((s_zodi_obs - s_zodi) ** 2).sum()

    plt.pcolor(gammas, alphas, chisqs, cmap="viridis", vmin=chisqs.min(), vmax=4000)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\gamma$")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "grid_chisq.pdf", dpi=250)
    plt.show()


def mh(iterations: int, save_figs: bool = False) -> None:
    """Run the Metropolis-Hastings Markov Chain Monte Carlo algorithm and plot the results."""

    n_days = 2
    n_tods = 1000

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
    theta_0 = (1.2, 0.92)
    step_sizes = (0.01, 0.005)

    print("alpha og:", theta_0[0])
    print("gamma og:", theta_0[1])

    # print("n_0 og:", theta_0[0]*theta_0_phys[0])
    # print("gamma og:", theta_0[1]*theta_0_phys[1])

    theta_trace, chisq_trace = metropolis_hastings(
        f=zodi_model,
        y=zodi_and_noise_timestream,
        # theta_0=theta_0_phys,
        theta_0=theta_0,
        step_sizes=step_sizes,
        iterations=iterations,
    )

    # theta_trace[0, :] *= theta_0[0]
    # theta_trace[1, :] *= theta_0[1]

    print("alpha:", theta_trace[0, -1])
    print("gamma:", theta_trace[1, -1])

    np.save("theta_trace_mh.npy", theta_trace)
    np.save("chisq_trace_mh.npy", chisq_trace)

    # print ML, mean and std of the parameters
    print("ML alpha:", theta_trace[0, np.argmin(chisq_trace)])
    print("ML gamma:", theta_trace[1, np.argmin(chisq_trace)])
    print("mean alpha:", np.mean(theta_trace[0]))
    print("mean gamma:", np.mean(theta_trace[1]))
    print("std alpha:", np.std(theta_trace[0]))
    print("std gamma:", np.std(theta_trace[1]))

    # visualize the data
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(theta_trace[0])
    ax[1].plot(theta_trace[1])
    ax[2].plot(chisq_trace)

    ax[0].set_ylabel(r"$\alpha$")
    ax[1].set_ylabel(r"$\gamma$")
    ax[2].set_ylabel(r"$\chi^2$")

    ax[2].set_xlabel("Iteration")
    plt.tight_layout()
    if save_figs:
        plt.savefig(FIG_DIR / "mh.pdf", dpi=250)

    fig, ax = plt.subplots(3, 1, sharex=True)
    x = np.arange(1001, 10000)
    ax[0].plot(x, theta_trace[0, 1000:])
    ax[1].plot(x, theta_trace[1, 1000:])
    ax[2].plot(x, chisq_trace[1000:])

    ax[0].set_ylabel(r"$\alpha$")
    ax[1].set_ylabel(r"$\gamma$")
    ax[2].set_ylabel(r"$\chi^2$")

    ax[2].set_xlabel("Iteration")
    plt.tight_layout()
    if save_figs:
        plt.savefig(FIG_DIR / "mh_postburnin.pdf", dpi=250)

    corner.corner(theta_trace[:, 1000:].T, labels=[r"$\alpha$", r"$\gamma$"])
    plt.tight_layout()
    if save_figs:
        plt.savefig(FIG_DIR / "mh_corner.pdf", dpi=250)
    plt.show()


def mhmc_linear(iterations: int, save_figs: bool = False) -> None:
    """Run the Metropolis-Hastings Markov Chain Monte Carlo algorithm and plot the results."""

    data = HoggData.from_table(TABLE)

    theta_0 = (1.8, 25)
    step_sizes = (0.01, 0.1)

    included_data = slice(5, None)

    f_lin = partial(linear_model, x=data.x[included_data])

    print("m og:", theta_0[0])
    print("b og:", theta_0[1])

    theta_trace = np.load("theta_trace.npy")
    cov = np.cov(theta_trace)
    # print(cov)
    # exit()
    theta_trace, chisq_trace = metropolis_hastings(
        iterations=iterations,
        f=f_lin,
        theta_0=theta_0,
        step_sizes=step_sizes,
        y=data.y[included_data],
        sigma_y=data.sigma_y[included_data],
        cov=cov,
    )

    print("m:", theta_trace[0, -1])
    print("b:", theta_trace[1, -1])

    # visualize the data
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(theta_trace[0, 1000:])
    ax[1].plot(theta_trace[1, 1000:])
    ax[2].plot(chisq_trace[1000:])

    ax[0].set_ylabel(r"$m$")
    ax[1].set_ylabel(r"$b$")
    ax[2].set_ylabel(r"$\chi^2$")

    ax[2].set_xlabel("Iteration")
    plt.tight_layout()
    if save_figs:
        plt.savefig(FIG_DIR / "mcmc_lin.pdf", dpi=250)

    x_new = np.linspace(0, 300, 20)
    m, b = theta_trace[0, -1], theta_trace[1, -1]
    y_hat = linear_model((m, b), x_new)

    plt.figure()
    plt.errorbar(
        data.x[included_data],
        data.y[included_data],
        yerr=data.sigma_y[included_data],
        fmt="o",
    )
    plt.plot(x_new, y_hat, label=rf"$y = {m:.2f}x + {b:.2f}$")
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    corner.corner(theta_trace[:, 1000:].T, labels=[r"$m$", r"$b$"])
    plt.tight_layout()
    if save_figs:
        plt.savefig(FIG_DIR / "corner_lin.pdf", dpi=250)

    plt.show()


def mhc(iterations: int, L: int, eps: float, save_figs: bool = False) -> None:
    n_days = 2
    n_tods = 1000
    params = ["alpha", "gamma"]

    # read in the data
    data = DirbeData.new(range(1, n_days), slice(0, n_tods))

    # initialize the model f(theta)
    zodi_model = ZodiModel.from_data(data)
    alpha = zodi_model.get_param_value("alpha")
    gamma = zodi_model.get_param_value("gamma")

    y = np.squeeze(np.concatenate([data.zodi]))
    sigma_y = np.ones_like(y)

    theta_0 = (1.2, 0.92)
    gradient = partial(
        zodi_gradient_loglikelihood, f=zodi_model, y=y, sigma_y=sigma_y, params=params
    )

    print("alpha:", alpha)
    print("gamma:", gamma)

    theta_trace, chisq_trace = hamiltonian(
        iterations=iterations,
        f=zodi_model,
        gradient=gradient,
        y=y,
        sigma_y=sigma_y,
        theta_0=theta_0,
        path_length=L,
        step_size=eps,
    )

    print("alpha:", theta_trace[0, -1])
    print("gamma:", theta_trace[1, -1])

    np.save("theta_trace_m.npy", theta_trace)
    np.save("chisq_trace_m.npy", chisq_trace)

    # print ML, mean and std of the parameters
    print("ML alpha:", theta_trace[0, np.argmin(chisq_trace)])
    print("ML gamma:", theta_trace[1, np.argmin(chisq_trace)])
    print("mean alpha:", np.mean(theta_trace[0]))
    print("mean gamma:", np.mean(theta_trace[1]))
    print("std alpha:", np.std(theta_trace[0]))
    print("std gamma:", np.std(theta_trace[1]))

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(theta_trace[0])
    ax[1].plot(theta_trace[1])
    ax[2].plot(chisq_trace)

    ax[0].set_ylabel(r"$\alpha$")
    ax[1].set_ylabel(r"$\gamma$")
    ax[2].set_ylabel(r"$\chi^2$")

    ax[2].set_xlabel("Iteration")
    plt.tight_layout()
    if save_figs:
        plt.savefig(FIG_DIR / "hmc.pdf", dpi=250)

    if theta_trace.shape[1] > 200:
        # visualize the data
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(theta_trace[0][200:])
        ax[1].plot(theta_trace[1][200:])
        ax[2].plot(chisq_trace[200:])

        ax[0].set_ylabel(r"$\alpha$")
        ax[1].set_ylabel(r"$\gamma$")
        ax[2].set_ylabel(r"$\chi^2$")

        ax[2].set_xlabel("Iteration")
        plt.tight_layout()
        if save_figs:
            plt.savefig(FIG_DIR / "hmc_burnin.pdf", dpi=250)

        corner.corner(theta_trace.T, labels=[r"$\alpha$", r"$\gamma$"])
        plt.tight_layout()
        if save_figs:
            plt.savefig(FIG_DIR / "hmc_corner.pdf", dpi=250)
    plt.show()


def mhc_linear(iterations: int, L: int, eps: float, save_figs: bool = False) -> None:
    """Run the Metropolis-Hastings Markov Chain Monte Carlo algorithm and plot the results."""

    data = HoggData.from_table(TABLE)

    theta_0 = (2.0, 30.0)

    included_data = slice(5, None)

    f_lin = partial(linear_model, x=data.x[included_data])

    print("m og:", theta_0[0])
    print("b og:", theta_0[1])

    gradient = partial(
        linear_regression_gradient_loglikelihood,
        x=data.x[included_data],
        y=data.y[included_data],
        sigma_y=data.sigma_y[included_data],
    )
    theta_trace, chisq_trace = hamiltonian(
        iterations=iterations,
        f=f_lin,
        gradient=gradient,
        theta_0=theta_0,
        y=data.y[included_data],
        sigma_y=data.sigma_y[included_data],
        path_length=L,
        step_size=eps,
    )

    print("m:", theta_trace[0, -1])
    print("b:", theta_trace[1, -1])

    theta_trace = theta_trace[:, 1000:]
    chisq_trace = chisq_trace[1000:]
    # visualize the data
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(theta_trace[0])
    ax[1].plot(theta_trace[1])
    ax[2].plot(chisq_trace)

    ax[0].set_ylabel(r"$m$")
    ax[1].set_ylabel(r"$b$")
    ax[2].set_ylabel(r"$\chi^2$")

    ax[2].set_xlabel("Iteration")
    plt.tight_layout()
    if save_figs:
        plt.savefig(FIG_DIR / "mhc_lin.pdf", dpi=250)

    x_new = np.linspace(0, 300, 20)
    m, b = theta_trace[0, -1], theta_trace[1, -1]
    y_hat = linear_model((m, b), x_new)

    plt.figure()
    plt.errorbar(
        data.x[included_data],
        data.y[included_data],
        yerr=data.sigma_y[included_data],
        fmt="o",
    )
    plt.plot(x_new, y_hat, label=rf"$y = {m:.2f}x + {b:.2f}$")
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    corner.corner(theta_trace.T, labels=[r"$m$", r"$b$"])
    plt.tight_layout()
    if save_figs:
        plt.savefig(FIG_DIR / "mhc_corner_lin.pdf", dpi=250)

    plt.show()


if __name__ == "__main__":
    # plot_timestream()
    # plot_n_0()
    # plot_alpha()
    # plot_gamma()
    # fit_n_0()
    # grid_chisq()
    mh(iterations=10000, save_figs=False)
    # mhmc_linear(iterations=10000, save_figs=False)
    # mhc(iterations=5000, L=10, eps=0.001, save_figs=False)
    # mhc_linear(iterations=10000, L=25, eps=0.001, save_figs=False)
