from pathlib import Path

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import zodipy

from dirbe_data import DirbeData
from extended_zodipy import FittingModel

START_DAY = 50
VALID_DAYS = range(1, 11)
OBS_DAYS = range(50, 60)
SCANNING_STRATEGY_DATA_PATH = Path("../data/scanning_strategy_dirbe_05.h5")
FIG_DIR = Path("../figs/")
FIG_DIR.mkdir(exist_ok=True)
FREQ = 12 * u.micron
NSIDE = 512


def get_binned_map(model: zodipy.Zodipy, data: DirbeData) -> np.ndarray:
    """Evaluate a zodpy model over the dirbe scanning strategy over a range of days."""

    binned_map = np.zeros(hp.nside2npix(NSIDE))
    hit_map = np.zeros_like(binned_map)
    for pix, obs_pos, obs_time in zip(data.pixels, data.obs_pos, data.obs_time):
        emission_timestream: np.ndarray = model.get_emission_pix(
            freq=FREQ,
            nside=NSIDE,
            pixels=pix,
            obs_pos=obs_pos,
            obs_time=obs_time,
            return_comps=True,
            coord_in="G",
        ).value[0]  # only cloud component

        unique_pix, counts = np.unique(pix, return_counts=True)
        binned_map[unique_pix]
        bin_count = np.bincount(
            pix,
            weights=emission_timestream,
            minlength=binned_map.size,
        )
        binned_map[: len(bin_count)] += bin_count
        hit_map[unique_pix] += counts

    binned_map /= hit_map
    binned_map[hit_map == 0] = hp.UNSEEN

    return binned_map


def grid_chisq() -> None:
    """Plots gridded chisq for gamma and n_0"""
    N = 2000
    data = DirbeData.new(range(1, 2), slice(0, N))
    model = FittingModel(data=data, model="DIRBE", parallel=False)
    model_0 = FittingModel(data=data, model="DIRBE", parallel=False)

    s_zodi_true = model_0.evaluate()
    s_noise = np.random.randn(N)

    s_zodi_obs = s_zodi_true + s_noise

    gamma_0 = model_0.get_param_value("gamma")
    n_0 = model_0.get_param_value("n_0")

    n_0s = np.linspace(n_0 * 0.5, n_0 * 2, 50)
    gammas = np.linspace(gamma_0 * 0.5, gamma_0 * 2, 50)
    chisqs = np.zeros((n_0s.size, gammas.size))

    for i, n_0 in enumerate(n_0s):
        for j, gamma in enumerate(gammas):
            theta = {
                "n_0": n_0,
                "gamma": gamma,
            }
            model.update(theta)

            s_zodi = model.evaluate()
            chisqs[i, j] = ((s_zodi_obs - s_zodi) ** 2).sum()

    plt.pcolor(
        n_0s, gammas, chisqs - chisqs.min(), vmin=0, vmax=50 * N, cmap="viridis"
    )
    plt.xlabel(r"$n_0$")
    plt.ylabel(r"$\gamma$")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "grid_chisq.pdf", dpi=250)
    plt.show()


def plot_timestream() -> None:
    N = 2000
    data = DirbeData.new(range(1, 2), slice(0, N))
    model = FittingModel(data=data, model="DIRBE")

    s_zodi_true = model.evaluate()
    s_noise = np.random.randn(N)

    s_zodi_obs = s_zodi_true + s_noise

    plt.plot(s_zodi_obs, label="Observed")
    plt.plot(s_zodi_true, label="True")
    plt.ylabel("MJy/sr")
    plt.xlabel("Observations")
    plt.legend()
    plt.savefig(FIG_DIR / "timestream.pdf", dpi=250)
    plt.show()
    exit()


def plot_n_0() -> None:
    N = 2000
    data = DirbeData.new(range(1, 2), slice(0, N))
    model = FittingModel(data=data, model="DIRBE")

    n_0s = np.linspace(1e-3, 1e-2, 10)
    cmap = plt.get_cmap("rainbow")
    for i, n_0 in enumerate(n_0s):
        theta = {
            "n_0": n_0,
        }
        model.update(theta)

        s_zodi = model.evaluate()
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
    model = FittingModel(data=data, model="DIRBE")

    gammas = np.linspace(0.7, 1.3, 10)
    cmap = plt.get_cmap("rainbow")
    for i, gamma in enumerate(gammas):
        theta = {
            "gamma": gamma,
        }
        model.update(theta)

        s_zodi = model.evaluate()
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


def main():
    # plot_timestream()
    # plot_n_0()
    # plot_gamma()
    # fit_n_0()
    grid_chisq()


if __name__ == "__main__":
    main()
