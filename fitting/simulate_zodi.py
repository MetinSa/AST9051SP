from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Self

import astropy.units as u
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import zodipy
from astropy.time import Time

START_DAY = 50
VALID_DAYS = range(1, 11)
OBS_DAYS = range(50, 60)
SCANNING_STRATEGY_DATA_PATH = Path("../data/scanning_strategy_dirbe_05.h5")
FIG_DIR = Path("../figs/")
FREQ = 12 * u.micron
NSIDE = 512


@dataclass
class DirbeData:
    """Container for DIRBE data."""

    zodi: list[np.ndarray]
    pixels: list[np.ndarray]
    obs_pos: list[u.Quantity[u.AU]]
    obs_time: list[Time]

    @property
    def n_days(self) -> int:
        return len(self.obs_time)

    @classmethod
    def new(cls, days: range = VALID_DAYS, slc: slice = slice(None)) -> Self:
        if not set((days)).issubset(VALID_DAYS):
            raise ValueError(
                f"Days {days} for the DIRBE pointing are not tabulate in this exercise."
                f"Valid days are {VALID_DAYS}"
            )

        zodi: list[np.ndarray] = []
        pixels: list[np.ndarray] = []
        obs_pos: list[u.Quantity[u.AU]] = []
        obs_time: list[Time] = []

        for day in days:
            day += START_DAY - 1
            with h5py.File(SCANNING_STRATEGY_DATA_PATH, "r") as file:
                s_zodi: np.ndarray = file[f"{day:02}/zodi"][slc]
                s_pix: np.ndarray = file[f"{day:02}/pix"][slc]
                obs_pos_day = u.Quantity(file[f"{day:02}/satpos"], u.AU)
                obs_time_day = Time(file[f"{day:02}/time"], format="mjd", scale="utc")

            indices = pixels != 0
            s_zodi = s_zodi[indices]
            s_pix = s_pix[indices]

            zodi.append(s_zodi)
            pixels.append(s_pix)
            obs_pos.append(obs_pos_day)
            obs_time.append(obs_time_day)

        return cls(zodi=zodi, pixels=pixels, obs_pos=obs_pos, obs_time=obs_time)


class FittingModel(zodipy.Zodipy):
    def __init__(self, data: DirbeData, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data = data

    def update(self, theta: dict[str, float]) -> None:
        """Evalues a model given a set of parameters theta."""

        theta_0 = self.get_parameters()
        for param, value in theta.items():
            theta_0["comps"]["cloud"][param] = value
        self.update_parameters(theta_0)

    def evaluate(self) -> np.ndarray:
        """Evaluate a zodpy model over the dirbe scanning strategy over a range of days."""

        emission_timestream: list[np.ndarray] = []
        for pix, obs_pos, obs_time in zip(
            self.data.pixels, self.data.obs_pos, self.data.obs_time
        ):
            emission_timestream.append(
                self.get_emission_pix(
                    freq=FREQ,
                    nside=NSIDE,
                    pixels=pix,
                    obs_pos=obs_pos,
                    obs_time=obs_time,
                    return_comps=True,
                    coord_in="G",
                ).value[0]  # only cloud component
            )
        if len(emission_timestream) > 1:
            return np.concatenate(emission_timestream)

        return emission_timestream[0]


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


def get_param(model: zodipy.Zodipy, theta: str) -> float:
    return model.get_parameters()["comps"]["cloud"][theta]


def write_timestream() -> None:
    model = zodipy.Zodipy("DIRBE", parallel=True)
    with h5py.File(SCANNING_STRATEGY_DATA_PATH, "r+") as file:
        for day in range(50, 60):
            group = file[f"{day:02}"]

            pointing: np.ndarray = file[f"{day:02}/pix"][()]
            obs_pos: np.ndarray = file[f"{day:02}/satpos"][()]
            time: float = file[f"{day:02}/time"][()]

            obs_pos = u.Quantity(obs_pos, u.AU)
            obs_time = Time(time, format="mjd", scale="utc")

            emission_timestream: np.ndarray = model.get_emission_pix(
                pixels=pointing,
                obs_pos=obs_pos,
                obs_time=obs_time,
                freq=FREQ,
                nside=NSIDE,
                return_comps=True,
                coord_in="G",
            ).value[0]  # only cloud component
            group.create_dataset("zodi", data=emission_timestream)



def grid_chisq() -> None:
    """Plots gridded chisq for gamma and n_0"""
    N = 2000
    data = DirbeData.new(range(1, 2), slice(0, N))
    model = FittingModel(data=data, model="DIRBE", parallel=False)
    model_0 = FittingModel(data=data, model="DIRBE", parallel=False)

    s_zodi_true = model_0.evaluate()
    plt.plot(s_zodi_true)
    plt.ylabel("MJy/sr")
    plt.xlabel("Observations")
    plt.savefig(FIG_DIR / "timestream.pdf", dpi=250)
    plt.show()
    exit()
    s_noise = np.random.randn(N)

    s_zodi_obs = s_zodi_true + s_noise

    gamma_0 = get_param(model_0, "gamma")
    n_0 = get_param(model_0, "n_0")

    n_0s = np.linspace(n_0 * 0.5, n_0 * 2, 25)
    gammas = np.linspace(gamma_0 * 0.5, gamma_0 * 2, 25)
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
            print(chisqs[i, j])
    
    plt.pcolor(n_0s, gammas, chisqs-chisqs.min(), vmin=0, vmax=5*N)
    plt.colorbar()
    plt.show()

def main():
    grid_chisq()
    

if __name__ == "__main__":
    main()
