from dataclasses import dataclass
from pathlib import Path
from typing import Self

import astropy.units as u
import h5py
import numpy as np
from astropy.time import Time

START_DAY = 50
VALID_DAYS = range(1, 11)
SCANNING_STRATEGY_DATA_PATH = Path("../data/scanning_strategy_dirbe_05.h5")


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

            indices = s_pix != 0
            s_zodi = s_zodi[indices]
            s_pix = s_pix[indices]

            zodi.append(s_zodi)
            pixels.append(s_pix)
            obs_pos.append(obs_pos_day)
            obs_time.append(obs_time_day)

        return cls(zodi=zodi, pixels=pixels, obs_pos=obs_pos, obs_time=obs_time)


@dataclass
class HoggData:
    """Container for fitting a line to data table from Hogg et al. 2010."""

    x: np.ndarray
    y: np.ndarray
    sigma_y: np.ndarray
    sigma_x: np.ndarray
    rho_xy: np.ndarray

    @classmethod
    def from_table(cls, table_path: Path) -> Self:
        table = np.loadtxt(table_path, usecols=(1, 2, 3, 4, 5), unpack=True)
        return cls(*table)
