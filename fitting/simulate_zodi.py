from dataclasses import dataclass, asdict
from pathlib import Path

import astropy.units as u
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import zodipy
from astropy.time import Time

START_DAY = 50
VALID_DAYS = range(START_DAY, START_DAY + 10)
SCANNING_STRATEGY_DATA_PATH = Path("../data/scanning_strategy_dirbe_05.h5")
FREQ = 12 * u.micron
NSIDE = 512


@dataclass
class ScanningStrategyDay:
    """A simple dataclass to hold the scanning strategy for a given day."""

    pixels: np.ndarray
    obs_pos: u.Quantity[u.AU]
    obs_time: Time


def get_dirbe_day_scanning_strategy(day: int) -> ScanningStrategyDay:
    """Reads in the pointing timestream for a given day."""
    if day not in VALID_DAYS:
        raise ValueError(
            f"Day {day} for the DIRBE pointing is not tabulate in this exercise."
            f"Valid days are {VALID_DAYS}"
        )
    with h5py.File(SCANNING_STRATEGY_DATA_PATH, "r") as file:
        pointing = file[f"{day:02}/pix"][()]
        obs_pos = file[f"{day:02}/satpos"][()]
        time = file[f"{day:02}/time"][()]

    obs_pos = u.Quantity(obs_pos, u.AU)
    time = Time(time, format="mjd", scale="utc")

    return ScanningStrategyDay(
        pixels=pointing,
        obs_pos=obs_pos,
        obs_time=time,
    )


def main() -> None:
    model = zodipy.Zodipy("DIRBE", parallel=True)
    week_map = u.Quantity(np.zeros(hp.nside2npix(NSIDE)), u.MJy / u.sr)
    hit_map = np.zeros_like(week_map)

    for day in VALID_DAYS:
        scanning_strategy = get_dirbe_day_scanning_strategy(day)

        emission_timestream = model.get_emission_pix(
            **asdict(scanning_strategy),
            freq=FREQ,
            nside=NSIDE,
            return_comps=True,
        )[0]  # only cloud component

        unique_pix, counts = np.unique(scanning_strategy.pixels, return_counts=True)
        week_map[unique_pix] 
        bin_count = np.bincount(scanning_strategy.pixels, weights=emission_timestream, minlength=week_map.size)
        week_map[:len(bin_count)] += bin_count
        hit_map[unique_pix] += counts

    week_map /= hit_map
    week_map[hit_map == 0] = hp.UNSEEN
    hp.mollview(week_map, title="DIRBE 12 micron zodiacal light", unit="MJy/sr")
    plt.show()


if __name__ == "__main__":
    main()
