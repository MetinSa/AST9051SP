import astropy.units as u
import numpy as np
from dirbe_data import DirbeData
from zodipy import Zodipy

NSIDE = 512


class FittingModel(Zodipy):
    def __init__(self, data: DirbeData, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data = data

    def update(self, theta: dict[str, float]) -> None:
        """Evalues a model given a set of parameters theta."""

        theta_0 = self.get_parameters()
        for param, value in theta.items():
            theta_0["comps"]["cloud"][param] = value
        self.update_parameters(theta_0)

    def evaluate(self, freq: u.Quantity[u.micron] = 25 * u.micron) -> np.ndarray:
        """Evaluate a zodpy model over the dirbe scanning strategy over a range of days."""

        emission_timestream: list[np.ndarray] = []
        for pix, obs_pos, obs_time in zip(
            self.data.pixels, self.data.obs_pos, self.data.obs_time
        ):
            emission_timestream.append(
                self.get_emission_pix(
                    freq=freq,
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

    def get_param_value(self, param: str) -> float:
        """Get the value of a parameter in the model."""

        return self.get_parameters()["comps"]["cloud"][param]
