from typing import Self
import astropy.units as u
import numpy as np
from astropy.time import Time
from zodipy import Zodipy
from dirbe_data import DirbeData

DEFAULT_FREQ = 12 * u.micron
DEFAULT_NSIDE = 512


class ZodiModel(Zodipy):
    """Extended ZodiPy model which evaluates the diffuse cloud component for a given
    scanning strategy and new `n_0` and `gamma` values."""

    def __init__(
        self,
        pointing: list[np.ndarray],
        obs_pos: list[u.Quantity[u.AU]],
        obs_time: list[Time],
        freq: u.Quantity[u.micron],
        nside: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pointing = pointing
        self.obs_pos = obs_pos
        self.obs_time = obs_time
        self.freq = freq
        self.nside = nside

        model = self.get_parameters()
        cloud_model = model["comps"]["cloud"]
        model["comps"] = {"cloud": cloud_model}
        self.update_parameters(model)

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate the ZodiPy diffuse cloud model over a scanning strategy with new
        `n_0` and `gamma` parameters.
        """

        # update the zodipy model with new parameters
        theta_0 = self.get_parameters()
        n_0, gamma = theta
        theta_0["comps"]["cloud"]["n_0"] = n_0
        theta_0["comps"]["cloud"]["gamma"] = gamma
        self.update_parameters(theta_0)

        # evaluate the model over the scanning strategy
        emission_timestream: list[np.ndarray] = []
        for pix, obs_pos, obs_time in zip(self.pointing, self.obs_pos, self.obs_time):
            emission_timestream.append(
                self.get_emission_pix(
                    freq=self.freq,
                    nside=self.nside,
                    pixels=pix,
                    obs_pos=obs_pos,
                    obs_time=obs_time,
                    coord_in="G",
                ).value
            )
        return (
            np.squeeze(emission_timestream)
            if len(emission_timestream) == 1
            else np.concatenate(emission_timestream)
        )

    def get_param_value(self, param: str) -> float:
        """Get the value of a parameter in the model."""

        return self.get_parameters()["comps"]["cloud"][param]

    @classmethod
    def from_data(
        cls,
        data: DirbeData,
        freq: u.Quantity[u.micron] = DEFAULT_FREQ,
        nside: int = DEFAULT_NSIDE,
    ) -> Self:
        """Create a new instance of the ZodiModel from a DirbeData instance."""

        return cls(
            pointing=data.pixels,
            obs_pos=data.obs_pos,
            obs_time=data.obs_time,
            freq=freq,
            nside=nside,
            model="DIRBE",
            parallel=False,
        )
