from typing import Callable, Optional, Self
import astropy.units as u
import numpy as np
from astropy.time import Time
from zodipy import Zodipy
from data_containers import DirbeData

DEFAULT_FREQ = 12 * u.micron
DEFAULT_NSIDE = 512


def linear_model(theta: tuple[float, float], x: np.ndarray) -> np.ndarray:
    """Linear model."""
    m, b = theta
    return m * x + b


def quadratic_model(theta: tuple[float, float, float], x: np.ndarray) -> np.ndarray:
    """Quadratic model."""
    q, m, b = theta
    return q * x**2 + m * x + b


class ZodiModel(Zodipy):
    """Extended ZodiPy model which evaluates the diffuse cloud component for a given
    scanning strategy and new `alpha` and `gamma` values."""

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

    def __call__(self, theta: np.ndarray, gradient: Optional[str] = None) -> np.ndarray:
        """Evaluate the ZodiPy diffuse cloud model over a scanning strategy with new
        `alpha` and `gamma` parameters.
        """

        # update the zodipy model with new parameters
        theta_0 = self.get_parameters()
        alpha, gamma = theta
        theta_0["comps"]["cloud"]["alpha"] = alpha
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
                    gradient=gradient,
                ).value
            )

        return np.squeeze(np.concatenate([emission_timestream]))

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


def linear_regression_gradient_loglikelihood(
    theta: np.ndarray, x: np.ndarray, y: np.ndarray, sigma_y: Optional[np.ndarray]
) -> np.ndarray:
    """Gradient of the log likelihood for linear regression."""

    m, b = theta
    # Calculate predicted values
    y_hat = x * m + b

    # Calculate gradient
    grad = np.array(
        [sum(((y - y_hat) / sigma_y**2) * x), sum(((y - y_hat) / sigma_y**2))]
    )

    return grad


def zodi_gradient_loglikelihood(
    theta: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    y: np.ndarray,
    sigma_y: Optional[np.ndarray],
    params: list[str],
) -> np.ndarray:
    """Gradient of the log likelihood."""

    y_hat = f(theta)
    grad = np.array(
        [sum(((y - y_hat) / sigma_y**2) * f(theta, gradient=param)) for param in params]
    )
    return grad
