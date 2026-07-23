"""
the flow models p(log P_rot | log age, mass, sigma_mass). 
an age posterior is obtained by evaluating that density along the age grid at a fixed mass and rotation period,
multiplying by the age prior and then normalizing.
"""
from dataclasses import dataclass
import numpy as np
from .constants import LOGA_GRID

__all__ = ["AgePosterior", "flat_in_log_age", "flat_in_age"]

def flat_in_log_age(loga_grid: np.ndarray) -> np.ndarray:
    return np.zeros_like(loga_grid)

def flat_in_age(loga_grid: np.ndarray) -> np.ndarray:
    return loga_grid * np.log(10)


@dataclass(frozen=True)
class AgePosterior:
    """
    normalized age posterior on a log10(age / Myr) grid
    """

    loga_grid: np.ndarray
    pdf: np.ndarray

    @property
    def age_gyr_grid(self) -> np.ndarray:
        return 10 ** self.loga_grid / 1000

    def _quantile(self, q: float) -> float:
        cdf = np.cumsum(self.pdf) * np.gradient(self.loga_grid)
        cdf /= cdf[-1]
        return float(np.interp(q, cdf, self.loga_grid))

    @property
    def median_loga(self) -> float:
        return self._quantile(0.5)

    @property
    def median_gyr(self) -> float:
        """
        median of the age posterior (in Gyr)
        """
        return 10 ** self.median_loga / 1000

    @property
    def mode_gyr(self) -> float:
        """
        age in Gyr at highest posterior density
        """
        return 10 ** self.loga_grid[int(np.argmax(self.pdf))] / 1000

    def interval(self, level: float = 0.68) -> tuple[float, float]:
        """Equal-tailed credible interval in Gyr."""
        lo = self._quantile((1 - level) / 2)
        hi = self._quantile(1 - (1 - level) / 2)
        return 10 ** lo / 1000, 10 ** hi / 1000

    def summary(self, level: float = 0.68) -> dict:
        lo, hi = self.interval(level)
        med = self.median_gyr
        return {
            "age_gyr": med,
            "age_err_lo_gyr": med - lo,
            "age_err_hi_gyr": hi - med,
            "level": level,
        }

    def __repr__(self) -> str:
        lo, hi = self.interval()
        return (f"AgePosterior({self.median_gyr:.2f} "
                f"-{self.median_gyr - lo:.2f}/+{hi - self.median_gyr:.2f} Gyr, 68%)")


def normalize(log_probs: np.ndarray, loga_grid: np.ndarray = LOGA_GRID) -> np.ndarray:
    """
    exponentiate and normalize densities over the age grid
    """
    log_probs = np.atleast_2d(log_probs)
    log_probs = log_probs - log_probs.max(axis=1, keepdims=True)  # stability
    probs = np.exp(log_probs)
    probs /= np.trapezoid(probs, loga_grid, axis=1)[:, None]
    return probs
