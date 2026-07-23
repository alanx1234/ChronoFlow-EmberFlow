import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import zuko

from .constants import LOGA_GRID, LOGPROT_GRID
from .infer import AgePosterior, flat_in_log_age, normalize

WEIGHTS = Path(__file__).parent / "weights"

__all__ = ["EmberFlow"]

class EmberFlow:
    def __init__(self, flow, mean: np.ndarray, scale: np.ndarray, card: dict):
        self.flow = flow
        self._mean = mean
        self._scale = scale
        self.card = card
        self.cond_cols = card["cond_cols"]
        self._range = card["training_range"]

    @classmethod
    def load(cls, weights_dir: str | Path = WEIGHTS) -> "EmberFlow":
        """
        loads released model (trained on 6,584 stars)
        """
        d = Path(weights_dir)
        card = json.loads((d / "model_card.json").read_text(encoding="utf-8"))
        flow = zuko.flows.NSF(1, len(card["cond_cols"]),
                              transforms=card["transforms"],
                              hidden_features=tuple(card["hidden_features"]))

        flow.load_state_dict(torch.load(d / "flow.pt", weights_only=True))
        flow.eval()
        s = np.load(d / "scaler.npz")
        return cls(flow, s["mean"], s["scale"], card)


    def _normalize_cond(self, c: np.ndarray) -> torch.Tensor:
        return torch.tensor(((c - self._mean) / self._scale).astype(np.float32))

    def _warn_out_of_range(self, mass: np.ndarray, prot: np.ndarray) -> None:
        for name, vals, (lo, hi) in (
            ("mass_msun", mass, self._range["mass_msun"]),
            ("prot_days", prot, self._range["prot_days"]),
        ):
            n = int(np.sum((vals < lo) | (vals > hi)))
            if n:
                warnings.warn(
                    f"{n} star(s) outside the training range of {name} "
                    f"[{lo:.3g}, {hi:.3g}]; ages there are extrapolated and "
                    f"should not be trusted.", stacklevel=3)

    def _grid_log_probs(self, prot, mass, mass_err, loga_grid) -> np.ndarray:
        prot = np.atleast_1d(prot).astype(float)
        mass = np.atleast_1d(mass).astype(float)
        mass_err = np.atleast_1d(mass_err).astype(float)
        if mass.size == 1 and prot.size > 1:
            mass = np.repeat(mass, prot.size)
        if mass_err.size == 1 and prot.size > 1:
            mass_err = np.repeat(mass_err, prot.size)
        self._warn_out_of_range(mass, prot)

        n, g = prot.size, loga_grid.size
        cond = np.empty((n * g, 3), dtype=float)
        cond[:, 0] = np.tile(loga_grid, n)
        cond[:, 1] = np.repeat(mass, g)
        cond[:, 2] = np.repeat(mass_err, g)

        x = torch.tensor(np.repeat(np.log10(prot), g).astype(np.float32)).unsqueeze(1)
        with torch.no_grad():
            lp = self.flow(self._normalize_cond(cond)).log_prob(x).numpy()
        return lp.reshape(n, g)


    def age_posterior(self,
                      prot: float,
                      mass: float,
                      mass_err: float = 0.0,
                      prior: Callable[[np.ndarray], np.ndarray] = flat_in_log_age,
                      loga_grid: np.ndarray = LOGA_GRID,
                      ) -> AgePosterior:
        """age posterior for a single star

        prot     : rotation period in days
        mass     : stellar mass in solar masses
        mass_err : 1-sigma mass uncertainty; the model conditions on it, so
                   passing a realistic value matters
        prior    : log prior on log10(age/Myr); see infer.flat_in_age
        """
        lp = self._grid_log_probs(prot, mass, mass_err, loga_grid)
        lp = lp + prior(loga_grid)[None, :]
        return AgePosterior(loga_grid, normalize(lp, loga_grid)[0])

    def age_posteriors(self,
                       table,
                       prot_col: str = "prot_days",
                       mass_col: str = "mass_msun",
                       mass_err_col: str = "mass_msun_err",
                       prior: Callable[[np.ndarray], np.ndarray] = flat_in_log_age,
                       loga_grid: np.ndarray = LOGA_GRID,
                       ) -> np.ndarray:
        """
        a vector of posteriors for a table of stars
        """
        err = table[mass_err_col] if mass_err_col in table else np.zeros(len(table))
        lp = self._grid_log_probs(np.asarray(table[prot_col]),
                                  np.asarray(table[mass_col]),
                                  np.asarray(err), loga_grid)
        return normalize(lp + prior(loga_grid)[None, :], loga_grid)

    def summarize(self, table, level: float = 0.68, **kw):
        """
        age_posteriors() summarized median and uncertainty interval per star.
        """
        import pandas as pd
        post = self.age_posteriors(table, **kw)
        rows = [AgePosterior(kw.get("loga_grid", LOGA_GRID), p).summary(level)
                for p in post]
        return pd.DataFrame(rows, index=getattr(table, "index", None))

    def prot_density(self,
                     age_gyr,
                     mass,
                     mass_err: float = 0.0,
                     logprot_grid: np.ndarray = LOGPROT_GRID,
                     ) -> tuple[np.ndarray, np.ndarray]:
        """
        evaluate p(log10 P_rot | age, mass) on a log-period grid
        """
        scalar = np.isscalar(age_gyr) and np.isscalar(mass)
        age, m = np.broadcast_arrays(np.atleast_1d(age_gyr).astype(float),
                                     np.atleast_1d(mass).astype(float))
        age, m = age.ravel(), m.ravel()
        n, g = age.size, logprot_grid.size

        cond = np.empty((n * g, 3), dtype=float)
        cond[:, 0] = np.repeat(np.log10(age * 1000), g)
        cond[:, 1] = np.repeat(m, g)
        cond[:, 2] = mass_err

        x = torch.tensor(np.tile(logprot_grid, n).astype(np.float32)).unsqueeze(1)
        with torch.no_grad():
            lp = self.flow(self._normalize_cond(cond)).log_prob(x).numpy()

        pdf = normalize(lp.reshape(n, g), logprot_grid)
        return logprot_grid, (pdf[0] if scalar else pdf)

    def sample_prot(self,
                    age_gyr: float,
                    mass: float,
                    mass_err: float = 0.0,
                    n: int = 1000,
                    ) -> np.ndarray:
        """
        draw simulated rotation periods (days) from p(P_rot | age, mass).r
        """
        cond = np.array([[np.log10(age_gyr * 1000), mass, mass_err]], dtype=float)
        with torch.no_grad():
            s = self.flow(self._normalize_cond(cond)).sample((n,))
        return 10 ** s.numpy().reshape(-1)
