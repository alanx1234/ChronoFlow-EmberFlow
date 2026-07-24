# EmberFlow

Emberflow extends [ChronoFlow](https://github.com/philvanlane/chronoflow)
([Van-Lane et al. 2025](https://arxiv.org/abs/2412.12244)) into a mass-conditioned model calibrated for M dwarfs.

The model uses a conditional normalizing flow that learns the density of
rotation period given age $\tau$, stellar mass $M_\star$, and mass uncertainty $\sigma_M$:

$$p\left(\log P_{\mathrm{rot}} \mid \log  \tau\, M_\star\, \sigma_M\right)$$

We can also use Bayesian inference to estimate ages for M dwarfs from a rotation period and a stellar mass (in solar masses). 
EmberFlow returns a full age posterior per star, learned at the population level, instead of a single point estimate.
This matters for M dwarfs in particular because a single rotation period can map to multiple ages.

For convenience, the [web tool](https://chronoflow-emberflow.vercel.app/) is also available.

## Install

This project is managed with uv:

```bash
uv sync         
uv run jupyter lab # for the tutorial notebooks
```

With pip instead:

```bash
pip install -e ".[plot]"
```

## Tutorial notebooks

We include tutorial notebooks for reference on how to use EmberFlow.

| | |
|---|---|
| `01_intro.ipynb` | Age inference for a single star |
| `02_catalog_inference.ipynb` | Running the model over a catalog of stars |
| `03_learned_density.ipynb` | Visualizing the flow's learned density |
| `04_training.ipynb` | Training your own model from scratch |

## Data

`data/training_stars.csv`  includes 6,584 unique M dwarfs with
measured rotation periods and age estimates. 

**Identifiers and position**

| Column | Description |
|---|---|
| `source_paper` | Literature source of the rotation period and age (see `sources.csv`) |
| `star_name` | Star name or `Gaia DR3 <id>` for stars inherited from the ChronoFlow catalog |
| `cluster_name` | — | Cluster or association for coeval stars |
| `source_id` | Gaia DR3 source ID |
| `twomass_id` | 2MASS identifier |
| `ra`, `dec` | Right ascension and declination |

**Rotation and age** 

| Column | Unit | Description |
|---|---|---|
| `prot_days` | days | Measured rotation period |
| `age_gyr` | Gyr | Literature age |
| `age_err_lo_gyr`, `age_err_hi_gyr` | Gyr | Lower and upper age uncertainties |

**Mass and its derivation**

| Column | Unit | Description |
|---|---|---|
| `mass_msun` | $M_\odot$ | Stellar mass |
| `mass_msun_err_lo`, `mass_msun_err_hi` | $M_\odot$ | 16th / 84th-percentile mass uncertainties |
| `parallax`, `parallax_error` | mas | Gaia DR3 parallax and uncertainty |
| `k_m`, `k_cmsig` | mag | 2MASS $K_s$ magnitude and uncertainty |
| `A_Ks`, `A_Ks_err` | mag | $K_s$ extinction and uncertainty |
| `k_m_0` | mag | De-reddened $K_s$ magnitude, $K_{s,0} = K_s - A_{K_s}$ |
| `M_Ks` | mag | Absolute $K_s$ magnitude |

**Gaia photometry and quality**

| Column | Description |
|---|---|
|  `bp_rp_0` | Gaia BP–RP de-reddened color  |
| `phot_g_mean_mag` | Gaia $G$ magnitude |
| `ruwe` | Gaia RUWE |

**Magnetic activity**

| Column | Description |
|---|---|
| `tau_ce_days`, `tau_ce_err_days` |  Convective turnover timescale |
| `rossby_number` | Rossby number, $Ro = P_{\rm rot}/\tau_{\rm ce}$ |


## Citation

If you use EmberFlow please cite both this paper and Van-Lane et al. (2025). See `CITATION.cff`.

