import numpy as np

# age grid (1 Myr to ~13.8 Gyr).
LOGA_GRID = np.linspace(0, 4.14, 1000)

# prior bounds on log10(P_rot / days)
# used in the outlier component of the loss.
PRIOR_LOGPROT = (-1.75, 2.5)

# grid for evaluating p(log P_rot | age, mass)
LOGPROT_GRID = np.linspace(PRIOR_LOGPROT[0], PRIOR_LOGPROT[1], 500)

# prior bounds on log10=(age / Myr).
PRIOR_LOGA_MYR = (0, 4.14)

# outliers -> Fraction of stars assumed not to follow rotational evolution 
P_OUT = 0.05
