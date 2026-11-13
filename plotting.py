from numba import njit
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k1
import time


vel = [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.024083753856592685), np.float64(0.04405733046374334), np.float64(0.059722638864333676), np.float64(0.0736202501089282), np.float64(0.08655470752980308)]
F_d_0 = np.linspace(0.0,0.1,10)

plt.plot(F_d_0, vel,label='Original Lattice')
plt.ylabel("Particle Velocity")
plt.xlabel("Driving Force Strength")
plt.title("Driving Force vs Velocity")
plt.legend()
plt.show()