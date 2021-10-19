# This demo verifies the performance boosting of deep Euler over common Euler
# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import h5py
from DEM import DeepEuler



# %%
def lotka(t, x):
    y = np.empty(x.shape)
    y[0] =  x[0] - x[0]*x[1]
    y[1] = -x[1] + x[0]*x[1]
    return y

sol_euler = scipy.integrate.solve_ivp(lotka, [0, 50], [2.0, 1.0], step=0.1, method=DeepEuler, disable_residue=True)
sol_dem = scipy.integrate.solve_ivp(lotka, [0, 50], [2.0, 1.0], step=0.1, method=DeepEuler, model_file='training/model_e1_2110191109.pt')
sol = scipy.integrate.solve_ivp(lotka, [0, 50], [2.0, 1.0], rtol=1e-6, atol=1e-6)





# %%
plt.figure(num="Comparison")
plt.plot(sol.t,sol.y[0,:], lw=0.5, label="Dopri")
plt.plot(sol_euler.t, sol_euler.y[0, :], lw=0.5, label="Euler")
plt.plot(sol_dem.t,sol_dem.y[0, :], lw=0.5, label="DEM")
plt.xlabel("t")
plt.ylabel("x1")
plt.legend()
plt.show()


