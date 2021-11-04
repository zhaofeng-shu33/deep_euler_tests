# This demo verifies the performance boosting of deep Euler over common Euler
# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import h5py
from DEM import DeepEuler
from utils.data_utils import l2_error


# %%
def lotka(t, x):
    y = np.empty(x.shape)
    y[0] =  x[0] - x[0]*x[1]
    y[1] = -x[1] + x[0]*x[1]
    return y

end_interval = 25
dem_step_list = [0.1, 0.05, 0.01, 0.005]
euler_step_list = [0.002, 0.001, 0.0005, 0.0002]
sol = scipy.integrate.solve_ivp(lotka, [0, end_interval], [2.0, 1.0], rtol=1e-6, atol=1e-6, dense_output=True)
euler_nfev = []
euler_error = []
dem_error = []
dem_nfev = []
for i in range(4):
    sol_euler = scipy.integrate.solve_ivp(lotka, [0, end_interval], [2.0, 1.0], step=euler_step_list[i], method=DeepEuler, disable_residue=True)
    sol_dem = scipy.integrate.solve_ivp(lotka, [0, end_interval], [2.0, 1.0], step=dem_step_list[i], method=DeepEuler, model_file='training/model_e20_2110201533.pt')
    euler_nfev.append(sol_euler.nfev)
    euler_error.append(l2_error(sol, sol_euler))
    dem_error.append(l2_error(sol, sol_dem))
    dem_nfev.append(10 * sol_dem.nfev)


# %%
# plt.figure(num="Comparison")
# plt.plot(sol.t,sol.y[0,:], lw=0.5, label="Dopri")
plt.scatter(euler_error, np.log(euler_nfev), label="Euler")
plt.scatter(dem_error,  np.log(dem_nfev), label="DEM")
# plt.xlabel("t")
plt.ylabel("log(nfev)")
plt.legend()
plt.show()


