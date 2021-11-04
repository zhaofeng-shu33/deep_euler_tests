# This demo verifies the performance boosting of deep Euler over common Euler

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from DEM import DeepEuler
from utils.data_utils import l2_error
from utility import lotka_old


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['time_series', 'work_precision'], default='work_precision')
    parser.add_argument('--generalized', default=False, const=True, nargs='?')
    args = parser.parse_args()
    if args.action == 'time_series':
        end_interval = 15
        if args.generalized:
            sol_dem = solve_ivp(lotka_old, [0, end_interval], [2.0, 1.0], step=0.1, method=DeepEuler, theta=[1.0, 1, 1, 1], model_file='training/range_model_e20_2021_11_04.pt')
            title = 'DEM generalized'
        else:
            sol_dem = solve_ivp(lotka_old, [0, end_interval], [2.0, 1.0], step=0.1, method=DeepEuler, model_file='training/model_e10_2021_11_04.pt')
            title = 'DEM original'
        sol = solve_ivp(lotka_old, [0, end_interval], [2.0, 1.0], rtol=1e-6, atol=1e-6, dense_output=True)
        plt.plot(sol.t, sol.y[0, :], label='true y0')
        plt.plot(sol.t, sol.y[1, :], label='true y1')
        plt.plot(sol_dem.t, sol_dem.y[0, :], label='dem y0')
        plt.plot(sol_dem.t, sol_dem.y[1, :], label='dem y1')
        plt.legend()
        plt.xlabel('t')
        plt.title(title)
        plt.show()
    else:
        end_interval = 25
        dem_step_list = [0.1, 0.05, 0.01, 0.005]
        euler_step_list = [0.002, 0.001, 0.0005, 0.0002]
        sol = solve_ivp(lotka_old, [0, end_interval], [2.0, 1.0], rtol=1e-6, atol=1e-6, dense_output=True)
        euler_nfev = []
        euler_error = []
        dem_error = []
        dem_nfev = []
        for i in range(4):
            sol_euler = solve_ivp(lotka_old, [0, end_interval], [2.0, 1.0], step=euler_step_list[i], method=DeepEuler, disable_residue=True)
            sol_dem = solve_ivp(lotka_old, [0, end_interval], [2.0, 1.0], step=dem_step_list[i], method=DeepEuler, model_file='training/model_e10_2021_11_04.pt')
            euler_nfev.append(sol_euler.nfev)
            euler_error.append(l2_error(sol, sol_euler))
            dem_error.append(l2_error(sol, sol_dem))
            dem_nfev.append(10 * sol_dem.nfev)


        # plt.figure(num="Comparison")
        # plt.plot(sol.t,sol.y[0,:], lw=0.5, label="Dopri")
        plt.scatter(euler_error, np.log(euler_nfev), label="Euler")
        plt.scatter(dem_error,  np.log(dem_nfev), label="DEM")
        # plt.xlabel("t")
        plt.ylabel("log(nfev)")
        plt.legend()
        plt.show()


