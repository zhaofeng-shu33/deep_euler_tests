# This demo verifies the performance boosting of deep Euler over common Euler

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from DEM import DeepEuler
from utility import lotka_old

def l2_error(true_sol, sol_data):
    t_list = sol_data[:, 0]
    y_true = true_sol.sol(t_list)
    return np.sqrt(np.average(np.linalg.norm(sol_data[:, 1:] - y_true.T, axis=1) ** 2 / 2))

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
    else: # time series
        sol = solve_ivp(lotka_old, [0, 25], [2.0, 1.0], rtol=1e-6, atol=1e-6, dense_output=True)

        base_load_dir = 'build'
        time_array = np.loadtxt(f'{base_load_dir}/clock.txt')
        time_array_generalized = np.loadtxt(f'{base_load_dir}/clock_generalized.txt')
        time_array_embedded = np.loadtxt(f'{base_load_dir}/clock_generalized.txt')
        total_index = 4

        label_list = ['dem', 'dem generalized', 'dem embedded']
        file_suffix_list = ['.txt', '_generalized.txt', '_embedded.txt']
        for j in range(3):
            t_list_dem = []
            error_list_dem = []
            t_list_euler = []
            error_list_euler = []
            file_suffix = file_suffix_list[j]
            clock_file_name = f'{base_load_dir}/clock{file_suffix}'
            time_array = np.loadtxt(clock_file_name)
            for i in range(total_index):
                dem = np.loadtxt(f'{base_load_dir}/lotka_dem{i}{file_suffix}')
                euler = np.loadtxt(f'{base_load_dir}/lotka_euler{i}.txt')
                error_dem = l2_error(sol, dem)
                error_euler = l2_error(sol, euler)
                time_dem = time_array[i, 0]
                time_euler = time_array[i, 1]
                error_list_dem.append(error_dem)
                error_list_euler.append(error_euler)
                t_list_dem.append(time_dem)
                t_list_euler.append(time_euler)
            label_ = label_list[j]
            plt.figure(j)
            plt.scatter(error_list_dem, np.log(t_list_dem), label=label_)
            plt.scatter(error_list_euler, np.log(t_list_euler), label='euler')
            plt.xlabel('error')
            plt.ylabel('log(time)')
            plt.title(f'{label_} vs euler')
            plt.legend()
            plt.savefig(f'build/euler_comparison_with{file_suffix}.pdf')
    
        plt.show()


