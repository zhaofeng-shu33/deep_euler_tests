import numpy as np
from scipy.integrate import solve_ivp

from utils.data_utils import l2_error
from utility import lotka_old
from DEM import DeepEuler
end = 25

def test_euler_lotka():
    sol_euler = solve_ivp(lotka_old, [0, end], [2.0, 1.0], method=DeepEuler, step=0.1, disable_residue=True)
    sol = solve_ivp(lotka_old, [0, end], [2.0, 1.0], rtol=1e-6, atol=1e-6, dense_output=True)
    print('euler err', l2_error(sol, sol_euler))    
    
def test_hyper_euler_lotka():
    sol_dem = solve_ivp(lotka_old, [0, end], [2.0, 1.0], method=DeepEuler, step=0.1, model_file='training/model_e10_2021_11_04.pt')
    sol = solve_ivp(lotka_old, [0, end], [2.0, 1.0], rtol=1e-6, atol=1e-6, dense_output=True)
    print('dem err', l2_error(sol, sol_dem))

def test_hyper_euler_generalized_lotka():
    sol_dem_gen = solve_ivp(lotka_old, [0, end], [2.0, 1.0], method=DeepEuler, theta=[1.0, 1, 1, 1],
                    step=0.1, model_file='training/range_model_e20_2021_11_04.pt')
    sol = solve_ivp(lotka_old, [0, end], [2.0, 1.0], rtol=1e-6, atol=1e-6, dense_output=True)
    print('dem generalized err', l2_error(sol, sol_dem_gen))

def test_hyper_euler_generalized_embedded_lotka():
    sol_dem_gen = solve_ivp(lotka_old, [0, end], [2.0, 1.0], method=DeepEuler, theta=[1.0, 1, 1, 1],
                    step=0.1, model_file='training/range_embedded_model_e22_2021_11_10.pt')
    sol = solve_ivp(lotka_old, [0, end], [2.0, 1.0], rtol=1e-6, atol=1e-6, dense_output=True)
    print('dem generalized embedded err', l2_error(sol, sol_dem_gen))

if __name__ == '__main__':
    test_euler_lotka()
    test_hyper_euler_lotka()
    test_hyper_euler_generalized_lotka()
    test_hyper_euler_generalized_embedded_lotka()