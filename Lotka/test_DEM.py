import numpy as np
from scipy.integrate import solve_ivp

from utils.data_utils import l2_error
from utility import lotka_old
from DEM import DeepEuler

def test_hyper_euler_lotka():
    end = 15
    sol = solve_ivp(lotka_old, [0, end], [2.0, 1.0], method=DeepEuler, step=0.1, model_file='training/model_e1_2110191109.pt')

def test_hyper_euler_generalized_lotka():
    end = 15
    sol = solve_ivp(lotka_old, [0, end], [2.0, 1.0], method=DeepEuler, theta=[1.0, 1, 1, 1],
                    step=0.1, model_file='training/range_model_e110_2111032041.pt')

if __name__ == '__main__':
    test_hyper_euler_generalized_lotka()
