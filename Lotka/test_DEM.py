import numpy as np
from scipy.integrate import solve_ivp

from utility import lotka
from DEM import DeepEuler

def test_hyper_euler_lotka():
    end = 15
    sol = solve_ivp(lotka, [0, end], [2.0, 1.0], method=DeepEuler, step=0.1, model_file='training/model_e1_2110191109.pt')


if __name__ == '__main__':
    test_hyper_euler_lotka()
