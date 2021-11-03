#!/usr/bin/env python
# coding: utf-8
# generating data from a range of problems


import argparse
import numpy as np
import scipy.integrate
import h5py

def euler_truncation_error(arr, output_size): 
    #t0 x1 x2 x3 z1 ... z8 dx1 dx2 dx3 dz1 ... dz8
    #0   1  2  3 4       11 12  13  14  15      22
    dt = arr[1:,0] - arr[:-1,0]
    X = np.column_stack((arr[1:,0], arr[:-1,:1+output_size])) #t1 t0 x1(0) x2(0) x3(0) z(0)
    dt_m = np.copy(dt)
    for _ in range(1, output_size):
        dt_m = np.column_stack((dt_m, dt))
    Y = np.reciprocal(dt_m * dt_m) * (arr[1:,1:output_size+1] - arr[:-1,1:output_size+1] - dt_m * arr[:-1, output_size+1:])
    return X, Y

class LotkaVolterraProblem:
    def __init__(self):
        super().__init__()
        self.theta_range = [[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.5, 1.5]]
        U_MAX = 2
        self.u_0_range = [[1.5, 2.5], [0.8, U_MAX]]
        # self.h_log_range = [-5, -1]
        self.t_max_range = [1.0, 15]
        self.input_dim = 8
        self.u_0 = [2.0, 1.0] # used for plot
        self.t_max = 10.0 # used for plot
        self.theta = [1.0, 1, 1, 1]

    @staticmethod
    def ode_func(t, y, u_0, theta):
        # theta: parameter of the ODE function
        return [y[0] * (theta[0] - theta[1] * y[1]), - y[1] * (theta[2] - theta[3] * y[0])]

    def get_parameter_range(self):
        parameter_range = []
        parameter_range.extend(self.theta_range)
        parameter_range.extend(self.u_0_range)
        parameter_range.append(self.t_max_range)
        # parameter_range.append(self.h_log_range)
        return parameter_range

    def setup(self):
        if not hasattr(self, 'u_0_dim'):
            self.parameter_range = self.get_parameter_range()
            self.theta_dim = len(self.theta_range)
            self.u_0_dim = len(self.u_0_range)

    @staticmethod
    def get_train_data(theta, t_max, u_0, eval_num=1000):
        end = t_max
        output_size = 2
        np.random.seed = 42
        t = np.random.rand(eval_num + 1) * end
        t = np.sort(t)
        lotka_embedded = lambda t, y: LotkaVolterraProblem.ode_func(t, y, u_0, theta)

        sol = scipy.integrate.solve_ivp(lotka_embedded, [0, end], u_0, t_eval=t, rtol=1e-6, atol=1e-9)

        dydt = np.array(lotka_embedded(t, sol.y)) # needed to compute the residue error

        # dt = False # whether to use absolute time or time steps

        arr = np.column_stack((t, np.array(sol.y).T, dydt.T))
        x, y = euler_truncation_error(arr, output_size)

        return (x, y)

    def get_batch(self, size=64, eval_num=1000, verbose=False):
        '''
        size: number of different problems to consider
        eval_num: number of sampling time points for a single problem
        method: ODE solver to obtain error given the parameters
        '''
        self.setup()
        # generate uniform random samples
        para_array = np.zeros([len(self.parameter_range), size])
        for i, para in enumerate(self.parameter_range):
            para_array[i, :] = np.random.uniform(para[0], para[1], size)
        output_size = len(self.u_0)
        batch_x = np.zeros([size * eval_num, len(self.parameter_range) + output_size + 1])
        batch_y = np.zeros([size * eval_num, output_size]) # pre-allocation of the space
        j = 0
        while j < size:
            if verbose and j % 100 == 0:
                print(f"generate process: [{j:>5d}/{size:>5d}]")
            theta = para_array[0:self.theta_dim, j]
            u_0 = para_array[self.theta_dim:(self.theta_dim + self.u_0_dim), j]
            t_max = para_array[self.theta_dim + self.u_0_dim, j]
            _x, _y = self.get_train_data(theta, t_max, u_0, eval_num=eval_num)
            batch_x[j * eval_num : (j + 1) * eval_num, :(output_size + 2)] = _x
            # expand the columns of _x using para_array[:, j] (broadcasting)
            batch_x[j * eval_num : (j + 1) * eval_num, (output_size + 2):] = para_array[:-1, j]
            batch_y[j * eval_num : (j + 1) * eval_num, :] = _y
            j += 1
        return batch_x.astype(np.float32), batch_y.astype(np.float32)

def save_train_data(path_to_hdf, x, y):
    with h5py.File(path_to_hdf, 'w') as f:
        f.create_dataset(
            str('lotka_X'),
            x.shape,
            dtype   = np.float64,
            compression     = 'gzip',
            compression_opts= 6
            )
        f.create_dataset(
            str('lotka_Y'),
            y.shape,
            dtype   = np.float64,
            compression     = 'gzip',
            compression_opts= 6
            )
        f['lotka_X'][:, :] = x
        f['lotka_Y'][:, :] = y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--eval_num', type=int, default=1000)
    args = parser.parse_args()
    path_to_hdf = f'lotka_range_data_{args.size}_{args.eval_num}.hdf5'
    problem = LotkaVolterraProblem()
    x, y = problem.get_batch(verbose=True, size=args.size, eval_num=args.eval_num)
    save_train_data(path_to_hdf, x, y)
