#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import h5py

from utility import lotka


def euler_truncation_error(arr, output_size): 
    #t0 x1 x2 x3 z1 ... z8 dx1 dx2 dx3 dz1 ... dz8
    #0   1  2  3 4       11 12  13  14  15      22
    dt = arr[1:,0] - arr[:-1,0]
    X = np.column_stack((arr[1:,0], arr[:-1,:1+output_size])) #t1 t0 x1(0) x2(0) x3(0) z(0)
    dt_m = np.copy(dt)
    for _ in range(1, output_size):
        dt_m = np.column_stack((dt_m,dt))
    Y = np.reciprocal(dt_m*dt_m) * (arr[1:,1:output_size+1] - arr[:-1,1:output_size+1] - dt_m * arr[:-1, output_size+1:])
    return X, Y



def get_train_data(t_max=15, eval_num=1000, initial_value=[2, 1], theta=[1.0, 1.0, 1.0, 1.0]):
    end = t_max
    n = eval_num
    np.random.seed = 42
    t = np.random.rand(1000) * end
    t = np.sort(t)
    lotka_embedded = lambda t, y: lotka(t, y, theta)
    sol = scipy.integrate.solve_ivp(lotka_embedded, [0, end], initial_value, t_eval=t, rtol=1e-6, atol=1e-6)

    dydt = lotka_embedded(t, sol.y)

    path_to_hdf = 'lotka_data2.hdf5'
    dt = False # whether to use absolute time or time steps

    arr = np.column_stack((t, np.array(sol.y).T, dydt.T))

    l = arr.shape[0]
    b = 1
    
    sum = 0
    for i in range(b, n):
        sum = sum + l - i - 1

    with h5py.File(path_to_hdf, 'w') as f:
        f.create_dataset(
            str('lotka_X'),
            (sum, 3 if dt else 4),
            dtype   = np.float64,
            compression     = 'gzip',
            compression_opts= 6
            )
        f.create_dataset(
            str('lotka_Y'),
            (sum, 2),
            dtype   = np.float64,
            compression     = 'gzip',
            compression_opts= 6
            )
        begin = 0
        end = l-1
        X = f['lotka_X']
        Y = f['lotka_Y']
        x, y = euler_truncation_error(arr, 2)
        if dt: 
            x = np.column_stack((x[:,0] - x[:,1],x[:,2],x[:,3]))
        X[begin:end,:] = x
        Y[begin:end,:] = y
        for i in range(b + 1, n):
            for j in range(i):
                x,y = euler_truncation_error(arr[j::i, :], 2)
                if dt:
                    x = np.column_stack((x[:,0] - x[:,1],x[:,2],x[:,3]))
                begin = end
                end = begin + x.shape[0]
                X[begin:end, :] = x
                Y[begin:end, :] = y

if __name__ == '__main__':
    get_train_data()
