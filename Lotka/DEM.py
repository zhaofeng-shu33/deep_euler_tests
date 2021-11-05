import numpy as np
import torch

from ivp_enhanced.rk import RungeKuttaAdaptive

from model import MLPs

class DeepEuler(RungeKuttaAdaptive):
    n_stages = 1
    order = 1

    def __init__(self, fun, t0, y0, t_bound, step=None, vectorized=False,
                 model_file=None, disable_residue=False, theta=None, **extraneous):
        super().__init__(fun, t0, y0, t_bound, vectorized=vectorized, step=step, adaptive=False, **extraneous)
        # load the residue model
        if not disable_residue:
            state_dic = torch.load(model_file)['model_state_dict']
            input_dim = state_dic['l_in.weight'].shape[1]
            if state_dic.get('l4.weight') is not None:
                self.residue_model = MLPs.SimpleMLP(input_dim, len(y0), 80)
                self.is_embedded = False
            else:
                self.residue_model = MLPs.SimpleMLPGen(4, len(y0), 80, input_dim)
                self.is_embedded = True

            self.residue_model.load_state_dict(state_dic)
            if input_dim > 5:
                self.u_0= y0 # save y0 used for generalized NN model
                self.theta = theta # should be a list for generalized NN model
                self.generalized = True
            else:
                self.generalized = False
        self.disable_residue = disable_residue

    def residue_predict(self, t, y, h):
        if self.generalized:
            concatenated_tensor = torch.from_numpy(np.hstack(([t, t + h], y, self.theta, self.u_0)))
        else:
            concatenated_tensor = torch.from_numpy(np.hstack(([t, t + h], y)))
        if self.is_embedded:
            concatenated_tensor = concatenated_tensor.reshape((1, -1))
        output = self.residue_model(concatenated_tensor.float()).detach().numpy()
        if self.is_embedded:
            output = output.reshape(-1)
        return output

    def _step_impl(self):
        t = self.t
        y = self.y

        h_abs = self.h_abs


        step_accepted = False

        while not step_accepted:
            step_accepted = True
            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)
            if self.disable_residue:
                y_new = y + h * self.fun(t, y)
            else:
                y_new = y + h * self.fun(t, y) + h ** 2 * self.residue_predict(t, y, h)
            f_new = self.fun(t + h, y_new)
        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True, None