import numpy as np
import torch

from ivp_enhanced.rk import RungeKuttaAdaptive

from model import MLPs

class DeepEuler(RungeKuttaAdaptive):
    n_stages = 1
    order = 1

    def __init__(self, fun, t0, y0, t_bound, step=None, vectorized=False, model_file=None, disable_residue=False, **extraneous):
        super().__init__(fun, t0, y0, t_bound, vectorized=vectorized, step=step, adaptive=False, **extraneous)
        # load the residue model
        if not disable_residue:
            self.residue_model = MLPs.SimpleMLP(2 + len(y0), len(y0), 80)
            state_dic = torch.load(model_file)['model_state_dict']
            self.residue_model.load_state_dict(state_dic)
        self.disable_residue = disable_residue

    def residue_predict(self, t, y, h):
        concatenated_tensor = torch.from_numpy(np.hstack(([t, t + h], y)))
        return self.residue_model(concatenated_tensor.float()).detach().numpy()

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