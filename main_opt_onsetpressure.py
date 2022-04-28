"""
Solve a simple test optimization problem
"""

import os.path as path
from pprint import pprint

import h5py
from scipy import optimize
from blockarray import blockvec  as bvec

import libhopf
import libfunctionals as libfuncs
from test_libhopf import setup_hopf_state


if __name__ == '__main__':
    # Load the Hopf system
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('./mesh', mesh_name+'.xml')

    hopf, xhopf, props0 = setup_hopf_state(mesh_path)

    # Load the measurement glottal width error + reduced gradient
    func_onset_pressure = libfuncs.OnsetPressureFunctional(hopf)
    func_onset_frequency = libfuncs.AbsOnsetFrequencyFunctional(hopf)
    func = func_onset_pressure + 1000.0*(func_onset_frequency-float(abs(xhopf['omega'][0])))**2

    redu_grad = libhopf.ReducedGradient(func, hopf)

    # Create the objective function and gradient needed for optimization
    x0 = bvec.concatenate_vec([redu_grad.props.copy(), redu_grad.camp.copy()])
    x0['amp'].set(1.0)

    opt_options = {
        'disp': 1,
        'maxiter': 100
    }
    def opt_callback(xk):
        print("In callback")

    with h5py.File("out/opt_hist.h5", mode='w') as f:
        grad_manager = libhopf.OptGradManager(redu_grad, f)
        opt_res = optimize.minimize(
            grad_manager.grad, x0.to_mono_ndarray(),
            method='L-BFGS-B',
            jac=True,
            options=opt_options,
            callback=opt_callback
            )

    pprint(opt_res)

