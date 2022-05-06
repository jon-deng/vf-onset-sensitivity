"""
Solve a simple test optimization problem
"""

import sys
import os.path as path
from pprint import pprint

import h5py
import numpy as np
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
    func_egrad_norm = libfuncs.ModulusGradientNormSqr(hopf)

    # `alpha` is a weight on the gradient smoothing of modulus
    # A characteristic gradient of 1kPa/1cm over a volume of 1cm^2 has a
    # functional value of 1e8 Ba^2*cm^2 (cgs unit)
    # 'unit' value of alpha should then start around 1e-8
    for alpha in 10**np.array([-np.inf]+[-10, -8, -6, -5.5, -5.0, -4.5, -4, -2, 0]):
        func = (
            func_onset_pressure
            + 1000.0*(func_onset_frequency-float(abs(xhopf['omega'][0])))**2
            + alpha*func_egrad_norm
        )

        redu_grad = libhopf.ReducedGradient(func, hopf)

        # Create the objective function and gradient needed for optimization
        x0 = bvec.concatenate_vec([props0.copy(), redu_grad.camp.copy()])
        x0[-2:] = 0.0 # these correspond to complex amp. parameters

        ## Optimization
        opt_options = {
            'disp': 99,
            'maxiter': 50,
            'ftol': 0.0,
            'maxls': 100
        }
        def opt_callback(xk):
            print("In callback")

        fpath = f"out/optimize_onset_pressure/opt_hist_alpha{alpha:.2e}.h5"
        if not path.isfile(fpath):
            with h5py.File(fpath, mode='w') as f:
                grad_manager = libhopf.OptGradManager(redu_grad, f)
                # opt_res = optimize.minimize(
                #     grad_manager.grad, x0.to_mono_ndarray(),
                #     method='L-BFGS-B',
                #     jac=True,
                #     options=opt_options,
                #     callback=opt_callback
                # )

                opt_res = optimize.fmin_l_bfgs_b(
                    grad_manager.grad, x0.to_mono_ndarray(),
                    factr=0.0, iprint=99
                )

            pprint(opt_res)
        else:
            print(f"File {fpath} already exists.")
