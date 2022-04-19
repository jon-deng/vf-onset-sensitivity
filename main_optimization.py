"""
Solve a simple test optimization problem
"""

import os.path as path
from pprint import pprint

from scipy import optimize
from blocktensor import vec as bvec

import libhopf
import libfunctionals as libfuncs
from test_libhopf import setup_hopf_state


if __name__ == '__main__':
    # Load the Hopf system
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('./mesh', mesh_name+'.xml')

    hopf, xhopf, props0 = setup_hopf_state(mesh_path)

    # Load the measurement glottal width error + reduced gradient
    # func = libfuncs.OnsetPressureFunctional(hopf)
    func = libfuncs.GlottalWidthErrorFunctional(hopf)

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
    opt_obj, opt_grad = libhopf.make_opt_grad(redu_grad)
    opt_res = optimize.minimize(
        opt_obj, x0.to_ndarray(),
        method='L-BFGS-B',
        jac=opt_grad,
        options=opt_options,
        callback=opt_callback
        )

    pprint(opt_res)

