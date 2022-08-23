"""
Testing code for finding hopf bifurcations of coupled FE VF models
"""
# import sys
from os import path
import warnings
import h5py
import numpy as np

from blockarray import linalg as bla, blockvec as bvec

import libhopf
import libfunctionals as libfuncs
from libsetup import load_hopf_model, set_default_props
from test_hopf import _test_taylor

# pylint: disable=redefined-outer-name
# pylint: disable=no-member

def test_solve_hopf_newton(hopf, xhopf_0):
    xhopf, info = libhopf.solve_hopf_newton(hopf, xhopf_0)

def test_solve_reduced_gradient(func, hopf, props_0, dprops, xhopf_0):
    """
    Test the solve_reduced_gradient function

    Parameters
    ----------
    func :
        The functional to compute the gradient of
    hopf :
        The Hopf model
    props0, dprops :
        The initial properties and increment in properties
    xhopf0 :
        An initial guess for solving Hopf bifurcations. Solving Hopf bifurcations is implictly
        needed in computing the reduced gradient through finite differences
    """

    def res(props):
        hopf.set_props(props)
        x, info = libhopf.solve_hopf_newton(hopf, xhopf_0)

        func.set_state(x)
        func.set_props(props)
        return func.assem_g()

    def jac(props):
        hopf.set_props(props)
        x, info = libhopf.solve_hopf_newton(hopf, xhopf_0)

        for xx in (func, hopf):
            xx.set_state(x)
            xx.set_props(props)

        return libhopf.solve_reduced_gradient(func, hopf)

    _test_taylor(props_0, dprops, res, jac, action=bla.dot, norm=lambda x: x)

def test_ReducedGradient(redu_grad, props_list):
    """
    Test the ReducedGradientManager object
    """

    hopf = redu_grad.res
    for props in props_list:
        # For each property in a list of properties to test, set the properties
        # of the ReducedGradient; the ReducedGradient should handle solving the
        # Hopf system implictly
        redu_grad.set_props(props)
        # print(redu_grad.assem_g())

        # Next, check that the Hopf system was correctly solved in
        # ReducedGradient by checking the Hopf residual
        hopf.set_state(redu_grad.hist_state[-1])
        hopf.set_props(redu_grad.hist_props[-1])
        print(bla.norm(hopf.assem_res()))

def test_OptGradManager(redu_grad, props_list):
    with h5py.File("out/_test_make_opt_grad.h5", mode='w') as f:
        grad_manager = libhopf.OptGradManager(redu_grad, f)

        for props in props_list:
            print(grad_manager.grad(props.to_mono_ndarray()))

        print(f.keys())

        for key in list(f.keys()):
            if 'hopf_newton_' in key:
                print(f"{key}: {f[key][:]}")

def test_bound_hopf_bifurcation(hopf, bound_pairs):
    bounds, omegas = libhopf.bound_hopf_bifurcations(hopf, bound_pairs)
    print(f"Hopf bifurcations between {bounds[0]} and {bounds[1]}")
    print(f"with growth rates between {omegas[0]} and {omegas[1]}")

def test_gen_hopf_initial_guess_from_bounds(hopf, bound_pairs):
    eref = hopf.res.state.copy()
    eref[:] = 1.0

    xhopf_0 = libhopf.gen_hopf_initial_guess_from_bounds(hopf, bound_pairs)

    xhopf_n, info = libhopf.solve_hopf_newton(hopf, xhopf_0)
    print(f"Solved Hopf system from automatic initial guess with info {info}")


if __name__ == '__main__':
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('./mesh', mesh_name+'.msh')
    hopf, res, dres = load_hopf_model(mesh_path, sep_method='smoothmin', sep_vert_label='separation')

    xhopf = hopf.state.copy()
    props0 = hopf.props.copy()
    set_default_props(props0, res.solid.forms['mesh.mesh'])

    func = libfuncs.OnsetPressureFunctional(hopf)

    # Create the ReducedGradientManager object;
    # currently this required the Hopf system have state and properties that
    # initially satisfy the equations
    xhopf_0 = xhopf.copy()
    xhopf_0[:] = 0.0
    hopf.set_state(xhopf_0)
    hopf.set_props(props0)
    redu_grad = libhopf.ReducedGradient(func, hopf)

    dprops = props0.copy()
    dprops[:] = 0
    dprops['emod'] = 1.0

    with warnings.catch_warnings():
        # warnings.filterwarnings('error', category=UserWarning)

        test_solve_hopf_newton(hopf, xhopf)

        lbs = [400.0*10]
        ubs = [600.0*10]
        test_bound_hopf_bifurcation(hopf.res, (lbs, ubs))

        test_gen_hopf_initial_guess_from_bounds(hopf, (lbs, ubs))

        test_solve_reduced_gradient(func, hopf, props0, dprops, xhopf)

        props_list = [props0 + alpha*dprops for alpha in np.arange(0, 1000, 100)]
        test_ReducedGradient(redu_grad, props_list)

        camp = func.camp.copy()
        props_list = [
            bvec.concatenate_vec([props0 + alpha*dprops, camp])
            for alpha in np.linspace(0, 100, 3)
            ]
        test_OptGradManager(redu_grad, props_list)
