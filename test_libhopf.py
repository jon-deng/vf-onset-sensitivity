"""
Test the `libhopf` module
"""
# import sys
from os import path
import warnings
import pytest
import h5py
import numpy as np

from blockarray import linalg as bla, blockvec as bv

import libhopf
import libfunctionals as libfuncs
import libsetup
from test_hopf import taylor_convergence

# pylint: disable=redefined-outer-name
# pylint: disable=no-member

@pytest.fixture()
def setup_hopf_model():
    """Return a Hopf bifurcation model"""
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('./mesh', mesh_name+'.msh')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        hopf, res, dres = libsetup.load_hopf_model(
            mesh_path,
            sep_method='smoothmin', sep_vert_label='separation'
        )
    return hopf

@pytest.fixture()
def setup_props(setup_hopf_model):
    """Return Hopf model properties"""
    hopf = setup_hopf_model

    props = hopf.props.copy()
    libsetup.set_default_props(props, hopf.res.solid.forms['mesh.mesh'])
    return props

class TestHopfModel:
    """
    Test correctness and functionality of the `HopfModel` class
    """
    @pytest.fixture()
    def setup_state_linearization(self, setup_hopf_model):
        hopf = setup_hopf_model
        state = hopf.state.copy()
        state[:] = 0

        dstate = state.copy()
        dstate['u'] = 1e-5

        hopf.apply_dirichlet_bvec(dstate)
        return (state, dstate)

    def test_assem_dres_dstate(
            self,
            setup_hopf_model,
            setup_props,
            setup_state_linearization
        ):

        hopf = setup_hopf_model
        props = setup_props
        state0, dstate = setup_state_linearization

        def hopf_res(x):
            hopf.set_state(x)
            res = hopf.assem_res()
            hopf.apply_dirichlet_bvec(res)
            return hopf.assem_res()

        def hopf_jac(x):
            hopf.set_state(x)
            dres_dstate = hopf.assem_dres_dstate()
            hopf.apply_dirichlet_bmat(dres_dstate)
            return dres_dstate

        taylor_convergence(state0, dstate, hopf_res, hopf_jac)

class TestHopf:
    @pytest.fixture()
    def setup_bound_pairs(self):
        """Return lower/upper subglottal pressure bounds"""
        lbs = [400.0*10]
        ubs = [600.0*10]
        return (lbs, ubs)

    def test_bound_hopf_bifurcations(self, setup_hopf_model, setup_props, setup_bound_pairs):
        """Test `bound_hopf_bifurcations`"""
        hopf = setup_hopf_model
        bound_pairs = setup_bound_pairs
        props = setup_props
        hopf.set_props(props)

        bounds, omegas = libhopf.bound_hopf_bifurcations(hopf.res, bound_pairs)
        print(f"Hopf bifurcations between {bounds[0]} and {bounds[1]}")
        print(f"with growth rates between {omegas[0]} and {omegas[1]}")

    def test_gen_hopf_initial_guess_from_bounds(self, setup_hopf_model, setup_props, setup_bound_pairs):
        """Test `gen_hopf_initial_guess_from_bounds`"""
        hopf = setup_hopf_model
        bound_pairs = setup_bound_pairs
        props = setup_props
        hopf.set_props(props)

        eref = hopf.res.state.copy()
        eref[:] = 1.0

        xhopf_0 = libhopf.gen_hopf_initial_guess_from_bounds(hopf, bound_pairs)

        xhopf_n, info = libhopf.solve_hopf_newton(hopf, xhopf_0)
        print(f"Solved Hopf system from automatic initial guess with info {info}")

    @pytest.fixture()
    def setup_xhopf_0(self, setup_hopf_model, setup_props, setup_bound_pairs):
        """Return an initial guess for the hopf state"""
        hopf = setup_hopf_model
        bound_pairs = setup_bound_pairs
        props = setup_props
        hopf.set_props(props)

        xhopf_0 = libhopf.gen_hopf_initial_guess_from_bounds(hopf, bound_pairs)
        return xhopf_0

    def test_solve_hopf_newton(self, setup_hopf_model, setup_props, setup_xhopf_0):
        xhopf_0 = setup_xhopf_0
        hopf = setup_hopf_model
        props = setup_props
        hopf.set_props(props)

        xhopf, info = libhopf.solve_hopf_newton(hopf, xhopf_0)

class TestFunctionalGradient:

    @pytest.fixture()
    def setup_linearization(self, setup_props, setup_hopf_model):
        """
        Return a linearization point and direction for a Hopf model

        Returns
        -------
        xhopf, props, dprops :
            `xhopf` - the state corresponding to the Hopf bifurcation
            to `props`
            `props` - the Hopf model properties
            `dprops` - the Hopf model properties perturbation
        """
        hopf = setup_hopf_model
        props = setup_props

        # Create the ReducedGradientManager object;
        # currently this required the Hopf system have state and properties that
        # initially satisfy the equations
        hopf.set_props(props)
        psubs = np.arange(100, 1000, 100)*10
        xhopf_0 = libhopf.gen_hopf_initial_guess(hopf, psubs)
        xhopf, info = libhopf.solve_hopf_newton(hopf, xhopf_0)

        hopf.set_state(xhopf_0)

        dprops = props.copy()
        dprops[:] = 0
        dprops['emod'] = 1.0
        return xhopf, props, dprops

    @pytest.fixture()
    def setup_func(self, setup_hopf_model):
        """Return a Hopf model functional"""
        hopf = setup_hopf_model
        return libfuncs.OnsetPressureFunctional(hopf)

    def test_solve_reduced_gradient(self, setup_func, setup_hopf_model, setup_linearization):
        """
        Test `solve_reduced_gradient`

        Parameters
        ----------
        func :
            The functional to compute the gradient of
        hopf :
            The Hopf model
        setup_linearization: (xhopf, props, dprops)
        """
        hopf = setup_hopf_model
        func = setup_func
        xhopf, props, dprops = setup_linearization

        def res(props):
            hopf.set_props(props)
            x, info = libhopf.solve_hopf_newton(hopf, xhopf)

            func.set_state(x)
            func.set_props(props)
            return func.assem_g()

        def jac(props):
            hopf.set_props(props)
            x, info = libhopf.solve_hopf_newton(hopf, xhopf)

            for model in (func, hopf):
                model.set_state(x)
                model.set_props(props)

            return libhopf.solve_reduced_gradient(func, hopf)

        taylor_convergence(
            props, dprops, res, jac,
            action=bla.dot, norm=lambda x: x
        )


    @pytest.fixture()
    def setup_redu_grad(self, setup_func, setup_hopf_model):
        """Return a `ReducedGradient` instance"""
        func = setup_func
        hopf = setup_hopf_model
        return libhopf.ReducedGradient(func, hopf)

    @pytest.fixture()
    def setup_rg_propss(self, setup_func, setup_linearization):
        """Return an iterable of `Hopf.props` vectors"""
        _, props, dprops = setup_linearization
        func = setup_func

        # camp = func.camp.copy()
        propss = [
            bv.concatenate_vec([props + alpha*dprops])
            for alpha in np.linspace(0, 100, 3)
        ]
        return propss

    def test_ReducedGradient(self, setup_redu_grad, setup_rg_propss):
        """
        Test the ReducedGradientManager object
        """
        redu_grad = setup_redu_grad
        propss = setup_rg_propss

        hopf = redu_grad.res
        for props in propss:
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

    @pytest.fixture()
    def setup_og_propss(self, setup_func, setup_linearization):
        """Return an iterable of `Hopf.props` + camp vectors"""
        _, props, dprops = setup_linearization
        func = setup_func

        camp = func.camp.copy()
        propss = [
            bv.concatenate_vec([props + alpha*dprops, camp])
            for alpha in np.linspace(0, 100, 3)
        ]
        return propss

    def test_OptGradManager(self, setup_redu_grad, setup_og_propss):
        """
        Test the ReducedGradientManager object
        """
        redu_grad = setup_redu_grad
        propss = setup_og_propss

        with h5py.File("out/_test_make_opt_grad.h5", mode='w') as f:
            grad_manager = libhopf.OptGradManager(redu_grad, f)

            for props in propss:
                print(grad_manager.grad(props.to_mono_ndarray()))

            print(f.keys())

            for key in list(f.keys()):
                if 'hopf_newton_' in key:
                    print(f"{key}: {f[key][:]}")
