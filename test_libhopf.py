"""
Test the `libhopf` module
"""
# import sys
from os import path
import warnings
import pytest
import h5py
import numpy as np

from femvf.parameters import parameterization as pazn
from blockarray import linalg as bla, blockvec as bv, subops

import libhopf
import libfunctionals as libfuncs
import libsetup
from test_hopf import taylor_convergence

from petsc4py import PETSc
# pylint: disable=redefined-outer-name
# pylint: disable=no-member, invalid-name

@pytest.fixture()
def hopf_model():
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
def prop(hopf_model):
    """Return Hopf model properties"""
    hopf = hopf_model

    prop = hopf.prop.copy()
    libsetup.set_default_props(prop, hopf.res.solid.forms['mesh.mesh'])
    prop['kcontact'][:] = 1e0
    return prop

class TestHopfModel:
    """
    Test `HopfModel`
    """

    @pytest.fixture()
    def xhopf_props(self, hopf_model, prop):
        """
        Return a linearization point for the Hopf model

        The point is not necessarily a Hopf bifurcation.
        """
        hopf = hopf_model

        # NOTE: Some `state` components can't be zero or Hopf jacobians may have zero rank
        (state_labels,
        mode_real_labels,
        mode_imag_labels,
        psub_labels,
        omega_labels) = hopf.labels_hopf_components

        # Create a pure x-shearing motion to use for displacement/velocities
        y = hopf.res.solid.forms['coeff.state.u1'].function_space().tabulate_dof_coordinates()[1::2, 1]
        ux = 1e-2*(y-y.min())/(y.max()-y.min())
        uy = 0

        state = hopf.state.copy()
        state[:] = 0

        # TODO: Note that some subtle bugs may only appear for certain
        # linearization conditions
        # (for example non zero 'u_mode_real' but 0 'u')
        # so you should probably parameterize this in the future
        disp_labels = ['u', 'v']
        suffixes = ['', '_mode_real', '_mode_imag']
        for label in disp_labels:
            for suffix in suffixes:
                state[label+suffix][:-1:2] = ux
                state[label+suffix][1::2] = uy

        state[mode_real_labels] = 1.0
        state[mode_imag_labels] = 1.0
        PSUB = 100*10
        state[psub_labels] = PSUB
        state[omega_labels] = 1.0
        hopf.apply_dirichlet_bvec(state)

        return (state, prop)

    @pytest.fixture(
        params=[
            'u', 'v',
            'u_mode_real', 'v_mode_real',
            'u_mode_imag', 'v_mode_imag',
            'psub', 'omega'
        ]
    )
    def setup_dstate(self, hopf_model, request):
        """Return a state perturbation"""
        hopf = hopf_model

        dstate = hopf.state.copy()
        dstate[:] = 0

        label = request.param
        print(f"Testing along direction {label}")
        dstate[label] = 1e-4

        hopf.apply_dirichlet_bvec(dstate)
        return dstate

    def test_assem_dres_dstate(
            self,
            hopf_model,
            xhopf_props,
            setup_dstate
        ):
        """Test `HopfModel.assem_dres_dstate`"""

        hopf = hopf_model
        state, prop = xhopf_props
        dstate = setup_dstate
        hopf.set_prop(prop)

        def hopf_res(x):
            hopf.set_state(x)
            res = hopf.assem_res()
            hopf.apply_dirichlet_bvec(res)
            return hopf.assem_res()

        def hopf_jac(x, dx):
            hopf.set_state(x)
            dres_dstate = hopf.assem_dres_dstate()
            hopf.apply_dirichlet_bmat(dres_dstate)
            return bla.mult_mat_vec(dres_dstate, dx)

        taylor_convergence(state, dstate, hopf_res, hopf_jac)

    def test_assem_dres_dstate_adjoint(
            self,
            hopf_model,
            xhopf_props,
            setup_dstate
        ):
        """
        Test the adjoint of `HopfModel.assem_dres_dstate`

        This should be true as long as the tranpose is computed correctly.
        """

        hopf = hopf_model
        state, prop = xhopf_props
        dstate = setup_dstate
        hopf.set_state(state)
        hopf.set_prop(prop)

        dres_dstate = hopf.assem_dres_dstate()
        hopf.apply_dirichlet_bmat(dres_dstate)

        dres_adj = state.copy()
        dres_adj[:] = 0
        dres_adj['psub'] = 1
        # adj_res[:] = 1
        hopf.apply_dirichlet_bvec(dres_adj)

        dres_dstate_adj = hopf.assem_dres_dstate().transpose()
        hopf.apply_dirichlet_bmat(dres_dstate_adj)

        self._test_operator_adjoint(
            dres_dstate, dres_dstate_adj,
            dstate, dres_adj
        )

    def test_assem_dres_dstate_inv(
            self,
            hopf_model,
            xhopf_props,
            setup_dstate
        ):
        """Test `HopfModel.assem_dres_dstate`"""

        hopf = hopf_model
        state, prop = xhopf_props
        dstate = setup_dstate
        hopf.set_state(state)
        hopf.set_prop(prop)

        dres_dstate = hopf.assem_dres_dstate()
        hopf.apply_dirichlet_bmat(dres_dstate)
        hopf.apply_dirichlet_bvec(dstate)
        dres = bla.mult_mat_vec(dres_dstate, dstate)

        dstate_test = dres.copy()
        _dres_dstate = dres_dstate.to_mono_petsc()
        _dstate_test = _dres_dstate.getVecRight()
        subops.solve_petsc_lu(_dres_dstate, dres.to_mono_petsc(), out=_dstate_test)
        dstate_test.set_mono(_dstate_test)

        err = dstate-dstate_test
        print(err.norm())
        assert np.isclose(err.norm(), 0, rtol=1e-8, atol=1e-9)

    @pytest.fixture(
        params=[
            ('emod', 1e2),
            ('rho', 1e-2),
            ('rho_air', 1e-4),
            ('umesh', 1.0e-4)
        ]
    )
    def dprop(self, hopf_model, request):
        """Return a properties perturbation"""
        hopf = hopf_model

        dprop = hopf.prop.copy()
        dprop[:] = 0.0
        label, value = request.param
        dprop[label] = value
        print(f"Testing along direction {label}")
        return dprop

    def test_assem_dres_dprops(
            self,
            hopf_model,
            xhopf_props,
            dprop
        ):
        """Test `HopfModel.assem_dres_dprops`"""

        hopf = hopf_model
        state, prop = xhopf_props
        hopf.set_state(state)

        def hopf_res(x):
            hopf.set_prop(x)
            res = hopf.assem_res()
            hopf.apply_dirichlet_bvec(res)
            return res

        def hopf_jac(x, dx):
            hopf.set_prop(x)
            dres_dprops = hopf.assem_dres_dprops()
            hopf.zero_rows_dirichlet_bmat(dres_dprops)
            return bla.mult_mat_vec(dres_dprops, dx)
        taylor_convergence(prop, dprop, hopf_res, hopf_jac)

    def test_assem_dres_dprops_adjoint(
            self,
            hopf_model,
            xhopf_props,
            dprop
        ):
        """
        Test the adjoint of `HopfModel.assem_dres_dprops`

        This should be true as long as the tranpose is computed correctly.
        """
        hopf = hopf_model
        state, prop = xhopf_props
        hopf.set_state(state)
        hopf.set_prop(prop)

        dres_dprops = hopf.assem_dres_dprops()

        dres_adj = state.copy()
        dres_adj[:] = 1
        # dres_adj['psub'] = 1
        hopf.apply_dirichlet_bvec(dres_adj)

        dres_dprops_adj = dres_dprops.transpose()

        self._test_operator_adjoint(
            dres_dprops, dres_dprops_adj,
            dprop, dres_adj
        )

    def _test_operator_adjoint(self, op, op_adj, dx, dy_adj):
        """
        Test an operator and its adjoint are consistent
        """
        # NOTE: op maps x -> y
        # op_adj maps y_adj -> x_adj

        # Compute the value of a linear functional `adj_res` on a residual vector
        # given a known input `dstate`
        dy = bla.mult_mat_vec(op, dx)
        dx_adj = bla.mult_mat_vec(op_adj, dy_adj)

        print(bv.dot(dx_adj, dx), bv.dot(dy_adj, dy))
        assert np.isclose(
            bv.dot(dx_adj, dx), bv.dot(dy_adj, dy),
            rtol=1e-9, atol=1e-9
        )

class TestHopfUtilities:
    """
    Test utility functions for the Hopf bifurcation system
    """
    @pytest.fixture()
    def bound_pairs(self):
        """Return lower/upper subglottal pressure bounds"""
        lbs = [400.0*10]
        ubs = [600.0*10]
        return (lbs, ubs)

    def test_bound_hopf_bifurcations(self, hopf_model, prop, bound_pairs):
        """Test `bound_hopf_bifurcations`"""
        hopf = hopf_model
        hopf.set_prop(prop)

        dyn_model = hopf.res
        dyn_props = hopf.res.prop
        dyn_control = hopf.res.control

        bounds, omegas = libhopf.bound_ponset(
            dyn_model, dyn_control, dyn_props, bound_pairs
        )
        print(f"Hopf bifurcations between {bounds[0]} and {bounds[1]}")
        print(f"with growth rates between {omegas[0]} and {omegas[1]}")

    def test_gen_hopf_initial_guess_from_bounds(self, hopf_model, prop, bound_pairs):
        """Test `gen_hopf_initial_guess_from_bounds`"""
        hopf = hopf_model

        xhopf_0 = libhopf.gen_xhopf_0_from_bounds(hopf, prop, bound_pairs)

        xhopf_n, info = libhopf.solve_hopf_by_newton(hopf, xhopf_0, prop)
        print(f"Solved Hopf system from automatic initial guess with info {info}")

    @pytest.fixture()
    def setup_xhopf_0(self, hopf_model, prop, bound_pairs):
        """Return an initial guess for the hopf state"""
        hopf = hopf_model

        xhopf_0 = libhopf.gen_xhopf_0_from_bounds(hopf, prop, bound_pairs)
        return xhopf_0

    def test_solve_hopf_newton(self, hopf_model, prop, setup_xhopf_0):
        """Test `solve_hopf_newton`"""
        xhopf_0 = setup_xhopf_0
        hopf = hopf_model

        xhopf, info = libhopf.solve_hopf_by_newton(hopf, xhopf_0, prop)

@pytest.fixture(
    params=[
        pazn.TractionShape,
        pazn.Identity
    ]
)
def parameterization(hopf_model, request):
    """
    Return a parameterization
    """
    model = hopf_model.res
    Param = request.param
    return Param(model)

@pytest.fixture()
def params(hopf_model, parameterization):
    p0 = parameterization.x.copy()
    p0['emod'][:] = 10*1e3*10
    p0['rho'] = 1
    p0['rho_air'] = 1.225e-3
    p0['nu'] = 0.45
    p0['eta'] = 5

    libsetup.set_default_props(p0, hopf_model.res.solid.forms['mesh.mesh'])
    return p0

@pytest.fixture(
    params=[
        ('emod', 1e2),
        ('umesh', 1.0e-4)
    ]
)
def dparams(params, request):
    """Return a `params` perturbation"""
    dparams = params.copy()
    dparams[:] = 0

    key, val = request.param
    if key in dparams:
        dparams[key] = val
    return dparams

def solve_linearization(hopf, prop):
    """
    Return a linearization point corresponding to a Hopf bifurcation

    Returns
    -------
    xhopf, prop :
        `xhopf` - the state corresponding to the Hopf bifurcation
        to `prop`
    `prop` - the Hopf model properties
    """
    hopf.set_prop(prop)
    psubs = np.arange(100, 1000, 100)*10
    xhopf_0 = libhopf.gen_xhopf_0(hopf, prop, psubs)
    xhopf, info = libhopf.solve_hopf_by_newton(hopf, xhopf_0, prop)
    return xhopf, info

@pytest.fixture()
def xhopf_props(hopf_model, prop):
    """
    Return a linearization point corresponding to a Hopf bifurcation
    """
    hopf = hopf_model
    xhopf, info = solve_linearization(hopf, prop)
    return xhopf, prop

@pytest.fixture()
def xhopf_params(hopf_model, params):
    """
    Return a linearization point corresponding to a Hopf bifurcation
    """
    p0, parameterization = params
    hopf = hopf_model
    xhopf, info = solve_linearization(hopf, parameterization.apply(p0))
    return xhopf, p0, parameterization

@pytest.fixture(
    params=[
        ('emod', 1e2),
        ('umesh', 1.0e-4)
    ]
)
def dprops_dir(request):
    return request.param

@pytest.fixture()
def dprop(prop, dprops_dir):
    """Return a `prop` perturbation"""

    dprop = prop.copy()
    dprop[:] = 0

    key, val = dprops_dir
    dprop[key] = val
    return dprop

@pytest.fixture(
    params=[
        libfuncs.StrainEnergyFunctional,
        libfuncs.OnsetPressureFunctional,
        libfuncs.AbsOnsetFrequencyFunctional
    ]
)
def functional(hopf_model, request):
    """Return a Hopf model functional"""
    hopf = hopf_model
    Functional = request.param
    return Functional(hopf)

class TestFunctionalGradient:
    """
    Test functions operating on functionals
    """
    # The below operators represent 'reduced' operators on the residual
    # This operator represents the map between property changes and state
    # through the implicit function theorem on the Hopf system residual
    def test_dstate_dprops(
            self,
            hopf_model,
            xhopf_props,
            dprop
        ):
        """Test a combined operator of `HopfModel`"""

        hopf = hopf_model
        xhopf, prop = xhopf_props

        def res(prop):
            x, info = libhopf.solve_hopf_by_newton(hopf, xhopf, prop)
            # print(info)
            assert info['status'] == 0
            return x

        def jac(prop, dprop):
            # Set the linearization point
            x, info = libhopf.solve_hopf_by_newton(hopf, xhopf, prop)
            assert info['status'] == 0
            # print(info)
            hopf.set_state(x)

            # Compute the jacobian action
            dres_dprops = hopf.assem_dres_dprops()
            hopf.zero_rows_dirichlet_bmat(dres_dprops)
            dres = bla.mult_mat_vec(dres_dprops, dprop)
            _dres = dres.to_mono_petsc()

            dstate = hopf.state.copy()
            _dstate = dstate.to_mono_petsc()
            dres_dstate = hopf.assem_dres_dstate()
            hopf.apply_dirichlet_bmat(dres_dstate)
            _dres_dstate = dres_dstate.to_mono_petsc()
            subops.solve_petsc_lu(_dres_dstate, -1*_dres, out=_dstate)
            dstate.set_mono(_dstate)

            return dstate
        taylor_convergence(prop, dprop, res, jac)

    # def test_dstate_dprops_adjoint():

    def test_solve_reduced_gradient(
            self,
            functional,
            hopf_model,
            xhopf_props,
            dprop
        ):
        """
        Test `solve_reduced_gradient`

        Parameters
        ----------
        func :
            The functional to compute the gradient of
        hopf :
            The Hopf model
        xhopf_props: (xhopf, prop, dprop)
        """
        hopf = hopf_model
        func = functional
        xhopf, prop = xhopf_props
        # dprop = dprop

        def res(prop):
            hopf.set_prop(prop)
            x, info = libhopf.solve_hopf_by_newton(hopf, xhopf, prop)
            assert info['status'] == 0

            func.set_state(x)
            func.set_prop(prop)
            return np.array(func.assem_g())

        def jac(prop, dprop):
            hopf.set_prop(prop)
            x, info = libhopf.solve_hopf_by_newton(hopf, xhopf, prop)

            hopf.set_state(x)
            func.set_state(x)
            func.set_prop(prop)

            return bla.dot(
                libhopf.solve_reduced_gradient(func, hopf, xhopf, prop),
                dprop
            )

        taylor_convergence(
            prop, dprop, res, jac,
            norm=lambda x: x
        )

class TestReducedFunctional:
    """
    Test `libhopf.ReducedFunctional`
    """

    @pytest.fixture()
    def rhopf(self, hopf_model):
        """
        Return a reduced Hopf model
        """
        hopf = hopf_model
        rhopf = libhopf.ReducedHopfModel(
            hopf
        )
        return rhopf

    @pytest.fixture()
    def rfunctional(self, functional, rhopf):
        """Return a `ReducedFunctional` instance"""
        func = functional

        return libhopf.ReducedFunctional(func, rhopf)

    @pytest.fixture()
    def props_list(
            self,
            xhopf_props,
            dprop
        ):
        """Return an iterable of `Hopf.prop` vectors"""
        _, prop = xhopf_props
        dprop = dprop

        propss = [
            bv.concatenate_vec([prop + alpha*dprop])
            for alpha in np.linspace(0, 100, 3)
        ]
        return propss

    @pytest.fixture()
    def norm(self, dprop):
        """
        Return a scaled norm

        This is used to generate reasonable step sizes for
        finite differences.
        """
        scale = dprop.copy()
        scale[:] = 1
        scale['emod'][:] = 1e4
        scale['umesh'][:] = 0.1

        def scaled_norm(x):
            return bla.norm(x/scale)

        return scaled_norm

    def test_set_props(self, rfunctional, props_list):
        """
        Test `ReducedFunctional.set_prop` solves for a Hopf bifurcation
        """
        hopf = rfunctional.rhopf_model.hopf
        for prop in props_list:
            # For each property in a list of properties to test, set the properties
            # of the ReducedFunctional; the ReducedFunctional should handle solving the
            # Hopf system implictly
            rfunctional.set_prop(prop)
            # print(redu_grad.assem_g())

            # Next, check that the Hopf system was correctly solved in
            # ReducedGradient by checking the Hopf residual
            hopf.set_state(rfunctional.rhopf_model.assem_state())
            hopf.set_prop(rfunctional.rhopf_model.prop)
            print(bla.norm(hopf.assem_res()))

    def test_assem_d2g_dprops2(
            self, rfunctional, xhopf_props, dprop, norm
        ):
        """
        Test `ReducedFunctional.assem_d2g_dprops2`
        """
        h = 1e-2
        xhopf, prop = xhopf_props

        # norm_dprops = norm(dprop)
        # unit_dprops = dprop/norm_dprops
        # unit_dprops.print_summary()

        def assem_grad(prop):
            # print(bla.norm(prop))
            rfunctional.set_prop(prop)
            return rfunctional.assem_dg_dprops().copy()

        def assem_hvp(prop, dprop):
            # print(bla.norm(prop))
            rfunctional.set_prop(prop)
            return rfunctional.assem_d2g_dprops2(dprop, h=h, norm=norm).copy()

        alphas, errs, mags, conv_rates = taylor_convergence(
            prop, dprop, assem_grad, assem_hvp, norm=bla.norm
        )

class TestOptGradManager:
    """
    Test the `OptGradManager` class
    """

    @pytest.fixture()
    def params(self, parameterization, xhopf_props, dprop):
        """
        Return a sequence of parameters
        """
        xhopf, prop = xhopf_props

        p0 = parameterization.x.copy()
        for key, subvec in prop.items():
            if key in p0:
                p0[key] = subvec

        dp = parameterization.x.copy()
        dp[:] = 0
        for key, subvec in dprop.items():
            if key in p0:
                dp[key] = subvec
        return [p0 + alpha*dp for alpha in np.linspace(0, 1, 2)]

    def test_OptGradManager(
            self,
            rfunctional,
            parameterization,
            params
        ):
        """
        Test the ReducedGradientManager object
        """
        redu_grad = rfunctional

        with h5py.File("out/_test_make_opt_grad.h5", mode='w') as f:
            grad_manager = libhopf.OptGradManager(redu_grad, f, parameterization)

            for param in params:
                print(grad_manager.grad(param.to_mono_ndarray()))

            print(f.keys())

            for key in list(f.keys()):
                if 'hopf_newton_' in key:
                    print(f"{key}: {f[key][:]}")

class TestReducedFunctionalHessianContext:

    @pytest.fixture()
    def rhopf(self, hopf_model):
        """
        Return a reduced Hopf model
        """
        hopf = hopf_model
        rhopf = libhopf.ReducedHopfModel(
            hopf
        )
        return rhopf

    @pytest.fixture()
    def rfunctional(self, functional, rhopf):
        """Return a `ReducedFunctional` instance"""
        func = functional

        return libhopf.ReducedFunctional(func, rhopf)

    @pytest.fixture()
    def context(self, rfunctional, parameterization):
        """
        Return a PETSc Python mat context
        """
        return libhopf.ReducedFunctionalHessianContext(
            rfunctional, parameterization
        )

    @pytest.fixture()
    def mat(self, context):
        """
        Return a PETSc Python mat
        """
        n = context.rfunctional.prop.mshape[0]
        m = n

        ret_mat = PETSc.Mat().createPython((m, n))
        ret_mat.setPythonContext(context)
        ret_mat.setUp()
        return ret_mat

    def test_mult(self, mat, context, params, dparams):
        """
        Test a PETSc Python mat's `mult` operation
        """
        context.set_params(params)

        x = dparams.to_mono_petsc()
        y = mat.getVecLeft()
        mat.mult(x, y)
