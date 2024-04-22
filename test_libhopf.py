"""
Test the `libhopf.hopf` module
"""

from typing import Tuple

# NOTE: Put this here to avoid `jax` import error
from femvf.parameters import transform

# import sys
from os import path
import warnings
import pytest
import h5py
import numpy as np
from petsc4py import PETSc

from blockarray import linalg as bla, blockvec as bv, blockmat as bm, subops

from libhopf import hopf, functional as libfunctional, setup
from libtest import taylor_convergence

# pylint: disable=redefined-outer-name
# pylint: disable=no-member, invalid-name

HopfModel = hopf.HopfModel
BVec = bv.BlockVector
BVecPair = Tuple[BVec, BVec]
BMat = bm.BlockMatrix


@pytest.fixture()
def hopf_model():
    """Return a Hopf bifurcation model"""
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_name = 'M5_CB_GA3_CL0.50'
    mesh_path = path.join('./mesh', mesh_name + '.msh')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        hopf_model, *_ = setup.load_hopf_model(
            mesh_path, sep_method='smoothmin', sep_vert_label='separation'
        )
    return hopf_model


@pytest.fixture()
def prop(hopf_model):
    """Return Hopf model properties"""

    prop = hopf_model.prop.copy()
    setup.set_default_props(prop, hopf_model.res.solid.residual.mesh())
    prop['kcontact'][:] = 1e0
    return prop


class TestHopfModel:
    """
    Test `HopfModel`
    """

    @pytest.fixture()
    def xhopf_prop(self, hopf_model: HopfModel, prop: bv.BlockVector):
        """
        Return a linearization point for the Hopf model

        The point is not necessarily a Hopf bifurcation.
        """
        # NOTE: Some `state` components can't be zero or Hopf jacobians may have zero rank
        (
            state_labels,
            mode_real_labels,
            mode_imag_labels,
            psub_labels,
            omega_labels,
        ) = hopf_model.labels_hopf_components

        # Create a pure x-shearing motion to use for displacement/velocities
        y = (
            hopf_model.res.solid.residual.form['coeff.state.u1']
            .function_space()
            .tabulate_dof_coordinates()[1::2, 1]
        )
        ux = 1e-2 * (y - y.min()) / (y.max() - y.min())
        uy = 0

        state = hopf_model.state.copy()
        state[:] = 0

        # TODO: Note that some subtle bugs may only appear for certain
        # linearization conditions
        # (for example non zero 'u_mode_real' but 0 'u')
        # so you should probably parameterize this in the future
        disp_labels = ['u', 'v']
        suffixes = ['', '_mode_real', '_mode_imag']
        for label in disp_labels:
            for suffix in suffixes:
                state[label + suffix][:-1:2] = ux
                state[label + suffix][1::2] = uy

        state[mode_real_labels] = 1.0
        state[mode_imag_labels] = 1.0
        PSUB = 100 * 10
        state[psub_labels] = PSUB
        state[omega_labels] = 1.0
        hopf_model.apply_dirichlet_bvec(state)

        return (state, prop)

    @pytest.fixture(
        params=[
            'u',
            'v',
            'u_mode_real',
            'v_mode_real',
            'u_mode_imag',
            'v_mode_imag',
            'psub',
            'omega',
        ]
    )
    def dstate(self, hopf_model: HopfModel, request: str):
        """Return a state perturbation"""
        dstate = hopf_model.state.copy()
        dstate[:] = 0

        label = request.param
        print(f"Testing along direction {label}")
        dstate[label] = 1e-4

        hopf_model.apply_dirichlet_bvec(dstate)
        return dstate

    def test_assem_dres_dstate(
        self, hopf_model: HopfModel, xhopf_prop: BVecPair, dstate: BVec
    ):
        """Test `HopfModel.assem_dres_dstate`"""
        state, prop = xhopf_prop
        hopf_model.set_prop(prop)

        def hopf_res(x):
            hopf_model.set_state(x)
            res = hopf_model.assem_res()
            hopf_model.apply_dirichlet_bvec(res)
            return hopf_model.assem_res()

        def hopf_jac(x, dx):
            hopf_model.set_state(x)
            dres_dstate = hopf_model.assem_dres_dstate()
            hopf_model.apply_dirichlet_bmat(dres_dstate)
            return bla.mult_mat_vec(dres_dstate, dx)

        taylor_convergence(state, dstate, hopf_res, hopf_jac)

    def test_assem_dres_dstate_adjoint(
        self, hopf_model: HopfModel, xhopf_prop: BVecPair, dstate: BVec
    ):
        """
        Test the adjoint of `HopfModel.assem_dres_dstate`

        This should be true as long as the tranpose is computed correctly.
        """
        state, prop = xhopf_prop
        hopf_model.set_state(state)
        hopf_model.set_prop(prop)

        dres_dstate = hopf_model.assem_dres_dstate()
        hopf_model.apply_dirichlet_bmat(dres_dstate)

        dres_adj = state.copy()
        dres_adj[:] = 0
        dres_adj['psub'] = 1
        # adj_res[:] = 1
        hopf_model.apply_dirichlet_bvec(dres_adj)

        dres_dstate_adj = hopf_model.assem_dres_dstate().transpose()
        hopf_model.apply_dirichlet_bmat(dres_dstate_adj)

        self._test_operator_adjoint(dres_dstate, dres_dstate_adj, dstate, dres_adj)

    def test_assem_dres_dstate_inv(
        self, hopf_model: HopfModel, xhopf_prop: BVecPair, dstate: BVec
    ):
        """Test `HopfModel.assem_dres_dstate`"""
        state, prop = xhopf_prop
        hopf_model.set_state(state)
        hopf_model.set_prop(prop)

        dres_dstate = hopf_model.assem_dres_dstate()
        hopf_model.apply_dirichlet_bmat(dres_dstate)
        hopf_model.apply_dirichlet_bvec(dstate)
        dres = bla.mult_mat_vec(dres_dstate, dstate)

        dstate_test = dres.copy()
        _dres_dstate = dres_dstate.to_mono_petsc()
        _dstate_test = _dres_dstate.getVecRight()
        subops.solve_petsc_lu(_dres_dstate, dres.to_mono_petsc(), out=_dstate_test)
        dstate_test.set_mono(_dstate_test)

        err = dstate - dstate_test
        print(err.norm())
        assert np.isclose(err.norm(), 0, rtol=1e-8, atol=1e-9)

    @pytest.fixture(
        params=[
            ('emod', 1e1),
            ('rho', 1e-2),
            ('fluid.rho_air', 1e-6),
            ('umesh', 1.0e-4),
        ]
    )
    def dprop(self, hopf_model: HopfModel, request: Tuple[str, float]):
        """Return a properties perturbation"""
        dprop = hopf_model.prop.copy()
        dprop[:] = 0.0
        label, value = request.param
        if 'fluid.' == label[:6]:
            for n in range(len(hopf_model.res.fluids)):
                dprop[f'fluid{n}.{label[6:]}'][:] = value
        else:
            dprop[label] = value
        print(f"Testing along direction {label}")
        return dprop

    def test_assem_dres_dprop(
        self, hopf_model: HopfModel, xhopf_prop: BVecPair, dprop: BVec
    ):
        """Test `HopfModel.assem_dres_dprop`"""
        state, prop = xhopf_prop
        hopf_model.set_state(state)

        def hopf_res(x):
            hopf_model.set_prop(x)
            res = hopf_model.assem_res()
            hopf_model.apply_dirichlet_bvec(res)
            return res

        def hopf_jac(x, dx):
            hopf_model.set_prop(x)
            dres_dprop = hopf_model.assem_dres_dprop()
            hopf_model.zero_rows_dirichlet_bmat(dres_dprop)
            return bla.mult_mat_vec(dres_dprop, dx)

        taylor_convergence(prop, dprop, hopf_res, hopf_jac)

    def test_assem_dres_dprop_adjoint(
        self, hopf_model: HopfModel, xhopf_prop: BVecPair, dprop: BVec
    ):
        """
        Test the adjoint of `HopfModel.assem_dres_dprop`

        This should be true as long as the tranpose is computed correctly.
        """
        state, prop = xhopf_prop
        hopf_model.set_state(state)
        hopf_model.set_prop(prop)

        dres_dprop = hopf_model.assem_dres_dprop()

        dres_adj = state.copy()
        dres_adj[:] = 1
        # dres_adj['psub'] = 1
        hopf_model.apply_dirichlet_bvec(dres_adj)

        dres_dprop_adj = dres_dprop.transpose()

        self._test_operator_adjoint(dres_dprop, dres_dprop_adj, dprop, dres_adj)

    def _test_operator_adjoint(self, op: BMat, op_adj: BVec, dx: BVec, dy_adj: BVec):
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
        assert np.isclose(bv.dot(dx_adj, dx), bv.dot(dy_adj, dy), rtol=1e-9, atol=1e-9)


class TestHopfUtilities:
    """
    Test utility functions for the Hopf bifurcation system
    """

    @pytest.fixture()
    def bound_pairs(self):
        """Return lower/upper subglottal pressure bounds"""
        lbs = [100.0 * 10]
        ubs = [800.0 * 10]
        return (lbs, ubs)

    def test_bound_hopf_bifurcations(
        self, hopf_model: HopfModel, prop: BVec, bound_pairs
    ):
        """Test `bound_hopf_bifurcations`"""
        hopf_model.set_prop(prop)

        dyn_model = hopf_model.res
        dyn_prop = hopf_model.res.prop
        dyn_control = hopf_model.res.control

        bounds, omegas = hopf.bound_ponset(
            dyn_model, dyn_control, dyn_prop, bound_pairs
        )
        print(f"Hopf bifurcations between {bounds[0]} and {bounds[1]}")
        print(f"with growth rates between {omegas[0]} and {omegas[1]}")

    def test_gen_hopf_initial_guess_from_bounds(
        self, hopf_model: HopfModel, prop: BVec, bound_pairs
    ):
        """Test `gen_hopf_initial_guess_from_bounds`"""
        hopf_model.set_prop(prop)

        dyn_model = hopf_model.res
        dyn_prop = dyn_model.prop

        xhopf_0 = hopf.gen_xhopf_0_from_bounds(
            dyn_model, dyn_prop, hopf_model.E_MODE, bound_pairs
        )

        xhopf_n, info = hopf.solve_hopf_by_newton(hopf_model, xhopf_0, prop)
        print(f"Solved Hopf system from automatic initial guess with info {info}")

    @pytest.fixture()
    def setup_xhopf_0(self, hopf_model: HopfModel, prop: BVec, bound_pairs):
        """Return an initial guess for the hopf state"""
        hopf_model.set_prop(prop)

        dyn_model = hopf_model.res
        dyn_prop = dyn_model.prop

        xhopf_0 = hopf.gen_xhopf_0_from_bounds(
            dyn_model, dyn_prop, hopf_model.E_MODE, bound_pairs
        )
        return xhopf_0

    def test_solve_hopf_newton(self, hopf_model: HopfModel, prop: BVec, setup_xhopf_0):
        """Test `solve_hopf_newton`"""
        xhopf_0 = setup_xhopf_0
        xhopf, info = hopf.solve_hopf_by_newton(hopf_model, xhopf_0, prop)
        print(info)


@pytest.fixture(
    params=[
        transform.TractionShape,
        # transform.Identity
    ]
)
def parameterization(hopf_model: HopfModel, request):
    """
    Return a parameterization
    """
    model = hopf_model.res
    Param = request.param
    return Param(model)


@pytest.fixture()
def param(hopf_model: HopfModel, parameterization):
    p0 = parameterization.x.copy()
    p0['emod'][:] = 10 * 1e3 * 10
    p0['rho'] = 1
    p0['nu'] = 0.45
    p0['eta'] = 5

    setup.set_default_props(p0, hopf_model.res.solid.residual.mesh())
    return p0


@pytest.fixture(params=[('emod', 1e2), ('umesh', 1.0e-4)])
def dparam(param: BVec, request):
    """Return a `params` perturbation"""
    dparams = param.copy()
    dparams[:] = 0

    key, val = request.param
    if key in dparams:
        dparams[key] = val
    return dparams


def solve_linearization(hopf_model: HopfModel, prop: BVec):
    """
    Return a linearization point corresponding to a Hopf bifurcation

    Returns
    -------
    xhopf, prop :
        `xhopf` - the state corresponding to the Hopf bifurcation
        to `prop`
    `prop` - the Hopf model properties
    """
    hopf_model.set_prop(prop)
    psubs = np.arange(1, 1000, 100) * 10
    xhopf_0 = hopf_model.state.copy()
    xhopf_0[:] = hopf.gen_xhopf_0(hopf_model.res, prop, hopf_model.E_MODE, psubs)
    xhopf, info = hopf.solve_hopf_by_newton(hopf_model, xhopf_0, prop)
    return xhopf, info


@pytest.fixture()
def xhopf_prop(hopf_model: HopfModel, prop: BVec) -> BVecPair:
    """
    Return a linearization point corresponding to a Hopf bifurcation
    """
    xhopf, info = solve_linearization(hopf_model, prop)
    return xhopf, prop


@pytest.fixture()
def xhopf_params(hopf_model: HopfModel, param: BVec):
    """
    Return a linearization point corresponding to a Hopf bifurcation
    """
    p0, parameterization = param
    xhopf, info = solve_linearization(hopf_model, parameterization.apply(p0))
    return xhopf, p0, parameterization


@pytest.fixture(params=[('emod', 1e2), ('umesh', 1.0e-4)])
def dprop_dir(request):
    return request.param


@pytest.fixture()
def dprop(prop: BVec, dprop_dir):
    """Return a `prop` perturbation"""

    dprop = prop.copy()
    dprop[:] = 0

    key, val = dprop_dir
    dprop[key] = val
    return dprop


@pytest.fixture(
    params=[
        libfunctional.StrainEnergyFunctional,
        libfunctional.OnsetPressureFunctional,
        libfunctional.AbsOnsetFrequencyFunctional,
    ]
)
def functional(hopf_model: HopfModel, request):
    """Return a Hopf model functional"""
    Functional = request.param
    return Functional(hopf_model)


class TestFunctionalGradient:
    """
    Test functions operating on functionals
    """

    # The below operators represent 'reduced' operators on the residual
    # This operator represents the map between property changes and state
    # through the implicit function theorem on the Hopf system residual
    def test_dstate_dprop(
        self, hopf_model: HopfModel, xhopf_prop: BVecPair, dprop: BVec
    ):
        """Test a combined operator of `HopfModel`"""
        xhopf, prop = xhopf_prop

        def res(prop):
            x, info = hopf.solve_hopf_by_newton(hopf_model, xhopf, prop)
            # print(info)
            assert info['status'] == 0
            return x

        def jac(prop, dprop):
            # Set the linearization point
            x, info = hopf.solve_hopf_by_newton(hopf_model, xhopf, prop)
            assert info['status'] == 0
            # print(info)
            hopf_model.set_state(x)

            # Compute the jacobian action
            dres_dprop = hopf_model.assem_dres_dprop()
            hopf_model.zero_rows_dirichlet_bmat(dres_dprop)
            dres = bla.mult_mat_vec(dres_dprop, dprop)
            _dres = dres.to_mono_petsc()

            dstate = hopf_model.state.copy()
            _dstate = dstate.to_mono_petsc()
            dres_dstate = hopf_model.assem_dres_dstate()
            hopf_model.apply_dirichlet_bmat(dres_dstate)
            _dres_dstate = dres_dstate.to_mono_petsc()
            subops.solve_petsc_lu(_dres_dstate, -1 * _dres, out=_dstate)
            dstate.set_mono(_dstate)

            return dstate

        taylor_convergence(prop, dprop, res, jac)

    # def test_dstate_dprop_adjoint():

    def test_solve_reduced_gradient(
        self, functional, hopf_model: HopfModel, xhopf_prop: BVecPair, dprop: BVec
    ):
        """
        Test `solve_reduced_gradient`

        Parameters
        ----------
        func :
            The functional to compute the gradient of
        hopf :
            The Hopf model
        xhopf_prop: (xhopf, prop, dprop)
        """
        func = functional
        xhopf, prop = xhopf_prop
        # dprop = dprop

        def res(prop):
            hopf_model.set_prop(prop)
            x, info = hopf.solve_hopf_by_newton(hopf_model, xhopf, prop)
            assert info['status'] == 0

            func.set_state(x)
            func.set_prop(prop)
            return np.array(func.assem_g())

        def jac(prop, dprop):
            hopf_model.set_prop(prop)
            x, info = hopf.solve_hopf_by_newton(hopf_model, xhopf, prop)

            hopf_model.set_state(x)
            func.set_state(x)
            func.set_prop(prop)

            return bla.dot(
                hopf.solve_reduced_gradient(func, hopf_model, xhopf, prop), dprop
            )

        taylor_convergence(prop, dprop, res, jac, norm=lambda x: x)


@pytest.fixture()
def rhopf(hopf_model: HopfModel):
    """
    Return a reduced Hopf model
    """
    rhopf = hopf.ReducedHopfModel(
        hopf_model, hopf_psub_intervals=10 * np.array([1.0, 800.0, 1600.0])
    )
    return rhopf


@pytest.fixture()
def rfunctional(
    functional: libfunctional.GenericFunctional, rhopf: hopf.ReducedHopfModel
):
    """Return a `ReducedFunctional` instance"""
    func = functional

    return hopf.ReducedFunctional(func, rhopf)


class TestReducedFunctional:
    """
    Test `hopf.ReducedFunctional`
    """

    @pytest.fixture()
    def props(self, xhopf_prop: BVecPair, dprop: BVec):
        """Return an iterable of `Hopf.prop` vectors"""
        _, prop = xhopf_prop

        props = [
            bv.concatenate([prop + alpha * dprop]) for alpha in np.linspace(0, 10, 3)
        ]
        return props

    @pytest.fixture()
    def norm(self, hopf_model: HopfModel, dprop: BVec):
        """
        Return a scaled norm

        This is used to generate reasonable step sizes for
        finite differences.
        """
        scale = dprop.copy()
        scale[:] = 1
        scale['emod'][:] = 1e4
        scale['umesh'][:] = 1e-3

        # Mass matrices for the different vector spaces
        form = hopf_model.res.solid.residual.form
        import dolfin as dfn

        dx = hopf_model.res.solid.residual.measure('dx')
        u = dfn.TrialFunction(form['coeff.prop.emod'].function_space())
        v = dfn.TestFunction(form['coeff.prop.emod'].function_space())
        M_EMOD = dfn.assemble(dfn.inner(u, v) * dx, tensor=dfn.PETScMatrix()).mat()

        # The `...[0]` is hard-coded because I did something weird with storing the
        # mesh displacement/shape property
        u = dfn.TrialFunction(form['coeff.prop.umesh'].function_space())
        v = dfn.TestFunction(form['coeff.prop.umesh'].function_space())
        M_SHAPE = dfn.assemble(dfn.inner(u, v) * dx, tensor=dfn.PETScMatrix()).mat()

        def scaled_norm(x):
            xs = x / scale
            dxs = xs.copy()
            dxs['emod'] = M_EMOD * xs.sub['emod']
            dxs['umesh'] = M_SHAPE * xs.sub['umesh']
            return bla.dot(x, dxs) ** 0.5

        return scaled_norm

    def test_set_prop(self, rfunctional: hopf.ReducedFunctional, props):
        """
        Test `ReducedFunctional.set_prop` solves for a Hopf bifurcation
        """
        hopf_model = rfunctional.rhopf_model.hopf
        for prop in props:
            # For each property in a list of properties to test, set the properties
            # of the ReducedFunctional; the ReducedFunctional should handle solving the
            # Hopf system implictly
            rfunctional.set_prop(prop)
            # print(redu_grad.assem_g())

            # Next, check that the Hopf system was correctly solved in
            # ReducedGradient by checking the Hopf residual
            hopf_model.set_state(rfunctional.rhopf_model.assem_state())
            hopf_model.set_prop(rfunctional.rhopf_model.prop)
            print(bla.norm(hopf_model.assem_res()))

    def test_assem_dg_dprop(
        self,
        rfunctional: hopf.ReducedFunctional,
        xhopf_prop: BVecPair,
        dprop: BVec,
    ):
        """
        Test `ReducedFunctional.assem_dg_dprop`
        """
        xhopf, prop = xhopf_prop

        def assem_g(prop):
            rfunctional.set_prop(prop)
            return rfunctional.assem_g()

        def assem_dg(prop, dprop):
            rfunctional.set_prop(prop)
            return bv.dot(rfunctional.assem_dg_dprop(), dprop)

        alphas, errs, mags, conv_rates = taylor_convergence(
            prop, dprop, assem_g, assem_dg, norm=lambda x: np.linalg.norm(x)
        )

    def test_assem_d2g_dprop2(
        self,
        rfunctional: hopf.ReducedFunctional,
        xhopf_prop: BVecPair,
        dprop: BVec,
        norm,
    ):
        """
        Test `ReducedFunctional.assem_d2g_dprop2`
        """
        h = 1e-6
        xhopf, prop = xhopf_prop

        def assem_grad(prop):
            # print(bla.norm(prop))
            rfunctional.set_prop(prop)
            return rfunctional.assem_dg_dprop().copy()

        def assem_hvp(prop, dprop):
            # print(bla.norm(prop))
            rfunctional.set_prop(prop)
            return rfunctional.assem_d2g_dprop2(dprop, h=h, norm=norm).copy()

        alphas, errs, mags, conv_rates = taylor_convergence(
            prop, dprop, assem_grad, assem_hvp, norm=bla.norm
        )
        # print(alphas, errs, mags, conv_rates)


class TestOptGradManager:
    """
    Test the `OptGradManager` class
    """

    @pytest.fixture()
    def params(self, parameterization, xhopf_prop: BVecPair, dprop):
        """
        Return a sequence of parameters
        """
        xhopf, prop = xhopf_prop

        p0 = parameterization.x.copy()
        for key, subvec in prop.items():
            if key in p0:
                p0[key] = subvec

        dp = parameterization.x.copy()
        dp[:] = 0
        for key, subvec in dprop.items():
            if key in p0:
                dp[key] = subvec
        return [p0 + alpha * dp for alpha in np.linspace(0, 1, 2)]

    def test_OptGradManager(self, rfunctional, parameterization, param):
        """
        Test the ReducedGradientManager object
        """
        redu_grad = rfunctional

        with h5py.File("out/_test_make_opt_grad.h5", mode='w') as f:
            grad_manager = hopf.OptGradManager(redu_grad, f, parameterization)

            for param in param:
                print(grad_manager.grad(param.to_mono_ndarray()))

            print(f.keys())

            for key in list(f.keys()):
                if 'hopf_newton_' in key:
                    print(f"{key}: {f[key][:]}")


class TestReducedFunctionalHessianContext:

    @pytest.fixture()
    def context(self, rfunctional, parameterization):
        """
        Return a PETSc Python mat context
        """
        return hopf.ReducedFunctionalHessianContext(rfunctional, parameterization)

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

    def test_mult(self, mat, context, param, dparam):
        """
        Test a PETSc Python mat's `mult` operation
        """
        context.set_params(param)

        x = dparam.to_mono_petsc()
        y = mat.getVecLeft()
        mat.mult(x, y)
