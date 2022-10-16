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
def props(hopf_model):
    """Return Hopf model properties"""
    hopf = hopf_model

    props = hopf.props.copy()
    libsetup.set_default_props(props, hopf.res.solid.forms['mesh.mesh'])
    props['kcontact'][:] = 1e0
    return props

class TestHopfModel:
    """
    Test `HopfModel`
    """

    @pytest.fixture()
    def linearization(self, hopf_model, props):
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

        props = props
        return (state, props)

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
            linearization,
            setup_dstate
        ):
        """Test `HopfModel.assem_dres_dstate`"""

        hopf = hopf_model
        state, props = linearization
        dstate = setup_dstate
        hopf.set_props(props)

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
            linearization,
            setup_dstate
        ):
        """
        Test the adjoint of `HopfModel.assem_dres_dstate`

        This should be true as long as the tranpose is computed correctly.
        """

        hopf = hopf_model
        state, props = linearization
        dstate = setup_dstate
        hopf.set_state(state)
        hopf.set_props(props)

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
            linearization,
            setup_dstate
        ):
        """Test `HopfModel.assem_dres_dstate`"""

        hopf = hopf_model
        state, props = linearization
        dstate = setup_dstate
        hopf.set_state(state)
        hopf.set_props(props)

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
    def dprops(self, hopf_model, request):
        """Return a properties perturbation"""
        hopf = hopf_model

        dprops = hopf.props.copy()
        dprops[:] = 0.0
        label, value = request.param
        dprops[label] = value
        print(f"Testing along direction {label}")
        return dprops

    def test_assem_dres_dprops(
            self,
            hopf_model,
            linearization,
            dprops
        ):
        """Test `HopfModel.assem_dres_dprops`"""

        hopf = hopf_model
        state, props = linearization
        dprops = dprops
        hopf.set_state(state)

        def hopf_res(x):
            hopf.set_props(x)
            res = hopf.assem_res()
            hopf.apply_dirichlet_bvec(res)
            return res

        def hopf_jac(x, dx):
            hopf.set_props(x)
            dres_dprops = hopf.assem_dres_dprops()
            hopf.zero_rows_dirichlet_bmat(dres_dprops)
            return bla.mult_mat_vec(dres_dprops, dx)
        taylor_convergence(props, dprops, hopf_res, hopf_jac)

    def test_assem_dres_dprops_adjoint(
            self,
            hopf_model,
            linearization,
            dprops
        ):
        """
        Test the adjoint of `HopfModel.assem_dres_dprops`

        This should be true as long as the tranpose is computed correctly.
        """
        hopf = hopf_model
        state, props = linearization
        dprops = dprops
        hopf.set_state(state)
        hopf.set_props(props)

        dres_dprops = hopf.assem_dres_dprops()

        dres_adj = state.copy()
        dres_adj[:] = 1
        # dres_adj['psub'] = 1
        hopf.apply_dirichlet_bvec(dres_adj)

        dres_dprops_adj = dres_dprops.transpose()

        self._test_operator_adjoint(
            dres_dprops, dres_dprops_adj,
            dprops, dres_adj
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
    def setup_bound_pairs(self):
        """Return lower/upper subglottal pressure bounds"""
        lbs = [400.0*10]
        ubs = [600.0*10]
        return (lbs, ubs)

    def test_bound_hopf_bifurcations(self, hopf_model, props, setup_bound_pairs):
        """Test `bound_hopf_bifurcations`"""
        hopf = hopf_model
        bound_pairs = setup_bound_pairs
        props = props
        hopf.set_props(props)

        bounds, omegas = libhopf.bound_hopf_bifurcations(hopf.res, bound_pairs)
        print(f"Hopf bifurcations between {bounds[0]} and {bounds[1]}")
        print(f"with growth rates between {omegas[0]} and {omegas[1]}")

    def test_gen_hopf_initial_guess_from_bounds(self, hopf_model, props, setup_bound_pairs):
        """Test `gen_hopf_initial_guess_from_bounds`"""
        hopf = hopf_model
        bound_pairs = setup_bound_pairs
        props = props
        hopf.set_props(props)

        eref = hopf.res.state.copy()
        eref[:] = 1.0

        xhopf_0 = libhopf.gen_hopf_initial_guess_from_bounds(hopf, bound_pairs)

        xhopf_n, info = libhopf.solve_hopf_newton(hopf, xhopf_0)
        print(f"Solved Hopf system from automatic initial guess with info {info}")

    @pytest.fixture()
    def setup_xhopf_0(self, hopf_model, props, setup_bound_pairs):
        """Return an initial guess for the hopf state"""
        hopf = hopf_model
        bound_pairs = setup_bound_pairs
        props = props
        hopf.set_props(props)

        xhopf_0 = libhopf.gen_hopf_initial_guess_from_bounds(hopf, bound_pairs)
        return xhopf_0

    def test_solve_hopf_newton(self, hopf_model, props, setup_xhopf_0):
        """Test `solve_hopf_newton`"""
        xhopf_0 = setup_xhopf_0
        hopf = hopf_model
        props = props
        hopf.set_props(props)

        xhopf, info = libhopf.solve_hopf_newton(hopf, xhopf_0)

@pytest.fixture()
def linearization(props, hopf_model):
    """
    Return a linearization point corresponding to a Hopf bifurcation

    Returns
    -------
    xhopf, props :
        `xhopf` - the state corresponding to the Hopf bifurcation
        to `props`
        `props` - the Hopf model properties
    """
    hopf = hopf_model

    # Create the ReducedGradientManager object;
    # currently this required the Hopf system have state and properties that
    # initially satisfy the equations
    hopf.set_props(props)
    psubs = np.arange(100, 1000, 100)*10
    xhopf_0 = libhopf.gen_hopf_initial_guess(hopf, psubs)
    xhopf, info = libhopf.solve_hopf_newton(hopf, xhopf_0)
    return xhopf, props

@pytest.fixture(
    params=[
        ('emod', 1e2),
        ('umesh', 1.0e-4)
    ]
)
def dprops(props, request):
    """Return a `props` perturbation"""

    dprops = props.copy()
    dprops[:] = 0

    key, val = request.param
    dprops[key] = val
    return dprops

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
            linearization,
            dprops
        ):
        """Test a combined operator of `HopfModel`"""

        hopf = hopf_model
        xhopf, props = linearization
        dprops = dprops

        def res(y):
            hopf.set_props(y)
            x, info = libhopf.solve_hopf_newton(hopf, xhopf)
            # print(info)
            assert info['status'] == 0
            return x

        def jac(y, dy):
            # Set the linearization point
            hopf.set_props(y)
            x, info = libhopf.solve_hopf_newton(hopf, xhopf)
            assert info['status'] == 0
            # print(info)
            hopf.set_state(x)

            # Compute the jacobian action
            dres_dprops = hopf.assem_dres_dprops()
            hopf.zero_rows_dirichlet_bmat(dres_dprops)
            dres = bla.mult_mat_vec(dres_dprops, dy)
            _dres = dres.to_mono_petsc()

            dstate = hopf.state.copy()
            _dstate = dstate.to_mono_petsc()
            dres_dstate = hopf.assem_dres_dstate()
            hopf.apply_dirichlet_bmat(dres_dstate)
            _dres_dstate = dres_dstate.to_mono_petsc()
            subops.solve_petsc_lu(_dres_dstate, -1*_dres, out=_dstate)
            dstate.set_mono(_dstate)

            return dstate
        taylor_convergence(props, dprops, res, jac)

    # def test_dstate_dprops_adjoint():

    def test_solve_reduced_gradient(
            self,
            functional,
            hopf_model,
            linearization,
            dprops
        ):
        """
        Test `solve_reduced_gradient`

        Parameters
        ----------
        func :
            The functional to compute the gradient of
        hopf :
            The Hopf model
        linearization: (xhopf, props, dprops)
        """
        hopf = hopf_model
        func = functional
        xhopf, props = linearization
        # dprops = dprops

        def res(props):
            hopf.set_props(props)
            x, info = libhopf.solve_hopf_newton(hopf, xhopf)
            assert info['status'] == 0

            func.set_state(x)
            func.set_props(props)
            return np.array(func.assem_g())

        def jac(props, dprops):
            hopf.set_props(props)
            x, info = libhopf.solve_hopf_newton(hopf, xhopf)

            hopf.set_state(x)
            func.set_state(x)
            func.set_props(props)

            return bla.dot(libhopf.solve_reduced_gradient(func, hopf), dprops)

        taylor_convergence(
            props, dprops, res, jac,
            norm=lambda x: x
        )

class TestReducedFunctional:
    """
    Test `ReducedFunctional` class
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
            linearization,
            dprops
        ):
        """Return an iterable of `Hopf.props` vectors"""
        _, props = linearization
        dprops = dprops

        propss = [
            bv.concatenate_vec([props + alpha*dprops])
            for alpha in np.linspace(0, 100, 3)
        ]
        return propss

    def test_ReducedGradient(self, rfunctional, props_list):
        """
        Test the ReducedGradientManager object
        """
        hopf = rfunctional.rhopf_model.res
        for props in props_list:
            # For each property in a list of properties to test, set the properties
            # of the ReducedGradient; the ReducedGradient should handle solving the
            # Hopf system implictly
            rfunctional.set_props(props)
            # print(redu_grad.assem_g())

            # Next, check that the Hopf system was correctly solved in
            # ReducedGradient by checking the Hopf residual
            hopf.set_state(rfunctional.rhopf_model.hist_state[-1])
            hopf.set_props(rfunctional.rhopf_model.hist_props[-1])
            print(bla.norm(hopf.assem_res()))

    def test_d2g_dprops2(self, rfunctional, linearization, dprops):
        """
        Test `ReducedFunctional.d2g_dprops2`
        """
        xhopf, props = linearization

        norm_dprops = bla.norm(dprops)
        unit_dprops = dprops/norm_dprops
        unit_dprops.print_summary()

        def assem_grad(props):
            # print(bla.norm(props))
            rfunctional.set_props(props)
            return rfunctional.assem_dg_dprops().copy()

        def assem_hvp(props, dprops):
            # print(bla.norm(props))
            rfunctional.set_props(props)
            return rfunctional.assem_d2g_dprops2(dprops).copy()

        h = 1e0
        # breakpoint()
        rfunctional.set_props(props)
        dgrad_fd = norm_dprops/(h)*(
            assem_grad(props+h*unit_dprops) - assem_grad(props)
        )

        rfunctional.set_props(props)
        dgrad_cd = norm_dprops/(2*h)*(
            assem_grad(props+h*unit_dprops) - assem_grad(props-h*unit_dprops)
        )
        dgrad_hvp = assem_hvp(props, dprops)
        print(bla.norm(dgrad_hvp), bla.norm(dgrad_fd), bla.norm(dgrad_cd), bla.norm(dgrad_hvp-dgrad_fd))

class TestOptGradManager:
    """
    Test the `OptGradManager` class
    """
    @pytest.fixture(
        params=[
            pazn.TractionShape,
            pazn.Identity
        ]
    )
    def parameterization(self, hopf_model, request):
        """
        Return a parameterization
        """
        model = hopf_model.res
        Param = request.param
        return Param(model, model.props)

    @pytest.fixture()
    def params(self, parameterization, linearization, dprops, ):
        """
        Return a sequence of parameters
        """
        xhopf, props = linearization

        p0 = parameterization.x.copy()
        for key, subvec in props.items():
            if key in p0:
                p0[key] = subvec

        dp = parameterization.x.copy()
        dp[:] = 0
        for key, subvec in dprops.items():
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
