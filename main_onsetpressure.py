"""
Conduct a sensitivity study of onset pressure and frequency
"""

from typing import Mapping, Optional, Any
from numpy.typing import NDArray

import argparse
import os.path as path
import multiprocessing as mp
import itertools
import warnings
from tqdm import tqdm

# NOTE: The import order for `dolfin` and other packages (numpy, scipy, etc.)
# seems to cause issues sometimes (at least on my installation)
import dolfin as dfn

dfn.set_log_level(50)
import h5py
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
from scipy import optimize

from blockarray import blockvec as bv, linalg as bla, h5utils
from femvf.models.dynamical import base as dbase
from femvf.parameters import transform as tfrm
from femvf.meshutils import process_celllabel_to_dofs_from_residual

from libhopf import hopf as libhopf, setup as libsetup, functional as libfuncs

from exputils import exputils

# pylint: disable=redefined-outer-name

parameter_types = {
    # For specifying the mesh
    'MeshName': str,
    # Controls how layer stiffnesses map to stiffness distributions
    # (discrete | linear)
    'LayerType': str,
    # Stiffnesses of the cover and body
    'Ecov': float,
    'Ebod': float,
    # Control which parameter to find the sensitivity to
    # (Stiffness | Shape | TractionShape)
    'ParamOption': str,
    # Control the functional to find the sensitivity of
    # (OnsetPressure | OnsetFlowrate | ...)
    'Functional': str,
    # The step size used when computing the Hessian
    'H': float,
    # The eigenvalue detection range string used by SLEPc
    # This is used when decomposing the Jacobian
    # (LARGEST_MAGNITUDE | LARGEST_REAL | ...)
    'EigTarget': str,
    # Controls where the fluid model separates
    # (depends on what you labelled vertices in the mesh)
    'SepPoint': str,
    # Controls the bifurcation parameter of the Hopf model
    # (psub | qsub)
    'BifParam': str,
}
ExpParamBasic = exputils.make_parameters(parameter_types, {})

# This is the range of subglottal pressures to check for onset pressure
PSUBS = np.linspace(0, 1500, 6) * 10

CLSCALE = 0.5


def setup_dyna_model(param: exputils.BaseParameters):
    """
    Return a Hopf model, and non-linear/linearized dynamical models

    Parameters
    ----------
    params: exputils.BaseParameters
        The experiment/case parameters
    """
    mesh_name = param['MeshName']
    mesh_path = path.join('./mesh', mesh_name + '.msh')

    mesh_name_parts = mesh_name.split('--')
    dz_strs = [part for part in mesh_name_parts if 'DZ' in part]
    nz_strs = [part for part in mesh_name_parts if 'NZ' in part]
    if len(dz_strs) == 1:
        dz = float(dz_strs[0][2:])
        nz = int(nz_strs[0][2:])
        zs = np.linspace(0, dz, nz+1)
    else:
        zs = None

    hopf, res, dres = libsetup.load_hopf_model(
        mesh_path,
        sep_method='arearatio',
        sep_vert_label=param['SepPoint'],
        bifparam_key=param['BifParam'],
        zs=zs
    )
    return hopf, res, dres


def setup_prop(param: exputils.BaseParameters, model: dbase.BaseDynamicalModel):
    """
    Return model properties

    Parameters
    ----------
    params: exputils.BaseParameters
        The experiment/case parameters
    model:
        The dynamical system model
    """
    prop = model.prop.copy()
    region_to_dofs = process_celllabel_to_dofs_from_residual(
        model.res.solid.residual,
        model.res.solid.residual.form['coeff.prop.emod'].function_space().dofmap(),
    )
    prop = set_prop(
        prop,
        model,
        region_to_dofs,
        param['Ecov'],
        param['Ebod'],
        layer_type=param['LayerType'],
    )
    return prop


def set_prop(
    prop: bv.BlockVector,
    hopf: libhopf.HopfModel,
    celllabel_to_dofs: Mapping[str, NDArray],
    emod_cov: float,
    emod_bod: float,
    layer_type='discrete',
):
    """
    Return the model properties vector with desired values
    """
    # Set any constant properties
    prop = libsetup.set_default_props(
        prop, hopf.res.solid.residual.mesh(), nfluid=len(hopf.res.fluids)
    )

    # Set cover and body layer properties
    emod = np.zeros(prop['emod'].shape)
    if layer_type == 'discrete':
        dofs_cov = np.array(celllabel_to_dofs['cover'], dtype=np.int32)
        dofs_bod = np.array(celllabel_to_dofs['body'], dtype=np.int32)
        dofs_share = set(dofs_cov) & set(dofs_bod)
        dofs_share = np.array(list(dofs_share), dtype=np.int32)

        emod[dofs_cov] = emod_cov
        emod[dofs_bod] = emod_bod
        emod[dofs_share] = 1 / 2 * (emod_cov + emod_bod)
    elif layer_type == 'linear':
        coord = (
            hopf.res.solid.residual.form['coeff.prop.emod']
            .function_space()
            .tabulate_dof_coordinates()
        )
        y = coord[:, 1]
        ymax, ymin = y.max(), y.min()
        emod[:] = emod_cov * (y - ymin) / (ymax - ymin) + emod_bod * (ymax - y) / (
            ymax - ymin
        )
    else:
        raise ValueError(f"Unknown `layer_type` {layer_type}")

    prop['emod'][:] = emod

    mesh = hopf.res.solid.residual.mesh()
    y_max = mesh.coordinates()[:, 1].max()
    y_gap = 0.05
    y_con_offset = 1 / 10 * y_gap
    prop['ymid'] = y_max + y_gap
    prop['ycontact'] = y_max + y_gap - y_con_offset

    prop['ncontact'][:] = 0.0
    prop['ncontact'][1] = 1.0
    return prop


def setup_functional(param: exputils.BaseParameters, model: dbase.BaseDynamicalModel):
    """
    Return a functional

    Parameters
    ----------
    params: exputils.BaseParameters
        The experiment/case parameters
    model:
        The dynamical system model
    """
    functionals = {
        'SubglottalFlowRate': libfuncs.SubglottalFlowRateFunctional(model),
        'SubglottalPressure': libfuncs.SubglottalPressureFunctional(model),
        'OnsetPressure': libfuncs.OnsetPressureFunctional(model),
        'OnsetFrequency': libfuncs.AbsOnsetFrequencyFunctional(model),
        'OnsetFlowRate': libfuncs.OnsetFlowRateFunctional(model),
    }

    return functionals[param['Functional']]


def setup_transform(
    param: exputils.BaseParameters, hopf: libhopf.HopfModel, prop: bv.BlockVector
):
    """
    Return a parameterization

    Parameters
    ----------
    params: exputils.BaseParameters
        The experiment/case parameters
    hopf:
        The Hopf system model
    prop:
        The Hopf model properties vector
    """
    const_vals = {key: np.array(subvec) for key, subvec in prop.sub_items()}
    parameter_scales = {'emod': 1e4}

    transform = None
    if param['ParamOption'] == 'Stiffness':
        const_vals.pop('emod')

        scale = tfrm.Scale(hopf.res.prop, scale=parameter_scales)
        const_subset = tfrm.ConstantSubset(hopf.res.prop, const_vals=const_vals)
        transform = scale * const_subset
    elif param['ParamOption'] == 'Shape':
        const_vals.pop('umesh')

        scale = tfrm.Scale(hopf.res.prop, scale=parameter_scales)
        const_subset = tfrm.ConstantSubset(hopf.res.prop, const_vals=const_vals)
        transform = scale * const_subset
    elif param['ParamOption'] == 'TractionShape':
        const_vals.pop('umesh')

        K, NU = 1e4, 0.2

        ## Apply `DirichletBC`s
        residual = hopf.res.solid.residual
        func_space = residual.form['coeff.prop.umesh'].function_space()

        mesh = residual.mesh()
        vertex_dim = 0
        cell_dim = mesh.topology().dim()
        facet_dim = cell_dim - 1

        # Fix the y-coordinate on the VF bottom surface
        mf = residual.mesh_function('facet')
        mf_label_to_value = residual.mesh_function_label_to_value('facet')
        dir_bc_const_y = dfn.DirichletBC(
            func_space.sub(1), 0.0, mf, mf_label_to_value['fixed']
        )

        # Fix the x-coordinate along the VF inferior edge
        class InferiorEdge(dfn.SubDomain):
            """
            Mark the inferior edge of the VF

            This is assumed to be at the origin
            """
            def inside(self, x: NDArray, on_boundary: bool):
                return np.linalg.norm(x[:2] - (0.0, 0.0)) == 0.0

        dir_bc_const_x = dfn.DirichletBC(
            func_space.sub(0), 0.0, InferiorEdge(), method='pointwise'
        )

        # Fix the z-coordinate at the VF anterior commisure (for 3D models)
        if mesh.topology().dim() >= 3:
            class AnteriorCommissure(dfn.SubDomain):
                """
                Mark the anterior commissure

                This is assumed to be at the origin as well
                """
                def inside(self, x: NDArray, on_boundary: bool):
                    # The anterior commissue is on the 'z = 0' plane
                    return (x[2] - 0.0) == 0.0

            dir_bc_const_x = dfn.DirichletBC(
                func_space.sub(2), 0.0, AnteriorCommissure(), method='pointwise'
            )

        traction_shape = tfrm.TractionShape(
            hopf.res,
            lame_lambda=(3 * K * NU) / (1 + NU),
            lame_mu=(3 * K * (1 - 2 * NU)) / (2 * (1 + NU)),
            dirichlet_bcs=[dir_bc_const_y, dir_bc_const_x],
        )
        extract = tfrm.ExtractSubset(traction_shape.x, keys_to_extract=('tmesh',))

        const_subset = tfrm.ConstantSubset(traction_shape.y, const_vals=const_vals)
        # scale = tfrm.Scale(const_subset.y, scale=parameter_scales)
        transform = extract * traction_shape * const_subset
        # transform = traction_shape * const_subset
        # transform = traction_shape * scale
    else:
        raise ValueError(f"Unknown 'ParamOption': {param['ParamOption']}")

    return transform, parameter_scales


def setup_reduced_functional(
    param: exputils.BaseParameters,
    lambda_tol: float = 10,
    lambda_intervals: NDArray = 10 * np.array([0, 500, 1000, 1500]),
    newton_params: Optional[Mapping[str, Any]] = None,
):
    """
    Return a reduced functional + additional stuff

    Parameters
    ----------
    param: exputils.BaseParameters
        The experiment/case parameters
    """
    ## Load the model and set model properties
    hopf, *_ = setup_dyna_model(param)

    _props = setup_prop(param, hopf)

    ## Setup the linearization/initial guess parameters
    transform, scale = setup_transform(param, hopf, _props)

    p = transform.x.copy()
    for key, subvec in _props.items():
        if key in p:
            p[key][:] = subvec

    if 'tmesh' in p:
        p['tmesh'][:] = 0

    # Apply the scaling that's used for `Scale`
    if isinstance(transform, tfrm.TransformComposition):
        has_scale_transform = any(
            isinstance(transform, tfrm.Scale) for transform in transform._transforms
        )
    else:
        has_scale_transform = isinstance(transform, tfrm.Scale)

    if has_scale_transform:
        for key, val in scale.items():
            if key in p:
                p[key][:] = p.sub[key] / val

    prop = transform.apply(p)
    assert np.isclose(bv.norm(prop - _props), 0)

    ## Solve for the Hopf bifurcation
    xhopf_0 = hopf.state.copy()
    with warnings.catch_warnings() as _:
        warnings.filterwarnings('always')
        if param['BifParam'] == 'psub':
            bifparam_tol = 100
            bifparams = PSUBS
        elif param['BifParam'] == 'qsub':
            bifparam_tol = 10
            bifparams = np.linspace(0, 300, 11)
        else:
            raise ValueError("")

        control = hopf.res.control
        xhopf_0[:] = libhopf.solve_hopf_by_range(
            hopf.res,
            control,
            prop,
            bifparams,
            bif_param_tol=bifparam_tol,
            eigvec_ref=hopf.E_MODE,
        )

    newton_params = {
        'absolute_tolerance': 1e-10,
        'relative_tolerance': 1e-5,
        'maximum_iterations': 5,
    }
    xhopf_n, info = libhopf.solve_hopf_by_newton(
        hopf, xhopf_0, prop, newton_params=newton_params, linear_solver='superlu'
    )
    if info['status'] != 0:
        raise RuntimeError(
            f"Hopf solution at linearization point didn't converge with info: {info}"
            f"; this happened for the parameter set {p}"
        )
    hopf.set_state(xhopf_n)

    ## Load the functional/objective function and gradient
    func = setup_functional(param, hopf)

    redu_hopf_model = libhopf.ReducedHopfModel(
        hopf,
        bif_param_tol=lambda_tol,
        bif_param_range=lambda_intervals,
        newton_params=newton_params,
    )
    redu_functional = libhopf.ReducedFunctional(func, redu_hopf_model)
    return redu_functional, hopf, xhopf_n, p, transform


def make_exp_params(study_name: str):
    """
    Return an iterable of experiment parameters (a parametric study)

    Parameters
    ----------
    study_name: str
        The name of the parametric study
    """

    default_param = ExpParamBasic(
        {
            'MeshName': f'M5_CB_GA3_CL{CLSCALE:.2f}',
            'LayerType': 'discrete',
            'Ecov': 2.5 * 10 * 1e3,
            'Ebod': 2.5 * 10 * 1e3,
            'ParamOption': 'Stiffness',
            'Functional': 'OnsetPressure',
            'H': 1e-3,
            'EigTarget': 'LARGEST_MAGNITUDE',
            'SepPoint': 'separation-inf',
            'BifParam': 'psub',
        }
    )

    emods = np.arange(2.5, 20, 2.5) * 10 * 1e3
    if study_name == 'none':
        return []
    elif study_name == 'test':
        # TODO: The flow rate driven model seems to have a lot of numerical error
        # and fails when using PETSc's LU solver
        params = [
            default_param.substitute(
                {
                    'Functional': 'OnsetPressure',
                    'MeshName': f'M5_CB_GA3_CL{0.5:.2f}',
                    'LayerType': 'discrete',
                    'Ecov': 6e4,
                    'Ebod': 6e4,
                    'ParamOption': 'TractionShape',
                    # 'ParamOption': 'Stiffness',
                    'BifParam': 'psub',
                }
            )
        ]
        return params
    elif study_name == 'test_3D':
        params = [
            default_param.substitute(
                {
                    'Functional': 'OnsetPressure',
                    'MeshName': f'M5_BC--GA3.00--DZ1.50e+00--NZ12--CL9.40e-01',
                    'LayerType': 'discrete',
                    'Ecov': 6e4,
                    'Ebod': 6e4,
                    'ParamOption': 'TractionShape',
                    # 'ParamOption': 'Stiffness',
                    'BifParam': 'psub',
                }
            )
        ]
        return params
    elif study_name == 'test_traction_shape':
        # Modify this one as you need for different tests
        emods = [6e4]
        mesh_names = (
            [f'M5_CB_GA3_CL{0.5:.2f}']
            + ['Trapezoid_GA1.00']
            + [f'Trapezoid_GA{angle:>0.2f}' for angle in range(5, 31, 5)]
        )
        layer_types = ['discrete'] + ['linear'] + len(range(5, 31, 5)) * ['linear']
        params = [
            default_param.substitute(
                {
                    'Functional': 'OnsetPressure',
                    'MeshName': mesh_name,
                    'LayerType': layer_type,
                    'Ecov': emod,
                    'Ebod': emod,
                    'BifParam': 'psub',
                    'ParamOption': 'TractionShape',
                }
            )
            for (mesh_name, layer_type), emod in itertools.product(
                zip(mesh_names, layer_types), emods
            )
        ]
        return params
    elif study_name == 'test_shape':
        params = [
            default_param.substitute(
                {
                    'Functional': 'OnsetPressure',
                    'MeshName': f'M5_CB_GA3_CL{0.5:.2f}',
                    'LayerType': 'discrete',
                    'Ecov': 6e4,
                    'Ebod': 6e4,
                    'BifParam': 'psub',
                    'ParamOption': 'TractionShape',
                    'H': h,
                }
            )
            for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
        ]
        return params
    elif study_name == 'optimize_traction_shape':
        params = [
            default_param.substitute(
                {
                    'Functional': 'OnsetPressure',
                    'MeshName': f'M5_CB_GA3_CL{0.5:.2f}',
                    'LayerType': 'discrete',
                    'Ecov': 6e4,
                    'Ebod': 6e4,
                    'BifParam': 'psub',
                    'ParamOption': 'TractionShape',
                    'H': 1e-3,
                }
            )
        ]
        return params
    elif study_name == 'main_traction_shape':
        functional_names = ['OnsetPressure']
        emod_covs = (
            np.concatenate(
                [
                    (1) * np.arange(2, 18, 4),
                    (1 / 2) * np.arange(2, 18, 4),
                    (1 / 3) * np.arange(2, 18, 4),
                    (1 / 4) * np.arange(2, 18, 4),
                ]
            )
            * 10
            * 1e3
        )
        emod_bods = (
            np.concatenate(
                [
                    1 * np.arange(2, 18, 4),
                    1 * np.arange(2, 18, 4),
                    1 * np.arange(2, 18, 4),
                    1 * np.arange(2, 18, 4),
                ]
            )
            * 10
            * 1e3
        )

        # emod_covs = np.array([6, 3]) * 10 * 1e3
        # emod_bods = np.array([6, 6]) * 10 * 1e3

        assert len(emod_covs) == len(emod_bods)

        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]
        params = (
            default_param.substitute(
                {
                    'Functional': func_name,
                    'Ecov': emod_cov,
                    'Ebod': emod_bod,
                    'MeshName': f'M5_CB_GA3_CL{0.5:.2f}',
                    'ParamOption': 'TractionShape',
                }
            )
            for func_name, (emod_cov, emod_bod) in itertools.product(
                functional_names, emods
            )
        )
        return params
    elif study_name == 'main_sensitivity':
        functional_names = ['OnsetPressure', 'OnsetFrequency']
        param_options = ['const_shape']
        emod_covs = (
            np.concatenate(
                [
                    (1) * np.arange(2, 18, 4),
                    (1 / 2) * np.arange(2, 18, 4),
                    (1 / 3) * np.arange(2, 18, 4),
                    (1 / 4) * np.arange(2, 18, 4),
                ]
            )
            * 10
            * 1e3
        )
        emod_bods = (
            np.concatenate(
                [
                    1 * np.arange(2, 18, 4),
                    1 * np.arange(2, 18, 4),
                    1 * np.arange(2, 18, 4),
                    1 * np.arange(2, 18, 4),
                ]
            )
            * 10
            * 1e3
        )

        # emod_covs = np.array([6, 3]) * 10 * 1e3
        # emod_bods = np.array([6, 6]) * 10 * 1e3

        assert len(emod_covs) == len(emod_bods)

        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]
        params = (
            default_param.substitute(
                {
                    'Functional': func_name,
                    'Ecov': emod_cov,
                    'Ebod': emod_bod,
                    'ParamOption': param_option,
                }
            )
            for func_name, (emod_cov, emod_bod), param_option in itertools.product(
                functional_names, emods, param_options
            )
        )
        return params
    elif study_name == 'main_sensitivity_flowdriven':
        functional_names = [
            'OnsetFlowRate',
            'OnsetFrequency',
            'SubglottalPressure',
            'SubglottalFlowRate',
        ]
        param_options = ['const_shape']
        emod_covs = np.array([6.0e4])
        emod_bods = np.array([6.0e4])

        assert len(emod_covs) == len(emod_bods)

        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]
        params = (
            default_param.substitute(
                {
                    'Functional': func_name,
                    'Ecov': emod_cov,
                    'Ebod': emod_bod,
                    'ParamOption': param_option,
                    'BifParam': 'qsub',
                }
            )
            for func_name, (emod_cov, emod_bod), param_option in itertools.product(
                functional_names, emods, param_options
            )
        )
        return params
    elif study_name == 'main_coarse_sensitivity':
        functional_names = ['OnsetPressure', 'OnsetFrequency']
        param_options = ['const_shape']
        emod_covs = (
            np.concatenate(
                [
                    (1) * np.arange(2, 18, 4),
                    # (1/2)*np.arange(2, 18, 4),
                    (1 / 3) * np.arange(2, 18, 4),
                    (1 / 4) * np.arange(2, 18, 4),
                ]
            )
            * 10
            * 1e3
        )
        emod_bods = (
            np.concatenate(
                [
                    1 * np.arange(2, 18, 4),
                    # 1*np.arange(2, 18, 4),
                    1 * np.arange(2, 18, 4),
                    1 * np.arange(2, 18, 4),
                ]
            )
            * 10
            * 1e3
        )

        assert len(emod_covs) == len(emod_bods)

        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]
        params = (
            default_param.substitute(
                {
                    'Functional': func_name,
                    'Ecov': emod_cov,
                    'Ebod': emod_bod,
                    'ParamOption': param_option,
                }
            )
            for func_name, (emod_cov, emod_bod), param_option in itertools.product(
                functional_names, emods, param_options
            )
        )
        return params
    elif study_name == 'test_sensitivity':
        functional_names = ['OnsetPressure', 'OnsetFrequency']
        param_options = ['const_shape']
        emod_covs = 1e4 * np.array([2])
        emod_bods = 1e4 * np.array([2])

        assert len(emod_covs) == len(emod_bods)

        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]
        params = (
            default_param.substitute(
                {
                    'Functional': func_name,
                    'Ecov': emod_cov,
                    'Ebod': emod_bod,
                    'ParamOption': param_option,
                }
            )
            for func_name, (emod_cov, emod_bod), param_option in itertools.product(
                functional_names, emods, param_options
            )
        )
        return params
    elif study_name == 'separation_effect':
        functional_names = ['OnsetPressure']
        param_options = ['const_shape']
        eig_targets = ['LARGEST_MAGNITUDE']
        layer_types = ['discrete']  # , 'linear'

        emod_covs = 1e4 * np.array([2, 6, 2])
        emod_bods = 1e4 * np.array([2, 6, 6])

        emod_covs = 1e4 * np.array([6, 2])
        emod_bods = 1e4 * np.array([6, 6])
        assert len(emod_covs) == len(emod_bods)
        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]

        hs = np.array([1e-3])

        clscales = (0.5, 0.25, 0.125)
        clscales = (0.5,)
        # mesh_names = [
        #     f'M5_CB_GA3_CL{clscale:.2f}_split' for clscale in clscales
        # ]
        # sep_points = [
        #     # 'separation-inf',
        #     'sep1'
        # ]

        mesh_names = [f'M5_CB_GA3_CL{clscale:.2f}_split6' for clscale in clscales]
        sep_points = [
            # 'separation-inf',
            'sep1',
            'sep2',
            'sep3',
            'sep4',
        ]
        params = (
            default_param.substitute(
                {
                    'MeshName': mesh_name,
                    'LayerType': layer_type,
                    'Functional': func_name,
                    'Ecov': emod_cov,
                    'Ebod': emod_bod,
                    'ParamOption': param_option,
                    'H': h,
                    'EigTarget': eig_target,
                    'SepPoint': sep_point,
                }
            )
            for mesh_name, layer_type, h, func_name, (
                emod_cov,
                emod_bod,
            ), param_option, eig_target, sep_point in itertools.product(
                mesh_names,
                layer_types,
                hs,
                functional_names,
                emods,
                param_options,
                eig_targets,
                sep_points,
            )
        )
        return params
    elif study_name == 'independence':
        functional_names = ['OnsetPressure']

        param_options = ['const_shape']

        eig_targets = ['LARGEST_MAGNITUDE']

        emod_covs = 1e4 * np.array([6, 2])
        emod_bods = 1e4 * np.array([6, 6])
        assert len(emod_covs) == len(emod_bods)
        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]

        hs = np.array([1e-2, 1e-3, 1e-4])
        mesh_names = [f'M5_CB_GA3_CL{clscale:.2f}' for clscale in (1, 0.5, 0.25)]
        params = (
            default_param.substitute(
                {
                    'MeshName': mesh_name,
                    'Functional': func_name,
                    'Ecov': emod_cov,
                    'Ebod': emod_bod,
                    'ParamOption': param_option,
                    'H': h,
                    'EigTarget': eig_target,
                }
            )
            for mesh_name, h, func_name, (
                emod_cov,
                emod_bod,
            ), param_option, eig_target in itertools.product(
                mesh_names, hs, functional_names, emods, param_options, eig_targets
            )
        )
        return params
    elif study_name == 'eig_target_effect':
        functional_names = ['OnsetPressure']

        param_options = ['const_shape']

        eig_targets = ['LARGEST_REAL']

        emod_covs = 1e4 * np.array([2])
        emod_bods = 1e4 * np.array([6])
        assert len(emod_covs) == len(emod_bods)
        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]

        hs = np.array([1e-3])
        mesh_names = [f'M5_CB_GA3_CL{clscale:.2f}' for clscale in (0.5, 0.25, 0.125)]
        params = (
            default_param.substitute(
                {
                    'MeshName': mesh_name,
                    'Functional': func_name,
                    'Ecov': emod_cov,
                    'Ebod': emod_bod,
                    'ParamOption': param_option,
                    'H': h,
                    'EigTarget': eig_target,
                }
            )
            for mesh_name, h, func_name, (
                emod_cov,
                emod_bod,
            ), param_option, eig_target in itertools.product(
                mesh_names, hs, functional_names, emods, param_options, eig_targets
            )
        )
        return params
    else:
        raise ValueError("Unknown `study_name` '{study_name}'")


def make_norm(hopf_model: libhopf.HopfModel):
    """
    Return a norm for the model properties vector

    The norm is used to ensure that a step size in the properties for
    finite differences (FD) is independent of discretization. This is so that
    a FD step size roughly behaves the same between coarse/fine discretizations.

    Parameters
    ----------
    hopf_model:
        The Hopf system model
    """

    scale = hopf_model.prop.copy()
    scale[:] = 1
    scale['emod'][:] = 1e4
    scale['umesh'][:] = 1e-4

    # To compute discretization independent norms, use mass matrices to define
    # norms for properties that vary with the mesh

    # Mass matrix for the space of elastic moduli:
    form = hopf_model.res.solid.residual.form
    dx = hopf_model.res.solid.residual.measure('dx')
    u = dfn.TrialFunction(form['coeff.prop.emod'].function_space())
    v = dfn.TestFunction(form['coeff.prop.emod'].function_space())
    M_EMOD = dfn.assemble(dfn.inner(u, v) * dx, tensor=dfn.PETScMatrix()).mat()

    # Mass matrix for the space of mesh deformations:
    u = dfn.TrialFunction(form['coeff.prop.umesh'].function_space())
    v = dfn.TestFunction(form['coeff.prop.umesh'].function_space())
    M_UMESH = dfn.assemble(dfn.inner(u, v) * dx, tensor=dfn.PETScMatrix()).mat()

    def norm(x):
        """
        Return a scaled norm

        This norm roughly accounts for the difference in scale between
        modulus, shape changes, etc.
        """
        xs = x / scale
        dxs = xs.copy()
        dxs['emod'] = M_EMOD * xs.sub['emod']
        dxs['umesh'] = M_UMESH * xs.sub['umesh']
        return bla.dot(x, dxs) ** 0.5

    return norm


def objective_bv(p: bv.BlockVector, rfunc: 'ReducedFunctional', transform: 'Transform'):
    prop = transform.apply(p)
    try:
        rfunc.set_prop(prop)
        fun = rfunc.assem_g()
        grad_prop = rfunc.assem_dg_dprop()
        grad = transform.apply_vjp(p, grad_prop)
    except RuntimeError as err:
        warnings.warn(
            "Couldn't solve objective function for input due to error: " f"{err}",
            category=RuntimeWarning,
        )
        fun = np.nan
        grad = transform.x.copy()
        grad[:] = np.nan

    return fun, grad


def objective(p, rfunc, transform):
    _p = transform.x.copy()
    _p.set_mono(p)

    fun, grad = objective_bv(_p, rfunc, transform)

    return fun, grad.to_mono_ndarray()


def run_functional_sensitivity(
    param: exputils.BaseParameters, output_dir='out/sensitivity', overwrite=False
):
    """
    Run an experiment where the sensitivity of a functional is saved
    """
    rfunc, hopf, xhopf_n, p0, transform = setup_reduced_functional(param)
    fpath = path.join(output_dir, param.to_str() + '.h5')
    if not path.isfile(fpath) or overwrite:
        ## Compute 1st order sensitivity of the functional
        rfunc.set_prop(transform.apply(p0))

        # DEBUG:
        # breakpoint()

        print("Solving for gradient")
        grad_prop = rfunc.assem_dg_dprop()
        grad_param = transform.apply_vjp(p0, grad_prop)
        print("Finished solving for gradient")

        ## Compute 2nd order sensitivity of the functional
        print("Solving for eigenvalues of Hessian")
        norm = make_norm(hopf)
        redu_hess_context = libhopf.ReducedFunctionalHessianContext(
            rfunc, transform, norm=norm, step_size=param['H']
        )

        print("Setting Hessian linearization point")
        redu_hess_context.set_param(p0)
        mat = PETSc.Mat().createPython(p0.mshape * 2)
        mat.setPythonContext(redu_hess_context)
        mat.setUp()

        print("Creating `SLEPc` eigenvalue solver")
        eps = SLEPc.EPS().create()
        eps.setOperators(mat)
        eps.setDimensions(5, 25)
        eps.setProblemType(SLEPc.EPS.ProblemType.HEP)

        if param['EigTarget'] == 'LARGEST_MAGNITUDE':
            which_eig = SLEPc.EPS.Which.LARGEST_MAGNITUDE
        elif param['EigTarget'] == 'LARGEST_REAL':
            which_eig = SLEPc.EPS.Which.LARGEST_REAL
        else:
            raise ValueError(f"Unknown `params['EigTarget']` key {param['EigTarget']}")
        eps.setWhichEigenpairs(which_eig)
        eps.setUp()

        print("Solving `SLEPc` eigenvalues")
        eigvecs = []
        eigvals = []

        eps.solve()

        neig = eps.getConverged()
        _real_evec = mat.getVecRight()
        for n in range(neig):
            eigval = eps.getEigenpair(n, _real_evec)
            eigvec = transform.x.copy()
            eigvec.set_mono(_real_evec)
            eigvals.append(eigval)
            eigvecs.append(eigvec)

        ## Write sensitivities to an h5 file
        with h5py.File(fpath, mode='w') as f:
            ## Write out the gradient vectors
            for key, vec in zip(['grad_props', 'grad_param'], [grad_prop, grad_param]):
                h5utils.create_resizable_block_vector_group(
                    f.require_group(key),
                    vec.labels,
                    vec.bshape,
                    dataset_kwargs={'dtype': 'f8'},
                )
                h5utils.append_block_vector_to_group(f[key], vec)

            ## Write out the hessian eigenvalues and vectors
            f.create_dataset('eigvals', data=np.array(eigvals).real)
            for key, vecs in zip(['hess_param'], [eigvecs]):
                if len(vecs) > 0:
                    h5utils.create_resizable_block_vector_group(
                        f.require_group(key),
                        vecs[0].labels,
                        vecs[0].bshape,
                        dataset_kwargs={'dtype': 'f8'},
                    )
                    for vec in vecs:
                        h5utils.append_block_vector_to_group(f[key], vec)

            ## Write out the state + properties vectors
            for key, vec in zip(
                ['state', 'prop', 'param'], [hopf.state, hopf.prop, p0]
            ):
                h5utils.create_resizable_block_vector_group(
                    f.require_group(key),
                    vec.labels,
                    vec.bshape,
                    dataset_kwargs={'dtype': 'f8'},
                )
                h5utils.append_block_vector_to_group(f[key], vec)
    else:
        print(f"Skipping existing file '{fpath}'")


def run_optimization(
    param: exputils.BaseParameters, output_dir='out/optimization', overwrite=False
):
    rfunc, hopf, xhopf_n, p0, transform = setup_reduced_functional(
        param, lambda_tol=10.0, lambda_intervals=10 * np.array([0, 500, 1000, 1500])
    )
    # breakpoint()

    # `_p` represents the `BlockVector` representation of the optimization
    # variable
    # `p` is the raw numpy array, which is needed for `minimize`
    # to interface with the gradient code, this is inserted into `_p`
    _p = transform.x.copy()

    with h5py.File(f'optimization_hist.h5', mode='w') as h5file:
        grad_manager = libhopf.OptGradManager(rfunc, h5file, transform)

        def f(p):
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                K = 0.0
                C = 1e0
                obj_reg = K * 1 / 2 * np.linalg.norm(p) ** 2
                grad_reg = K * p

                _p = transform.x.copy()
                _p.set_mono(p)
                # obj, grad_p = objective_bv(_p, rfunc, transform)
                obj, grad_p = grad_manager.grad(_p)
                grad = grad_p.to_mono_ndarray()
                print(f"Called objective function with ||p||={np.linalg.norm(p)}")
                print(f"Objective function returned f(p)={obj}")
                print(f"Objective function returned ||grad(p)||={np.linalg.norm(grad)}")
                return C * obj + obj_reg, C * grad + grad_reg

        def callback(intermediate_result):
            print(intermediate_result)

        opt_options = {'maxiter': 10}
        opt_info = optimize.minimize(
            f,
            p0.to_mono_ndarray(),
            method='L-BFGS-B',
            jac=True,
            callback=callback,
            options=opt_options,
        )
        print(opt_info)
    # breakpoint()


def run_test(
    param: exputils.BaseParameters, output_dir='out/optimization', overwrite=False
):
    rfunc, hopf, xhopf_n, p0, transform = setup_reduced_functional(param)

    breakpoint()
    psub_0, grad_0 = objective_bv(p0, rfunc, transform)
    # psub_0, grad_0 = objective(p0.to_mono_ndarray(), rfunc, transform)

    dp = p0.copy()
    dp[:] = 0
    if 'tmesh' in dp:
        dp['tmesh'][:] = 1.0
    elif 'emod' in dp:
        dp['emod'][:] = 1e1

    dh_base = 1e-3
    dhs = dh_base * 2 ** np.arange(0, 7)
    psubs = []
    for dh in dhs:
        psub, grad = objective((p0 + dh * dp).to_mono_ndarray(), rfunc, transform)
        psubs.append(float(psub))
        print(psub, np.linalg.norm(grad))

    dpsubs_approx = np.array(psubs - psub_0) / dhs
    dpsub_exact = np.dot(grad_0.to_mono_ndarray(), dp.to_mono_ndarray())
    print(dpsubs_approx)
    print(dpsub_exact)
    breakpoint()


if __name__ == '__main__':
    # Load the Hopf system

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--study-name', type=str, default='none')
    argparser.add_argument('--num-proc', type=int, default=1)
    argparser.add_argument('--clscale', type=float, default=1.0)
    argparser.add_argument('--overwrite', action='store_true')
    argparser.add_argument('--output-dir', type=str, default='out')
    argparser.add_argument('--study-type', type=str, default='sensitivity')
    clargs = argparser.parse_args()
    CLSCALE = clargs.clscale

    params = make_exp_params(clargs.study_name)

    # NOTE: Input as a parameter dictionary is needed to allow multiprocessing
    if clargs.study_type == 'sensitivity':

        def run(param_dict):
            param = ExpParamBasic(param_dict)
            return run_functional_sensitivity(
                param, output_dir=clargs.output_dir, overwrite=clargs.overwrite
            )

    elif clargs.study_type == 'optimization':

        def run(param_dict):
            param = ExpParamBasic(param_dict)
            return run_optimization(
                param, output_dir=clargs.output_dir, overwrite=clargs.overwrite
            )

    elif clargs.study_type == 'test':

        def run(param_dict):
            param = ExpParamBasic(param_dict)
            return run_test(
                param, output_dir=clargs.output_dir, overwrite=clargs.overwrite
            )

    else:
        raise ValueError(f"Unknown option `--study-type` {clargs.study_type}")

    if clargs.num_proc == 1:
        for param in tqdm(params):
            run(param)
    else:
        with mp.Pool(clargs.num_proc) as pool:
            pool.map(run, [p.data for p in params])
