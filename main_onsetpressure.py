"""
Conduct a sensitivity study of onset pressure and frequency
"""

from typing import Mapping
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

from blockarray import blockvec as bv, linalg as bla, h5utils
from femvf.models.dynamical import (
    base as dbase
)
from femvf.parameters import transform as tfrm
from femvf.meshutils import process_celllabel_to_dofs_from_residual

from libhopf import hopf as libhopf, setup as libsetup, functional as libfuncs

from exputils import exputils

# pylint: disable=redefined-outer-name

parameter_types = {
    'MeshName': str,
    'LayerType': str,
    'Ecov': float,
    'Ebod': float,
    'ParamOption': str,
    'Functional': str,
    'H': float,
    'EigTarget': str,
    'SepPoint': str,
    'BifParam': str
}
ExpParamBasic = exputils.make_parameters(parameter_types)

PSUBS = np.arange(0, 800, 50) * 10


def setup_dyna_model(params: exputils.BaseParameters):
    """
    Return a Hopf model, and non-linear/linearized dynamical models

    Parameters
    ----------
    params: exputils.BaseParameters
        The experiment/case parameters
    """
    mesh_name = params['MeshName']
    mesh_path = path.join('./mesh', mesh_name+'.msh')

    hopf, res, dres = libsetup.load_hopf_model(
        mesh_path, sep_method='arearatio',
        sep_vert_label=params['SepPoint'],
        bifparam_key=params['BifParam']
    )
    return hopf, res, dres

def setup_props(
        params: exputils.BaseParameters,
        model: dbase.BaseDynamicalModel
    ):
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
        model.res.solid.residual.form['coeff.prop.emod'].function_space().dofmap()
    )
    prop = set_prop(
        prop, model, region_to_dofs,
        params['Ecov'], params['Ebod'],
        layer_type=params['LayerType']
    )
    return prop

def set_prop(
        prop: bv.BlockVector,
        hopf: libhopf.HopfModel,
        celllabel_to_dofs: Mapping[str, NDArray],
        emod_cov: float,
        emod_bod: float,
        layer_type='discrete'
    ):
    """
    Return the model properties vector with desired values
    """
    # Set any constant properties
    prop = libsetup.set_default_props(
        prop, hopf.res.solid.residual.mesh(), nfluid=len(hopf.res.fluids)
    )

    # Set cover and body layer properties
    dofs_cov = np.array(celllabel_to_dofs['cover'], dtype=np.int32)
    dofs_bod = np.array(celllabel_to_dofs['body'], dtype=np.int32)
    dofs_share = set(dofs_cov) & set(dofs_bod)
    dofs_share = np.array(list(dofs_share), dtype=np.int32)

    emod = np.zeros(prop['emod'].shape)
    if layer_type == 'discrete':
        emod[dofs_cov] = emod_cov
        emod[dofs_bod] = emod_bod
        emod[dofs_share] = 1/2*(emod_cov + emod_bod)
    elif layer_type == 'linear':
        coord = (
            hopf.res.solid.residual.form['coeff.prop.emod']
            .function_space().tabulate_dof_coordinates()
        )
        y = coord[:, 1]
        ymax, ymin = y.max(), y.min()
        emod[:] = (
            emod_cov*(y-ymin)/(ymax-ymin)
            + emod_bod*(ymax-y)/(ymax-ymin)
        )
    else:
        raise ValueError(f"Unknown `layer_type` {layer_type}")

    prop['emod'][:] = emod

    mesh = hopf.res.solid.residual.mesh()
    y_max = mesh.coordinates()[:, 1].max()
    y_gap = 0.05
    y_con_offset = 1/10*y_gap
    prop['ymid'] = y_max + y_gap
    prop['ycontact'] = y_max + y_gap - y_con_offset

    prop['ncontact'][:] = 0.0
    prop['ncontact'][1] = 1.0
    return prop

def setup_functional(
        params: exputils.BaseParameters,
        model: dbase.BaseDynamicalModel
    ):
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
        'OnsetFlowRate': libfuncs.OnsetFlowRateFunctional(model)
    }

    return functionals[params['Functional']]

def setup_transform(
        params: exputils.BaseParameters,
        hopf: libhopf.HopfModel,
        prop: bv.BlockVector
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
    parameter_scales = {
        'emod': 1e4
    }

    transform = None
    if params['ParamOption'] == 'Stiffness':
        const_vals.pop('emod')

        scale = tfrm.Scale(
            hopf.res.prop, scale=parameter_scales
        )
        const_subset = tfrm.ConstantSubset(
             hopf.res.prop, const_vals=const_vals
        )
        transform = scale * const_subset
    elif params['ParamOption'] == 'Shape':
        const_vals.pop('umesh')

        scale = tfrm.Scale(
            hopf.res.prop, scale=parameter_scales
        )
        const_subset = tfrm.ConstantSubset(
             hopf.res.prop, const_vals=const_vals
        )
        transform = scale * const_subset
    elif params['ParamOption'] == 'TractionShape':
        const_vals.pop('umesh')

        K, NU = 1e1, 0.3

        ## Apply a fixed `DirichletBC` on the first facet at the origin
        # and a fixed y coordinate along the bottom surface
        residual = hopf.res.solid.residual
        def is_on_origin(facet):
            facet_points = [point for point in dfn.entities(facet, 0)]
            is_on_origins = [
                np.all(point.midpoint()[:] == np.zeros(3))
                for point in facet_points
            ]
            return np.any(is_on_origins)

        facets = [
            facet for facet in dfn.facets(residual.mesh())
            if (
                np.dot(facet.normal()[:], [1, 0, 0]) == 0
                and facet.midpoint()[1] == 0
                and is_on_origin(facet)
            )
        ]
        origin_facet = facets[0]

        mf = dfn.MeshFunction('size_t', residual.mesh(), 1, value=0)
        mf.set_value(origin_facet.index(), 1)

        fspace = residual.form['coeff.prop.umesh'].function_space()
        fixed_dis = dfn.Constant(residual.mesh().topology().dim()*[0.0])
        dirichlet_bc_first_facet = dfn.DirichletBC(fspace, fixed_dis, mf, 1)

        mf = residual.mesh_function('facet')
        mf_label_to_value = residual.mesh_function_label_to_value('facet')
        fixed_surface_ids = [mf_label_to_value[name] for name in residual.fixed_facet_labels]
        dirichlet_bcs_fixed_y = [
            dfn.DirichletBC(fspace.sub(1), dfn.Constant(0.0), mf, facet_val)
            for facet_val in fixed_surface_ids
        ]

        traction_shape = tfrm.TractionShape(
            hopf.res,
            lame_lambda=(3*K*NU)/(1+NU),
            lame_mu=(3*K*(1-2*NU))/(2*(1+NU)),
            dirichlet_bcs=[dirichlet_bc_first_facet]+ dirichlet_bcs_fixed_y
        )
        extract = tfrm.ExtractSubset(
            traction_shape.x, keys_to_extract=('tmesh',)
        )

        const_subset = tfrm.ConstantSubset(
            traction_shape.y,
            const_vals=const_vals
        )
        scale = tfrm.Scale(
            const_subset.y, scale=parameter_scales
        )
        transform = extract * traction_shape * scale * const_subset
        # transform = traction_shape * scale
    else:
        raise ValueError(f"Unknown 'ParamOption': {params['ParamOption']}")

    return transform, parameter_scales

def setup_reduced_functional(params: exputils.BaseParameters):
    """
    Return a reduced functional + additional stuff

    Parameters
    ----------
    params: exputils.BaseParameters
        The experiment/case parameters
    """
    ## Load the model and set model properties
    hopf, *_ = setup_dyna_model(params)

    _props = setup_props(params, hopf)

    ## Setup the linearization/initial guess parameters
    transform, scale = setup_transform(params, hopf, _props)

    param = transform.x.copy()
    for key, subvec in _props.items():
        if key in param:
            param[key][:] = subvec

    if 'tmesh' in param:
        param['tmesh'][:] = 0

    # Apply the scaling that's used for `Scale`
    if isinstance(transform, tfrm.TransformComposition):
        has_scale_transform = (
            any(isinstance(transform, tfrm.Scale) for transform in transform._transforms)
        )
    else:
        has_scale_transform = isinstance(transform, tfrm.Scale)

    if has_scale_transform:
        for key, val in scale.items():
            if key in param:
                param[key][:] = param.sub[key]/val

    prop = transform.apply(param)
    assert np.isclose(bv.norm(prop-_props), 0)

    ## Solve for the Hopf bifurcation
    xhopf_0 = hopf.state.copy()
    with warnings.catch_warnings() as _:
        warnings.filterwarnings('always')
        if params['BifParam'] == 'psub':
            bifparam_tol = 100
            bifparams = PSUBS
        elif params['BifParam'] == 'qsub':
            bifparam_tol = 10
            bifparams = np.linspace(0, 300, 11)
        else:
            raise ValueError("")

        xhopf_0[:] = libhopf.gen_xhopf_0(
            hopf.res, prop, hopf.E_MODE, bifparams,
            tol=bifparam_tol
        )

    newton_params = {
        'absolute_tolerance': 1e-10,
        'relative_tolerance': 1e-5,
        'maximum_iterations': 5
    }
    xhopf_n, info = libhopf.solve_hopf_by_newton(
        hopf, xhopf_0, prop, newton_params=newton_params
    )
    if info['status'] != 0:
        raise RuntimeError(
            f"Hopf solution at linearization point didn't converge with info: {info}"
            f"; this happened for the parameter set {params}"
        )
    hopf.set_state(xhopf_n)

    ## Load the functional/objective function and gradient
    func = setup_functional(params, hopf)

    redu_functional = libhopf.ReducedFunctional(
        func,
        libhopf.ReducedHopfModel(hopf, newton_params=newton_params)
    )
    return redu_functional, hopf, xhopf_n, param, transform


def make_exp_params(study_name: str):
    """
    Return an iterable of experiment parameters (a parametric study)

    Parameters
    ----------
    study_name: str
        The name of the parametric study
    """

    DEFAULT_PARAMS = ExpParamBasic({
        'MeshName': f'M5_CB_GA3_CL{CLSCALE:.2f}',
        'LayerType': 'discrete',
        'Ecov': 2.5*10*1e3,
        'Ebod': 2.5*10*1e3,
        'ParamOption': 'Stiffness',
        'Functional': 'OnsetPressure',
        'H': 1e-3,
        'EigTarget': 'LARGEST_MAGNITUDE',
        'SepPoint': 'separation-inf',
        'BifParam': 'psub'
    })

    emods = np.arange(2.5, 20, 2.5) * 10 * 1e3
    if study_name == 'none':
        return []
    elif study_name == 'test':
        # TODO: The flow rate driven model seems to have a lot of numerical error
        # and fails when using PETSc's LU solver
        params = [
            DEFAULT_PARAMS.substitute({
                'Functional': 'OnsetPressure',
                'MeshName': f'M5_CB_GA3_CL{0.5:.2f}',
                'LayerType': 'discrete',
                'Ecov': 6e4, 'Ebod': 6e4,
                'BifParam': 'psub'
            })
        ]
        return params
    elif study_name == 'test_traction_shape':
        params = [
            DEFAULT_PARAMS.substitute({
                'Functional': 'OnsetPressure',
                'MeshName': f'M5_CB_GA3_CL{0.5:.2f}',
                'LayerType': 'discrete',
                'Ecov': emod, 'Ebod': emod,
                'BifParam': 'psub',
                'ParamOption': 'TractionShape'
            })
            for emod in [6e4]
        ]
        return params
    elif study_name == 'test_shape':
        params = [
            DEFAULT_PARAMS.substitute({
                'Functional': 'OnsetPressure',
                'MeshName': f'M5_CB_GA3_CL{0.5:.2f}',
                'LayerType': 'discrete',
                'Ecov': 6e4, 'Ebod': 6e4,
                'BifParam': 'psub',
                'ParamOption': 'Shape',
                'H': h
            })
            for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
        ]
        return params
    elif study_name == 'main_traction_shape':
        functional_names = [
            'OnsetPressure'
        ]
        emod_covs = np.concatenate([
            (  1)*np.arange(2, 18, 4),
            (1/2)*np.arange(2, 18, 4),
            (1/3)*np.arange(2, 18, 4),
            (1/4)*np.arange(2, 18, 4),
        ]) * 10 * 1e3
        emod_bods = np.concatenate([
            1*np.arange(2, 18, 4),
            1*np.arange(2, 18, 4),
            1*np.arange(2, 18, 4),
            1*np.arange(2, 18, 4)
        ]) * 10 * 1e3

        # emod_covs = np.array([6, 3]) * 10 * 1e3
        # emod_bods = np.array([6, 6]) * 10 * 1e3

        assert len(emod_covs) == len(emod_bods)

        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]
        paramss = (
            DEFAULT_PARAMS.substitute({
                'Functional': func_name,
                'Ecov': emod_cov,
                'Ebod': emod_bod,
                'MeshName': f'M5_CB_GA3_CL{0.5:.2f}',
                'ParamOption': 'TractionShape'
            })
            for func_name, (emod_cov, emod_bod)
            in itertools.product(functional_names, emods)
        )
        return paramss
    elif study_name == 'main_sensitivity':
        functional_names = [
            'OnsetPressure',
            'OnsetFrequency'
        ]
        param_options = [
            'const_shape'
        ]
        emod_covs = np.concatenate([
            (  1)*np.arange(2, 18, 4),
            (1/2)*np.arange(2, 18, 4),
            (1/3)*np.arange(2, 18, 4),
            (1/4)*np.arange(2, 18, 4),
        ]) * 10 * 1e3
        emod_bods = np.concatenate([
            1*np.arange(2, 18, 4),
            1*np.arange(2, 18, 4),
            1*np.arange(2, 18, 4),
            1*np.arange(2, 18, 4)
        ]) * 10 * 1e3

        # emod_covs = np.array([6, 3]) * 10 * 1e3
        # emod_bods = np.array([6, 6]) * 10 * 1e3

        assert len(emod_covs) == len(emod_bods)

        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]
        paramss = (
            DEFAULT_PARAMS.substitute({
                'Functional': func_name,
                'Ecov': emod_cov,
                'Ebod': emod_bod,
                'ParamOption': param_option
            })
            for func_name, (emod_cov, emod_bod), param_option
            in itertools.product(functional_names, emods, param_options)
        )
        return paramss
    elif study_name == 'main_sensitivity_flowdriven':
        functional_names = [
            'OnsetFlowRate',
            'OnsetFrequency',
            'SubglottalPressure',
            'SubglottalFlowRate'
        ]
        param_options = [
            'const_shape'
        ]
        emod_covs = np.array([6.0e4])
        emod_bods = np.array([6.0e4])

        assert len(emod_covs) == len(emod_bods)

        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]
        paramss = (
            DEFAULT_PARAMS.substitute({
                'Functional': func_name,
                'Ecov': emod_cov,
                'Ebod': emod_bod,
                'ParamOption': param_option,
                'BifParam': 'qsub'
            })
            for func_name, (emod_cov, emod_bod), param_option
            in itertools.product(functional_names, emods, param_options)
        )
        return paramss
    elif study_name == 'main_coarse_sensitivity':
        functional_names = [
            'OnsetPressure',
            'OnsetFrequency'
        ]
        param_options = [
            'const_shape'
        ]
        emod_covs = np.concatenate([
            (  1)*np.arange(2, 18, 4),
            # (1/2)*np.arange(2, 18, 4),
            (1/3)*np.arange(2, 18, 4),
            (1/4)*np.arange(2, 18, 4),
        ]) * 10 * 1e3
        emod_bods = np.concatenate([
            1*np.arange(2, 18, 4),
            # 1*np.arange(2, 18, 4),
            1*np.arange(2, 18, 4),
            1*np.arange(2, 18, 4)
        ]) * 10 * 1e3

        assert len(emod_covs) == len(emod_bods)

        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]
        paramss = (
            DEFAULT_PARAMS.substitute({
                'Functional': func_name,
                'Ecov': emod_cov,
                'Ebod': emod_bod,
                'ParamOption': param_option
            })
            for func_name, (emod_cov, emod_bod), param_option
            in itertools.product(functional_names, emods, param_options)
        )
        return paramss
    elif study_name == 'test_sensitivity':
        functional_names = [
            'OnsetPressure',
            'OnsetFrequency'
        ]
        param_options = [
            'const_shape'
        ]
        emod_covs = 1e4 * np.array([2])
        emod_bods = 1e4 * np.array([2])

        assert len(emod_covs) == len(emod_bods)

        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]
        paramss = (
            DEFAULT_PARAMS.substitute({
                'Functional': func_name,
                'Ecov': emod_cov,
                'Ebod': emod_bod,
                'ParamOption': param_option
            })
            for func_name, (emod_cov, emod_bod), param_option
            in itertools.product(functional_names, emods, param_options)
        )
        return paramss
    elif study_name == 'separation_effect':
        functional_names = [
            'OnsetPressure'
        ]
        param_options = [
            'const_shape'
        ]
        eig_targets = [
            'LARGEST_MAGNITUDE'
        ]
        layer_types = [
            'discrete'#, 'linear'
        ]

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

        mesh_names = [
            f'M5_CB_GA3_CL{clscale:.2f}_split6' for clscale in clscales
        ]
        sep_points = [
            # 'separation-inf',
            'sep1', 'sep2', 'sep3', 'sep4'
        ]
        paramss = (
            DEFAULT_PARAMS.substitute({
                'MeshName': mesh_name,
                'LayerType': layer_type,
                'Functional': func_name,
                'Ecov': emod_cov,
                'Ebod': emod_bod,
                'ParamOption': param_option,
                'H': h,
                'EigTarget': eig_target,
                'SepPoint': sep_point
            })
            for mesh_name, layer_type, h, func_name, (emod_cov, emod_bod), param_option, eig_target, sep_point
            in itertools.product(
                mesh_names,
                layer_types,
                hs,
                functional_names,
                emods,
                param_options,
                eig_targets,
                sep_points
            )
        )
        return paramss
    elif study_name == 'independence':
        functional_names = [
            'OnsetPressure'
        ]

        param_options = [
            'const_shape'
        ]

        eig_targets = [
            'LARGEST_MAGNITUDE'
        ]

        emod_covs = 1e4 * np.array([6, 2])
        emod_bods = 1e4 * np.array([6, 6])
        assert len(emod_covs) == len(emod_bods)
        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]

        hs = np.array([1e-2, 1e-3, 1e-4])
        mesh_names = [
            f'M5_CB_GA3_CL{clscale:.2f}' for clscale in (1, 0.5, 0.25)
        ]
        paramss = (
            DEFAULT_PARAMS.substitute({
                'MeshName': mesh_name,
                'Functional': func_name,
                'Ecov': emod_cov,
                'Ebod': emod_bod,
                'ParamOption': param_option,
                'H': h,
                'EigTarget': eig_target
            })
            for mesh_name, h, func_name, (emod_cov, emod_bod), param_option, eig_target
            in itertools.product(
                mesh_names,
                hs,
                functional_names,
                emods,
                param_options,
                eig_targets
            )
        )
        return paramss
    elif study_name == 'eig_target_effect':
        functional_names = [
            'OnsetPressure'
        ]

        param_options = [
            'const_shape'
        ]

        eig_targets = [
            'LARGEST_REAL'
        ]

        emod_covs = 1e4 * np.array([2])
        emod_bods = 1e4 * np.array([6])
        assert len(emod_covs) == len(emod_bods)
        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]

        hs = np.array([1e-3])
        mesh_names = [
            f'M5_CB_GA3_CL{clscale:.2f}' for clscale in (0.5, 0.25, 0.125)
        ]
        paramss = (
            DEFAULT_PARAMS.substitute({
                'MeshName': mesh_name,
                'Functional': func_name,
                'Ecov': emod_cov,
                'Ebod': emod_bod,
                'ParamOption': param_option,
                'H': h,
                'EigTarget': eig_target
            })
            for mesh_name, h, func_name, (emod_cov, emod_bod), param_option, eig_target
            in itertools.product(
                mesh_names,
                hs,
                functional_names,
                emods,
                param_options,
                eig_targets
            )
        )
        return paramss
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
    M_EMOD = dfn.assemble(dfn.inner(u, v)*dx, tensor=dfn.PETScMatrix()).mat()

    # Mass matrix for the space of mesh deformations:
    u = dfn.TrialFunction(form['coeff.prop.umesh'].function_space())
    v = dfn.TestFunction(form['coeff.prop.umesh'].function_space())
    M_UMESH = dfn.assemble(dfn.inner(u, v)*dx, tensor=dfn.PETScMatrix()).mat()

    def norm(x):
        """
        Return a scaled norm

        This norm roughly accounts for the difference in scale between
        modulus, shape changes, etc.
        """
        xs = x/scale
        dxs = xs.copy()
        dxs['emod'] = M_EMOD * xs.sub['emod']
        dxs['umesh'] = M_UMESH * xs.sub['umesh']
        return bla.dot(x, dxs) ** 0.5

    return norm


def run_functional_sensitivity(
        params: exputils.BaseParameters,
        output_dir='out/sensitivity',
        overwrite=False
    ):
    """
    Run an experiment where the sensitivity of a functional is saved
    """
    rfunc, hopf, xhopf_n, p0, transform = setup_reduced_functional(params)
    fpath = path.join(output_dir, params.to_str()+'.h5')
    if not path.isfile(fpath) or overwrite:
        ## Compute 1st order sensitivity of the functional
        rfunc.set_prop(transform.apply(p0))

        # DEBUG:
        # breakpoint()

        grad_props = rfunc.assem_dg_dprop()
        grad_params = transform.apply_vjp(p0, grad_props)

        ## Compute 2nd order sensitivity of the functional
        norm = make_norm(hopf)
        redu_hess_context = libhopf.ReducedFunctionalHessianContext(
            rfunc, transform, norm=norm, step_size=params['H']
        )
        redu_hess_context.set_params(p0)
        mat = PETSc.Mat().createPython(p0.mshape*2)
        mat.setPythonContext(redu_hess_context)
        mat.setUp()

        eps = SLEPc.EPS().create()
        eps.setOperators(mat)
        eps.setDimensions(5, 25)
        eps.setProblemType(SLEPc.EPS.ProblemType.HEP)

        if params['EigTarget'] == 'LARGEST_MAGNITUDE':
            which_eig = SLEPc.EPS.Which.LARGEST_MAGNITUDE
        elif params['EigTarget'] == 'LARGEST_REAL':
            which_eig = SLEPc.EPS.Which.LARGEST_REAL
        else:
            raise ValueError(f"Unknown `params['EigTarget']` key {params['EigTarget']}")
        eps.setWhichEigenpairs(which_eig)
        eps.setUp()

        eps.solve()

        neig = eps.getConverged()
        eigvecs = []
        eigvals = []
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
            for (key, vec) in zip(
                    ['grad_props', 'grad_param'],
                    [grad_props, grad_params]
                ):
                h5utils.create_resizable_block_vector_group(
                    f.require_group(key), vec.labels, vec.bshape,
                    dataset_kwargs={'dtype': 'f8'}
                )
                h5utils.append_block_vector_to_group(
                    f[key], vec
                )

            ## Write out the hessian eigenvalues and vectors
            f.create_dataset('eigvals', data=np.array(eigvals).real)
            for (key, vecs) in zip(
                    ['hess_param'], [eigvecs]
                ):
                if len(vecs) > 0:
                    h5utils.create_resizable_block_vector_group(
                        f.require_group(key), vecs[0].labels, vecs[0].bshape,
                        dataset_kwargs={'dtype': 'f8'}
                    )
                    for vec in vecs:
                        h5utils.append_block_vector_to_group(
                            f[key], vec
                        )

            ## Write out the state + properties vectors
            for (key, vec) in zip(
                    ['state', 'prop', 'param'],
                    [hopf.state, hopf.prop, p0]
                ):
                h5utils.create_resizable_block_vector_group(
                    f.require_group(key), vec.labels, vec.bshape,
                    dataset_kwargs={'dtype': 'f8'}
                )
                h5utils.append_block_vector_to_group(f[key], vec)
    else:
        print(f"Skipping existing file '{fpath}'")


if __name__ == '__main__':
    # Load the Hopf system

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--study-name', type=str, default='none')
    argparser.add_argument('--num-proc', type=int, default=1)
    argparser.add_argument('--clscale', type=float, default=1.0)
    argparser.add_argument('--overwrite', action='store_true')
    argparser.add_argument('--output-dir', type=str, default='out')
    clargs = argparser.parse_args()
    CLSCALE = clargs.clscale

    params = make_exp_params(clargs.study_name)

    def run(param_dict):
        # TODO: Note the conversion of parameter dictionary to
        # `Parameter` object here is hard-coded; you'll have to change this
        # in the future if you change the study types.
        param = ExpParamBasic(param_dict)
        return run_functional_sensitivity(
            param, output_dir=clargs.output_dir, overwrite=clargs.overwrite
        )

    if clargs.num_proc == 1:
        for param in tqdm(params):
            run(param)
    else:
        with mp.Pool(clargs.num_proc) as pool:
            pool.map(run, [p.data for p in params])
