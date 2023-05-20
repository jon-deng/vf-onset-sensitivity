"""
Conduct a sensitivity study of onset pressure and frequency
"""

import argparse
import os.path as path
import multiprocessing as mp
from pprint import pprint
import itertools
import warnings
from tqdm import tqdm
from typing import Union

import h5py
import numpy as np
from scipy import optimize
from petsc4py import PETSc
from slepc4py import SLEPc
# NOTE: Importing `dolfin` after `scipy.optimize` is important!
# Importing it after seems to cause segfaults for some reason
import dolfin as dfn
dfn.set_log_level(50)

from blockarray import blockvec as bvec, linalg as bla, h5utils
from femvf import load
from femvf.models.transient import (
    solid as tsld, fluid as tfld, base as tbase
)
from femvf.models.dynamical import (
    base as dbase
)
from femvf.parameters import parameterization as pzn
from femvf.meshutils import process_celllabel_to_dofs_from_residual

import libsetup
import libhopf
import libfunctionals as libfuncs

import exputils

# pylint: disable=redefined-outer-name

ptypes = {
    'MeshName': str,
    'LayerType': str,
    'Ecov': float,
    'Ebod': float,
    'ParamOption': str,
    'Functional': str,
    'H': float,
    'EigTarget': str,
    'SepPoint': str
}
ExpParamBasic = exputils.make_parameters(ptypes)

PSUBS = np.arange(0, 800, 50) * 10

def setup_dyna_model(params: exputils.BaseParameters):
    """
    Return a dynamical model
    """
    mesh_name = params['MeshName']
    mesh_path = path.join('./mesh', mesh_name+'.msh')

    hopf, res, dres = libsetup.load_hopf_model(
        mesh_path, sep_method='fixed', sep_vert_label=params['SepPoint']
    )
    return hopf, res, dres

def setup_tran_model(params: exputils.BaseParameters):
    """
    Return a transient model
    """
    mesh_name = params['MeshName']
    mesh_path = path.join('./mesh', mesh_name+'.msh')

    model = load.load_transient_fsi_model(
        mesh_path, None,
        SolidType=tsld.KelvinVoigt,
        FluidType=tfld.BernoulliFixedSep,
        separation_vertex_label='sep-inf'
    )
    return model

def setup_props(
        params: exputils.BaseParameters,
        model: Union[tbase.BaseTransientModel, dbase.BaseDynamicalModel]
    ):
    """
    Return a properties vector
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

def set_prop(prop, hopf, celllabel_to_dofs, emod_cov, emod_bod, layer_type='discrete'):
    # Set any constant properties
    prop = libsetup.set_default_props(prop, hopf.res.solid.residual.mesh())

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
        coord = hopf.res.solid.residual.form['coeff.prop.emod'].function_space().tabulate_dof_coordinates()
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

    prop['ncontact'][:] = [0, 1]
    return prop

def setup_functional(
        params: exputils.BaseParameters,
        model: dbase.BaseDynamicalModel
    ):
    """
    Return a functional
    """
    functionals = {
        'OnsetPressure': libfuncs.OnsetPressureFunctional(model),
        'OnsetFrequency': libfuncs.AbsOnsetFrequencyFunctional(model)
    }

    return functionals[params['Functional']]

def setup_exp_params(study_name: str):
    """
    Return an iterable of parameters for a given study name
    """

    DEFAULT_PARAMS = ExpParamBasic({
        'MeshName': f'M5_CB_GA3_CL{CLSCALE:.2f}',
        'LayerType': 'discrete',
        'Ecov': 2.5*10*1e3,
        'Ebod': 2.5*10*1e3,
        'ParamOption': 'const_shape',
        'Functional': 'OnsetPressure',
        'H': 1e-3,
        'EigTarget': 'LARGEST_MAGNITUDE',
        'SepPoint': 'separation-inf'
    })

    emods = np.arange(2.5, 20, 2.5) * 10 * 1e3
    if study_name == 'none':
        return []
    elif study_name == 'test':
        params = [
            DEFAULT_PARAMS.substitute({
                'MeshName': f'M5_CB_GA3_CL{0.5:.2f}',
                'LayerType': 'discrete',
                'Ecov': (1/3) * 7e4, 'Ebod': 7e4
            })
        ]
        return params
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
            'LARGEST_MAGNITUDE',
            'LARGEST_REAL'
        ]

        emod_covs = 1e4 * np.array([6, 2])
        emod_bods = 1e4 * np.array([6, 6])
        assert len(emod_covs) == len(emod_bods)
        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]

        hs = np.array([1e-3, 1e-4])
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

def setup_parameterization(params, hopf, prop):
    """
    Return a parameterization
    """
    const_vals = {key: np.array(subvec) for key, subvec in prop.sub_items()}
    scale = {
        'emod': 1e4
    }

    parameterization = None
    if params['ParamOption'] == 'const_shape':
        const_vals.pop('emod')

        # scale = {'emod'}
        parameterization = pzn.ConstantSubset(
            hopf.res,
            const_vals=const_vals,
            scale=scale
        )
    else:
        raise ValueError(f"Unknown 'ParamOption': {params['ParamOption']}")

    return parameterization, scale

def setup_reduced_functional(params):
    """
    Return a reduced functional + additional stuff
    """
    ## Load the model and set model properties
    hopf, *_ = setup_dyna_model(params)

    _props = setup_props(params, hopf)

    ## Setup the linearization/initial guess parameters
    parameterization, scale = setup_parameterization(params, hopf, _props)

    p = parameterization.x.copy()
    for key, subvec in _props.items():
        if key in p:
            p[key][:] = subvec

    # Apply the scaling that's used for `ConstantSubset`
    if isinstance(parameterization, pzn.ConstantSubset):
        for key, val in scale.items():
            if key in p:
                p[key][:] = p.sub[key]/val

    prop = parameterization.apply(p)
    assert np.isclose(bvec.norm(prop-_props), 0)

    ## Solve for the Hopf bifurcation
    xhopf_0 = hopf.state.copy()
    with warnings.catch_warnings() as _:
        warnings.filterwarnings('always')
        xhopf_0[:] = libhopf.gen_xhopf_0(hopf.res, prop, hopf.E_MODE, PSUBS, tol=100.0)
    xhopf_n, info = libhopf.solve_hopf_by_newton(hopf, xhopf_0, prop)
    if info['status'] != 0:
        raise RuntimeError(
            f"Hopf solution at linearization point didn't converge with info: {info}"
            f"; this happened for the parameter set {params}"
        )
    else:
        hopf.set_state(xhopf_n)

    ## Load the functional/objective function and gradient
    func = setup_functional(params, hopf)

    redu_functional = libhopf.ReducedFunctional(
        func,
        libhopf.ReducedHopfModel(hopf)
    )
    return redu_functional, hopf, xhopf_n, p, parameterization

def setup_norm(hopf_model: libhopf.HopfModel):
    """
    Return a norm for scalar fields over a mesh

    The norm is used to ensure that a step size for finite differences (FD) is reasonable
    for different meshes. The norm should give a magnitude that is independent of mesh density.
    """

    scale = hopf_model.prop.copy()
    scale[:] = 1
    scale['emod'][:] = 1e4

    # Mass matrix for the space for elastic modulus
    form = hopf_model.res.solid.residual.form
    dx = hopf_model.res.solid.residual.measure('dx')
    u = dfn.TrialFunction(form['coeff.prop.emod'].function_space())
    v = dfn.TestFunction(form['coeff.prop.emod'].function_space())
    M_EMOD = dfn.assemble(dfn.inner(u, v)*dx, tensor=dfn.PETScMatrix()).mat()

    def norm(x):
        """
        Return a scaled norm

        This norm roughly accounts for the difference in scale between
        modulus, shape changes, etc.
        """
        xs = x/scale
        dxs = xs.copy()
        dxs['emod'] = M_EMOD * xs.sub['emod']
        return bla.dot(x, dxs) ** 0.5

    return norm

def run_functional_sensitivity(params, output_dir='out/sensitivity'):
    """
    Run an experiment where the sensitivity of a functional is saved
    """
    rfunc, hopf, xhopf_n, p0, parameterization = setup_reduced_functional(params)
    fpath = path.join(output_dir, params.to_str()+'.h5')
    if not path.isfile(fpath):
        ## Compute 1st order sensitivity of the functional
        rfunc.set_prop(parameterization.apply(p0))
        grad_props = rfunc.assem_dg_dprop()
        grad_params = parameterization.apply_vjp(p0, grad_props)

        ## Compute 2nd order sensitivity of the functional
        norm = setup_norm(hopf)
        redu_hess_context = libhopf.ReducedFunctionalHessianContext(
            rfunc, parameterization, norm=norm, step_size=params['H']
        )
        redu_hess_context.set_params(p0)
        mat = PETSc.Mat().createPython(p0.mshape*2)
        mat.setPythonContext(redu_hess_context)
        mat.setUp()

        eps = SLEPc.EPS().create()
        eps.setOperators(mat)
        eps.setDimensions(5, 20)
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
            eigvec = parameterization.x.copy()
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
    argparser.add_argument('--clscale', type=float, default=1)
    argparser.add_argument('--output-dir', type=str, default='out')
    clargs = argparser.parse_args()
    CLSCALE = clargs.clscale

    params = setup_exp_params(clargs.study_name)

    def run(param_dict):
        # TODO: Note the conversion of parameter dictionary to
        # `Parameter` object here is hard-coded; you'll have to change this
        # in the future if you change the study types.
        param = ExpParamBasic(param_dict)
        return run_functional_sensitivity(
            param, output_dir=clargs.output_dir
        )

    if clargs.num_proc == 1:
        for param in tqdm(params):
            run(param)
    else:
        with mp.Pool(clargs.num_proc) as pool:
            pool.map(run, [p.data for p in params])
