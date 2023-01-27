"""
Conduct a sensitivity/optimization study of onset pressure and frequency
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
# Importing it after seems to cause segfaults
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
from femvf.meshutils import process_celllabel_to_dofs_from_forms

import libsetup
import libhopf
import libfunctionals as libfuncs

import exputils

# pylint: disable=redefined-outer-name

ptypes = {
    'Name': str,
    'omega': float,
    'beta': float
}
FrequencyPenaltyFuncParam = exputils.make_parameters(ptypes)

ptypes = {
    'MeshName': str,
    'Ecov': float,
    'Ebod': float,
    'ParamOption': str,
    'Functional': FrequencyPenaltyFuncParam
}
ExpParamFreqPenalty = exputils.make_parameters(ptypes)

ptypes = {
    'MeshName': str,
    'Ecov': float,
    'Ebod': float,
    'ParamOption': str,
    'Functional': str
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
        mesh_path, sep_method='fixed', sep_vert_label='separation-inf'
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
    region_to_dofs = process_celllabel_to_dofs_from_forms(
        model.res.solid.forms,
        model.res.solid.forms['fspace.scalar_dg0'].dofmap()
    )
    prop = set_prop(
        prop, model, region_to_dofs,
        params['Ecov'], params['Ebod']
    )
    return prop

def set_prop(prop, hopf, celllabel_to_dofs, emod_cov, emod_bod):
    # Set any constant properties
    prop = libsetup.set_default_props(prop, hopf.res.solid.forms['mesh.mesh'])

    # Set cover and body layer properties
    dofs_cov = np.array(celllabel_to_dofs['cover'], dtype=np.int32)
    dofs_bod = np.array(celllabel_to_dofs['body'], dtype=np.int32)
    dofs_share = set(dofs_cov) & set(dofs_bod)
    dofs_share = np.array(list(dofs_share), dtype=np.int32)

    prop['emod'][dofs_cov] = emod_cov
    prop['emod'][dofs_bod] = emod_bod
    prop['emod'][dofs_share] = 1/2*(emod_cov + emod_bod)
    return prop

def setup_functional(
        params: exputils.BaseParameters,
        model: dbase.BaseDynamicalModel
    ):
    """
    Return a functional
    """
    func_onset_pressure = libfuncs.OnsetPressureFunctional(model)
    func_onset_frequency = libfuncs.AbsOnsetFrequencyFunctional(model)
    func_strain_energy = libfuncs.StrainEnergyFunctional(model)
    def get_named_functional(func_name):
        if func_name == 'OnsetPressure':
            func = func_onset_pressure
        elif func_name == 'OnsetFrequency':
            func = func_onset_frequency
        elif func_name == 'OnsetPressureStrainEnergy':
            func = func_strain_energy * func_onset_pressure
        else:
            raise ValueError("Unknown functional '{func_name}'")
        return func

    if isinstance(params['Functional'], str):
        func_name = params['Functional']
        func = get_named_functional(func_name)

    elif isinstance(params['Functional'], FrequencyPenaltyFuncParam):
        func_params = params['Functional']
        func_name = func_params['Name']
        omega = func_params['omega']
        beta = func_params['beta']

        func = get_named_functional(func_name)
        func = (
            func
            + beta*(func_onset_frequency-omega)**2
        )
    else:
        raise ValueError(f"Unknown functional type {type(params['Functional'])}")

    return func

def setup_opt_options(
        params: exputils.BaseParameters,
        parameterization
    ):
    """
    Return optimizer options
    """
    _lb = parameterization.x.copy()
    _ub = parameterization.x.copy()
    _lb[:] = -np.inf
    _ub[:] = np.inf
    # Add lower bound to emod
    _lb['emod'][:] = 0.5e3*10

    generic_options = {
        'bounds': np.stack([_lb.to_mono_ndarray(), _ub.to_mono_ndarray()], axis=-1)
    }
    specific_options = {
        'disp': 99,
        'maxiter': 150,
        'ftol': 0.0,
        'maxcor': 50
        # 'maxls': 100
    }
    return generic_options, specific_options

def setup_exp_params(study_name: str):
    """
    Return an iterable of parameters for a given study name
    """

    DEFAULT_PARAMS_PENALTY = ExpParamFreqPenalty({
        'MeshName': f'M5_CB_GA3_CL{CLSCALE:.2f}',
        'Ecov': 2.5*10*1e3,
        'Ebod': 2.5*10*1e3,
        'ParamOption': 'all',
        'Functional': {
            'Name': 'OnsetPressure',
            'omega': -1,
            'beta': 1000
        }
    })

    DEFAULT_PARAMS_BASIC = ExpParamBasic({
        'MeshName': f'M5_CB_GA3_CL{CLSCALE:.2f}',
        'Ecov': 2.5*10*1e3,
        'Ebod': 2.5*10*1e3,
        'ParamOption': 'all',
        'Functional': 'OnsetPressure'
    })

    emods = np.arange(2.5, 20, 2.5) * 10 * 1e3

    if study_name == 'none':
        return []
    elif study_name == 'test':
        return [DEFAULT_PARAMS_BASIC]
    elif study_name == 'main_minimization':
        functional_names = [
            'OnsetPressure',
            'OnsetPressureStrainEnergy'
        ]
        param_options = ['const_shape', 'traction_shape']

        paramss = (
            DEFAULT_PARAMS_PENALTY.substitute({
                'Functional/Name': func_name,
                'Ecov': emod,
                'Ebod': emod,
                'ParamOption': param_opt
            })
            for param_opt, func_name, emod
            in itertools.product(param_options, functional_names, emods)
        )
        return paramss
    elif study_name == 'main_sensitivity':
        functional_names = [
            'OnsetPressure',
            'OnsetFrequency',
            # 'OnsetPressureStrainEnergy'
        ]
        param_options = [
            'const_shape',
            # 'all'
        ]
        emod_covs = np.concatenate([
            2*np.arange(1, 10, 2),
            1*np.arange(1, 10, 2),
            2/3*np.arange(1, 10, 2),
            0.5*np.arange(1, 10, 2)
        ]) * 10 * 1e3
        emod_bods = np.concatenate([
            2*np.arange(1, 10, 2),
            2*np.arange(1, 10, 2),
            2*np.arange(1, 10, 2),
            2*np.arange(1, 10, 2)
        ]) * 10 * 1e3

        assert len(emod_covs) == len(emod_bods)

        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]
        paramss = (
            DEFAULT_PARAMS_BASIC.substitute({
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
            'OnsetPressure'
        ]
        param_options = [
            'const_shape'
        ]
        emod_covs = 1e4 * np.array([2])
        emod_bods = 1e4 * np.array([2])

        assert len(emod_covs) == len(emod_bods)

        emods = [(ecov, ebod) for ecov, ebod in zip(emod_covs, emod_bods)]
        paramss = (
            DEFAULT_PARAMS_BASIC.substitute({
                'Functional': func_name,
                'Ecov': emod_cov,
                'Ebod': emod_bod,
                'ParamOption': param_option
            })
            for func_name, (emod_cov, emod_bod), param_option
            in itertools.product(functional_names, emods, param_options)
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
        'emod': 1e4,
        'umesh': 1e-1
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
    elif params['ParamOption'] == 'all':
        const_vals.pop('emod')
        const_vals.pop('umesh')

        # scale = {'emod'}
        parameterization = pzn.ConstantSubset(
            hopf.res,
            const_vals=const_vals,
            scale=scale
        )
    elif params['ParamOption'] == 'traction_shape':
        parameterization = pzn.TractionShape(
            hopf.res, lame_lambda=1.0e4, lame_mu=1.0e4
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
            p[key][:] = p.sub[key]/val

    prop = parameterization.apply(p)
    assert np.isclose(bvec.norm(prop-_props), 0)

    ## Solve for the Hopf bifurcation
    xhopf_0 = hopf.state.copy()
    xhopf_0[:] = libhopf.gen_xhopf_0(hopf.res, prop, hopf.E_MODE, PSUBS, tol=100.0)
    xhopf_n, info = libhopf.solve_hopf_by_newton(hopf, xhopf_0, prop)
    hopf.set_state(xhopf_n)

    # Throw a warning if the Hopf system solution didn't converge
    if info['status'] == -1:
        warnings.warn(
            "Hopf system solution didn't converge with solver info: "
            f"{info}",
            RuntimeWarning
        )

    ## Load the functional/objective function and gradient
    _params = params.substitute({})
    if isinstance(params['Functional'], FrequencyPenaltyFuncParam):
        if params['Functional/omega'] == -1:
            _params = params.substitute({
                'Functional': {
                    'Name': params['Functional']['Name'],
                    'omega': abs(xhopf_n['omega'][0]),
                    'beta': 1000.0
                }
            })
    func = setup_functional(_params, hopf)

    redu_functional = libhopf.ReducedFunctional(
        func,
        libhopf.ReducedHopfModel(hopf)
    )
    return redu_functional, hopf, xhopf_n, p, parameterization

def run_minimize_functional(params, output_dir='out/minimization'):
    """
    Run an experiment where a functional is minimized
    """
    redu_grad, hopf, xhopf_n, p0, parameterization = setup_reduced_functional(params)

    ## Run the minimizer
    # Set optimizer options/callback
    gen_opts, spe_opts = setup_opt_options(params, parameterization)
    def opt_callback(xk):
        print("In callback")

    fpath = path.join(output_dir, params.to_str()+'.h5')
    if not path.isfile(fpath):
        with h5py.File(fpath, mode='w') as f:
            grad_manager = libhopf.OptGradManager(
                redu_grad, f, parameterization
            )
            opt_res = optimize.minimize(
                grad_manager.grad, p0.to_mono_ndarray(),
                method='L-BFGS-B',
                jac=True,
                **gen_opts,
                options=spe_opts,
                callback=opt_callback
            )
        pprint(opt_res)
    else:
        print(f"Skipping existing file '{fpath}'")

def setup_norm(hopf_model):
    """
    Return a norm for property vectors

    This norm should scale property vectors so that characteristic changes
    in the stiffness and shape properties (and others in general) are all
    scaled the same.
    """
    # NOTE: This is the old strategy
    # _scale = hopf_model.prop.copy()
    # _scale[:] = 1
    # _scale['umesh'][:] = 1e-1
    # _scale['emod'][:] = 1e4
    # def norm(x):
    #     return bvec.norm(x/_scale)

    scale = hopf_model.prop.copy()
    scale[:] = 1
    scale['emod'][:] = 1e4
    scale['umesh'][:] = 1e-6

    # Mass matrices for the different vector spaces
    forms = hopf_model.res.solid.forms
    import dolfin as dfn
    dx = forms['measure.dx']
    u = dfn.TrialFunction(forms['coeff.prop.emod'].function_space())
    v = dfn.TestFunction(forms['coeff.prop.emod'].function_space())
    M_EMOD = dfn.assemble(dfn.inner(u, v)*dx, tensor=dfn.PETScMatrix()).mat()

    # The `...[0]` is hard-coded because I did something weird with storing the
    # mesh displacement/shape property
    u = dfn.TrialFunction(forms['coeff.prop.umesh'][0].function_space())
    v = dfn.TestFunction(forms['coeff.prop.umesh'][0].function_space())
    M_SHAPE = dfn.assemble(dfn.inner(u, v)*dx, tensor=dfn.PETScMatrix()).mat()

    def norm(x):
        """
        Return a scaled norm

        This norm roughly accounts for the difference in scale between
        modulus, shape changes, etc.
        """
        xs = x/scale
        dxs = xs.copy()
        dxs['emod'] = M_EMOD * xs.sub['emod']
        dxs['umesh'] = M_SHAPE * xs.sub['umesh']
        return bla.dot(x, dxs)

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
            rfunc, parameterization, norm=norm, step_size=1e-2
        )
        redu_hess_context.set_params(p0)
        mat = PETSc.Mat().createPython(p0.mshape*2)
        mat.setPythonContext(redu_hess_context)
        mat.setUp()

        eps = SLEPc.EPS().create()
        eps.setOperators(mat)
        eps.setDimensions(5, 20)
        eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
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
    argparser.add_argument('--study-type', type=str, default='none')
    argparser.add_argument('--num-proc', type=int, default=1)
    argparser.add_argument('--clscale', type=float, default=1)
    clargs = argparser.parse_args()
    CLSCALE = clargs.clscale

    params = setup_exp_params(clargs.study_name)

    run = None
    if clargs.study_type == 'sensitivity':
        def run(param_dict):
            # TODO: Note the conversion of parameter dictionary to
            # `Parameter` object here is hard-coded; you'll have to change this
            # in the future if you change the study types.
            param = ExpParamBasic(param_dict)
            return run_functional_sensitivity(
                param, output_dir='out/sensitivity'
            )
    elif clargs.study_type == 'minimization':
        def run(param_dict):
            param = ExpParamFreqPenalty(param_dict)
            return run_minimize_functional(
                param, output_dir='out/minimization'
            )
    else:
        raise ValueError(f"Unknown '--study-type' {clargs.study_type}")

    if run is not None:
        if clargs.num_proc == 1:
            for param in tqdm(params):
                run(param)
        else:
            with mp.Pool(clargs.num_proc) as pool:
                pool.map(run, [p.data for p in params])
