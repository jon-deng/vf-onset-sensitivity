"""
Solve a simple test optimization problem
"""

import argparse
import os.path as path
from pprint import pprint
import itertools
from typing import Union

import h5py
import numpy as np
from scipy import optimize
from blockarray import blockvec as bvec, h5utils
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

# NOTE: Import `dolfin` after `scipy.optimize` is important!
# Importing it after seems to lead to segfaults!
import dolfin as dfn
dfn.set_log_level(50)
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

PSUBS = np.arange(0, 1500, 50) * 10

def get_dyna_model(params: exputils.BaseParameters):
    """
    Return the model corresponding to parameters
    """
    mesh_name = params['MeshName']
    mesh_path = path.join('./mesh', mesh_name+'.msh')

    hopf, res, dres = libsetup.load_hopf_model(
        mesh_path, sep_method='fixed', sep_vert_label='separation-inf'
    )
    return hopf, res, dres

def get_tran_model(params: exputils.BaseParameters):
    """
    Return the model corresponding to parameters
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

def get_props(
        model: Union[tbase.BaseTransientModel, dbase.BaseDynamicalModel],
        params: exputils.BaseParameters
    ):
    """
    Return the properties vector
    """
    props = model.props.copy()

    region_to_dofs = process_celllabel_to_dofs_from_forms(
        model.res.solid.forms, model.res.solid.forms['fspace.scalar_dg0'].dofmap()
    )

    props = set_props(props, model, region_to_dofs, params['Ecov'], params['Ebod'])
    return props

def set_props(props, hopf, celllabel_to_dofs, emod_cov, emod_bod):
    # Set any constant properties
    props = libsetup.set_default_props(props, hopf.res.solid.forms['mesh.mesh'])

    # Set cover and body layer properties
    dofs_cov = np.array(celllabel_to_dofs['cover'], dtype=np.int32)
    dofs_bod = np.array(celllabel_to_dofs['body'], dtype=np.int32)
    dofs_share = set(dofs_cov) & set(dofs_bod)
    dofs_share = np.array(list(dofs_share), dtype=np.int32)

    props['emod'][dofs_cov] = emod_cov
    props['emod'][dofs_bod] = emod_bod
    props['emod'][dofs_share] = 1/2*(emod_cov + emod_bod)
    return props

def get_functional(
        model: dbase.BaseDynamicalModel,
        params: exputils.BaseParameters
    ):
    """
    Return the specified functional
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


def get_exp_params(study_name: str):
    """
    Return an iterable of parameters for a given study name
    """

    DEFAULT_PARAMS_PENALTY = ExpParamFreqPenalty({
        'MeshName': 'M5_CB_GA3',
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
        'MeshName': 'M5_CB_GA3',
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
        param_options = ['all', 'const_shape']

        paramss = (
            DEFAULT_PARAMS_PENALTY.substitute({
                'Functional/Name': func_name,
                'Ecov': emod,
                'Ebod': emod,
                'ParamOption': param_opt
            })
            for func_name, emod, param_opt
            in itertools.product(functional_names, emods, param_options)
        )
        return paramss
    elif study_name == 'main_sensitivity':
        functional_names = [
            'OnsetPressure',
            'OnsetFrequency',
            'OnsetPressureStrainEnergy'
        ]
        paramss = (
            DEFAULT_PARAMS_BASIC.substitute({
                'Functional': func_name,
                'Ecov': emod,
                'Ebod': emod
            })
            for func_name, emod in itertools.product(functional_names, emods)
        )
        return paramss
    else:
        raise ValueError("Unknown `study_name` '{study_name}'")

def setup_redu_grad(params):
    ## Load the model and set model properties
    hopf, *_ = get_dyna_model(params)

    _props = get_props(hopf, params)

    if params['ParamOption'] == 'const_shape':
        parameterization = pzn.ConstantSubset(
            hopf.res, const_vals={'umesh': 0}
        )
    elif params['ParamOption'] == 'all':
        parameterization = pzn.TractionShape(
            hopf.res, lame_lambda=1e2, lame_mu=1e2
        )
    else:
        raise ValueError(f"Unknown 'ParamOption': {params['ParamOption']}")
    p = parameterization.x.copy()
    for key, subvec in _props.items():
        if key in p:
            p[key][:] = subvec

    hopf.set_props(parameterization.apply(p))

    ## Solve for the Hopf bifurcation
    xhopf_0 = libhopf.gen_hopf_initial_guess(hopf, PSUBS, tol=100.0)
    xhopf_n, info = libhopf.solve_hopf_newton(hopf, xhopf_0)
    hopf.set_state(xhopf_n)

    ## Load the functional/objective function and gradient
    # params = params.substitute(
    #     {'Functional/omega': xhopf_n['omega'][0]}
    # )
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
    func = get_functional(hopf, _params)

    redu_grad = libhopf.ReducedGradient(func, hopf)
    return redu_grad, hopf, xhopf_n, p, parameterization

def run_minimize_functional(params, output_dir='out/minimization'):
    """
    Run an experiment where a functional is minimized
    """
    redu_grad, hopf, xhopf_n, p0, parameterization = setup_redu_grad(params)

    ## Run the minimizer
    # Set optimizer options/callback
    opt_options = {
        'disp': 99,
        'maxiter': 150,
        'ftol': 0.0,
        'maxcor': 50
        # 'maxls': 100
    }
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
                options=opt_options,
                callback=opt_callback
            )
        pprint(opt_res)
    else:
        print(f"Skipping existing file '{fpath}'")

def run_functional_sensitivity(params, output_dir='out/sensitivity'):
    """
    Run an experiment where the sensitivity of a functional is saved
    """
    redu_grad, hopf, xhopf_n, p0, parameterization = setup_redu_grad(params)
    dprops = redu_grad.assem_dg_dprops()
    dparam = parameterization.apply_vjp(p0, dprops)

    ## Compute the sensitivity of the functional to properties
    fpath = path.join(output_dir, params.to_str()+'.h5')
    if not path.isfile(fpath):
        with h5py.File(fpath, mode='w') as f:
            ## Write out the gradient vectors
            for (key, vec) in zip(['dprops', 'dparam'], [dprops, dparam]):
                h5utils.create_resizable_block_vector_group(
                    f.require_group(key), vec.labels, vec.bshape
                )
                h5utils.append_block_vector_to_group(
                    f[key], vec
                )

            ## Write out the state, control, properties vector
            for (key, vec) in zip(['state', 'props'], [hopf.state, hopf.props]):
                h5utils.create_resizable_block_vector_group(
                    f.require_group(key), vec.labels, vec.bshape
                )
                h5utils.append_block_vector_to_group(
                    f[key], vec
                )
    else:
        print(f"Skipping existing file '{fpath}'")


if __name__ == '__main__':
    # Load the Hopf system

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--study-name', type=str, default='none')
    argparser.add_argument('--study-type', type=str, default='none')
    clargs = argparser.parse_args()

    paramss = get_exp_params(clargs.study_name)
    if clargs.study_type == 'none':
        pass
    elif clargs.study_type == 'sensitivity':
        for params in paramss:
            run_functional_sensitivity(params, output_dir='out/sensitivity')
    elif clargs.study_type == 'minimization':
        for params in paramss:
            run_minimize_functional(params, output_dir='out/minimization')
    else:
        raise ValueError(f"Unknown '--study-type' {clargs.study_type}")
