"""
Solve a simple test optimization problem
"""

import argparse
import os.path as path
from pprint import pprint
import itertools
from typing import Union
import math

import h5py
import numpy as np
from scipy import optimize
from blockarray import blockvec  as bvec
from femvf import load
from femvf.models.transient import (
    solid as tsld, fluid as tfld, coupled as tcpl, base as tbase
)
from femvf.models.dynamical import (
    solid as dsld, fluid as dfld, coupled as dcpl, base as dbase
)
from femvf.meshutils import process_celllabel_to_dofs_from_forms

import libsetup
import libhopf
import libfunctionals as libfuncs

import exputils

ptypes = {
    'Name': str,
    'omega': float,
    'beta': float
}
FrequencyConstrainedFuncParam = exputils.make_parameters(ptypes)

ptypes = {
    'MeshName': str,
    'Ecov': float,
    'Ebod': float,
    'Functional': FrequencyConstrainedFuncParam
}
ExpParam = exputils.make_parameters(ptypes)

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
    Return the functional
    """
    func_onset_pressure = libfuncs.OnsetPressureFunctional(model)
    func_onset_frequency = libfuncs.AbsOnsetFrequencyFunctional(model)
    if isinstance(params['Functional'], str):
        func_name = params['Functional']
        if func_name == 'OnsetPressure':
            func = func_onset_pressure
        elif func_name == 'OnsetFrequency':
            func = func_onset_frequency
        else:
            raise ValueError("Unknown functional '{func_name}'")
    elif isinstance(params['Functional'], FrequencyConstrainedFuncParam):
        func_params = params['Functional']
        omega = func_params['omega']
        beta = func_params['beta']
        if func_params['Name'] == 'OnsetPressure':
            func = (
                func_onset_pressure
                + beta*(func_onset_frequency-omega)**2
            )
        else:
            raise ValueError("Unknown functional '{func_name}'")
    else:
        raise ValueError(f"Unknown functional type {type(params['Functional'])}")

    return func


def get_params(study_name: str):
    """
    Return an iterable of parameters for a given study name
    """

    DEFAULT_PARAMS = ExpParam({
        'MeshName': 'M5_CB_GA3',
        'Ecov': 2.5*10*1e3,
        'Ebod': 2.5*10*1e3,
        'Functional': {
            'Name': 'OnsetPressure',
            'omega': np.nan,
            'beta': 1000
        }
    })
    if study_name == 'none':
        return []
    elif study_name == 'test':
        return [DEFAULT_PARAMS]
    elif study_name == 'main_optimization':
        emods = np.arange(2.5, 20, 2.5) * 10 * 1e3
        paramss = (
            DEFAULT_PARAMS.substitute({'Ecov': emod, 'Ebod': emod})
            for emod in emods
        )
        return paramss
    else:
        raise ValueError("Unknown `study_name` '{study_name}'")

def run_minimize_functional(params, output_dir='out'):
    """
    Run an experiment where a functional is minimized
    """
    ## Load the model and set model properties
    hopf, *_ = get_dyna_model(params)

    props = get_props(hopf, params)
    hopf.set_props(props)

    ## Solve for the Hopf bifurcation
    xhopf_0 = libhopf.gen_hopf_initial_guess(hopf, PSUBS, tol=100.0)
    xhopf_n, info = libhopf.solve_hopf_newton(hopf, xhopf_0)
    hopf.set_state(xhopf_n)

    ## Load the functional/objective function and gradient
    # params = params.substitute(
    #     {'Functional/omega': xhopf_n['omega'][0]}
    # )
    params = params.substitute({
        'Functional': {
            'Name': params['Functional']['Name'],
            'omega': abs(xhopf_n['omega'][0]),
            'beta': 1000.0
        }
    })
    func = get_functional(hopf, params)

    redu_grad = libhopf.ReducedGradient(func, hopf)

    ## Run the minimizer
    # Set the initial guess
    x0 = bvec.concatenate_vec([hopf.props.copy(), redu_grad.camp.copy()])
    x0[-2:] = 0.0 # these correspond to complex amp. parameters

    # Set optimizer options/callback
    opt_options = {
        'disp': 99,
        'maxiter': 150,
        'ftol': 0.0,
        # 'maxls': 100
    }
    def opt_callback(xk):
        print("In callback")

    fpath = path.join(output_dir, params.to_str()+'.h5')
    if not path.isfile(fpath):
        with h5py.File(fpath, mode='w') as f:
            grad_manager = libhopf.OptGradManager(redu_grad, f)
            opt_res = optimize.minimize(
                grad_manager.grad, x0.to_mono_ndarray(),
                method='L-BFGS-B',
                jac=True,
                options=opt_options,
                callback=opt_callback
            )
        pprint(opt_res)
    else:
        print(f"Skipping existing file '{fpath}'")

def run_functional_sensitivity(params, output_dir='out'):
    """
    Run an experiment where the sensitivity of a functional is saved
    """
    raise NotImplementedError()

if __name__ == '__main__':
    # Load the Hopf system

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--study-name', type=str, default='none')
    clargs = argparser.parse_args()

    paramss = get_params(clargs.study_name)
    for params in paramss:
        run_minimize_functional(params, output_dir='out')
