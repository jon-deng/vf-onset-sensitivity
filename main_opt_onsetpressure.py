"""
Solve a simple test optimization problem
"""

import os.path as path
from pprint import pprint
import itertools

import h5py
import numpy as np
from scipy import optimize
from blockarray import blockvec  as bvec
from femvf.meshutils import process_celllabel_to_dofs_from_forms

import libsetup
import libhopf
import libfunctionals as libfuncs

import exputils

ptypes = {
    'MeshName': str,
    'Ecov': float,
    'Ebod': float,
    'alpha': float,
    'FunctionalType': str
}
ExpParam = exputils.make_parameters(ptypes)

PSUBS = np.arange(0, 1500, 50) * 10

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

def run_minimize_onset_pressure(fpath, hopf, emod, alpha):
    """
    Run the onset pressure minimization experiment
    """
    # Set the homogenous cover/body moduli and any constant properties
    region_to_dofs = process_celllabel_to_dofs_from_forms(
        hopf.res.solid.forms, hopf.res.solid.forms['fspace.scalar'].dofmap()
    )
    set_props(hopf.props, hopf, region_to_dofs, emod, emod)
    hopf.set_props(hopf.props)

    # Solve for the Hopf bifurcation
    xhopf_0 = libhopf.gen_hopf_initial_guess(hopf, PSUBS, tol=100.0)
    xhopf_n, info = libhopf.solve_hopf_newton(hopf, xhopf_0)
    hopf.set_state(xhopf_n)

    # Create the onset pressure functional
    func_onset_pressure = libfuncs.OnsetPressureFunctional(hopf)
    func_onset_frequency = libfuncs.AbsOnsetFrequencyFunctional(hopf)
    func_egrad_norm = libfuncs.ModulusGradientNormSqr(hopf)

    func = (
        func_onset_pressure
        + 1000.0*(func_onset_frequency-float(abs(xhopf_n['omega'][0])))**2
        + alpha*func_egrad_norm
    )

    redu_grad = libhopf.ReducedGradient(func, hopf)

    # Create the objective function and gradient needed for optimization
    x0 = bvec.concatenate_vec([hopf.props.copy(), redu_grad.camp.copy()])
    x0[-2:] = 0.0 # these correspond to complex amp. parameters

    ## Optimization
    opt_options = {
        'disp': 99,
        'maxiter': 150,
        'ftol': 0.0,
        # 'maxls': 100
    }
    def opt_callback(xk):
        print("In callback")

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
        print(f"File {fpath} already exists.")

if __name__ == '__main__':
    # Load the Hopf system
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_name = 'M5_CB_GA3'
    mesh_path = path.join('./mesh', mesh_name+'.msh')

    hopf, res, dres = libsetup.load_hopf_model(
        mesh_path, sep_method='fixed', sep_vert_label='separation-inf'
    )

    # `alpha` is a weight for the first-order smoothing term of elastic modulus
    # alphas = 10**np.array([-np.inf])
    # A characteristic gradient of 1kPa/1cm over a volume of 1cm^2 has a
    # functional value of 1e8 Ba^2*cm^2 (cgs unit)
    # 'unit' value of alpha should then start around 1e-8
    # for alpha in alphas:
    alphas = 10**np.array([-np.inf]+[-10, -8, -6, -4])

    demod = 2.5
    emods = np.arange(2.5, 20+demod/2, demod)*10*1e3
    # emods = np.array([5.0]) * 10 * 1e3

    for emod, alpha in itertools.product(emods, alphas):
        fpath = f"out/minimize_onset_pressure/opt_hist_emod{emod:.2e}_alpha{alpha:.2e}.h5"
        run_minimize_onset_pressure(fpath, hopf, emod, alpha)
