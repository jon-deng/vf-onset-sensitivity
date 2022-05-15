"""
This script runs a linear stability analysis (LSA) for all the stress test cases

The stress test cases consist of all combinations of body and cover moduli where
the moduli range from 2.5 to 10 in steps of 2.5 (kPa).
"""

import argparse
import multiprocessing
import os.path as path
import itertools
from pprint import pprint

import petsc4py
import numpy as np
from scipy import optimize
import h5py
from femvf import meshutils, forward, statefile as sf
from femvf.signals import solid as sigsl
from vfsig import modal as modalsig
from blockarray import h5utils as bh5utils, blockvec as bv

import setup
import libhopf, libfunctionals as libfuncs
import postprocutils
# import h5utils

# pylint: disable=redefined-outer-name

# Range of psub to test for Hopf bifurcation
PSUBS = np.arange(100, 1500, 100)*10
EMODS = np.arange(2.5, 12.5+2.5, 2.5) * 1e3*10

# EMODS = np.array([2.0, 2.5, 4.5, 5.0, 7.0, 7.5,]) * 1e3*10

EMODS = np.arange(2.5, 10.5+0.5, 0.5) * 1e3*10
EMODS = np.arange(5.0, 7.0, 0.5) * 1e3 * 10
EMODS = np.array([2.5]) * 1e3*10

## The models are not pickalble so have to be outside for multi-processing
mesh_name = 'BC-dcov5.00e-02-cl1.00'
mesh_path = path.join('./mesh', mesh_name+'.xml')
RES_LAMP = setup.setup_transient_model(mesh_path)
RES_DYN, DRES_DYN = setup.setup_models(mesh_path)
RES_HOPF = libhopf.HopfModel(RES_DYN, DRES_DYN)

def set_props(props, celllabel_to_dofs, emod_cov, emod_bod):
    # Set any constant properties
    props = setup.set_constant_props(props, celllabel_to_dofs, RES_DYN)

    # Set cover and body layer properties

    dofs_cov = np.array(celllabel_to_dofs['cover'], dtype=np.int32)
    dofs_bod = np.array(celllabel_to_dofs['body'], dtype=np.int32)
    dofs_share = set(dofs_cov) & set(dofs_bod)
    dofs_share = np.array(list(dofs_share), dtype=np.int32)

    if hasattr(props['emod'], 'array'):
        props['emod'].array[dofs_cov] = emod_cov
        props['emod'].array[dofs_bod] = emod_bod
        props['emod'].array[dofs_share] = 1/2*(emod_cov + emod_bod)
    else:
        props['emod'][dofs_cov] = emod_cov
        props['emod'][dofs_bod] = emod_bod
        props['emod'][dofs_share] = 1/2*(emod_cov + emod_bod)

    return props

def _ignore_existing_path(func):
    def dec_func(fpath, *args, **kwargs):
        if not path.isfile(fpath):
            return func(fpath, *args, **kwargs)
        else:
            print(f"File {fpath} already exists.")
    return dec_func

@_ignore_existing_path
def run_lsa(fpath, emod_cov, emod_bod):
    # Get the cover/body layer DOFs
    _forms = RES_DYN.solid.forms
    celllabel_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(_forms, _forms['fspace.scalar'])
    props = set_props(RES_DYN.props, celllabel_to_dofs, emod_cov, emod_bod)
    RES_DYN.set_props(props)

    eigs_info = [libhopf.solve_least_stable_mode(RES_DYN, psub) for psub in PSUBS]

    omegas_real = [eiginfo[0].real for eiginfo in eigs_info]
    omegas_imag = [eiginfo[0].imag for eiginfo in eigs_info]
    eigvecs_real = [eiginfo[1] for eiginfo in eigs_info]
    eigvecs_imag = [eiginfo[2] for eiginfo in eigs_info]
    xfps = [eiginfo[3] for eiginfo in eigs_info]

    with h5py.File(fpath, mode='a') as f:
        f['psub'] = PSUBS
        f['omega_real'] = np.array(omegas_real)
        f['omega_imag'] = np.array(omegas_imag)

        for group_name, eigvecs in zip(
                ['eigvec_real', 'eigvec_imag', 'fixedpoint'],
                [eigvecs_real, eigvecs_imag, xfps]
            ):
            bh5utils.create_resizable_block_vector_group(
                f.require_group(group_name), RES_DYN.state.labels, RES_DYN.state.bshape
            )
            for eigvec in eigvecs:
                bh5utils.append_block_vector_to_group(f[group_name], eigvec)
    return PSUBS, omegas_real

@_ignore_existing_path
def run_solve_hopf(fpath, emod_cov, emod_bod):
    # Set the cover/body layer properties
    _forms = RES_HOPF.res.solid.forms
    celllabel_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(_forms, _forms['fspace.scalar'])
    props = set_props(RES_HOPF.props, celllabel_to_dofs, emod_cov, emod_bod)
    RES_HOPF.set_props(props)

    # Read the max real eigenvalue information from the LSA to determine if Hopf
    # bifurcations occur and a good starting point
    lsa_fname = f'LSA_ecov{emod_cov:.2e}_ebody{emod_bod:.2e}'
    lsa_fpath = f'out/stress_test/{lsa_fname}.h5'

    with h5py.File(lsa_fpath, mode='r') as f_lsa:
        omegas_real = f_lsa['omega_real'][:]

    is_hopf_bif = [(w2 > 0 and w1 <=0) for w1, w2 in zip(omegas_real[:-1], omegas_real[1:])]
    if is_hopf_bif.count(True) == 0:
        print(f"Case {lsa_fname} has no Hopf bifurcations")
        print(f"Real eigenvalue components are {omegas_real}")
    else:
        idx_hopf = is_hopf_bif.index(True)
        print(f"Case {lsa_fname} has a Hopf bifurcation between {PSUBS[idx_hopf]} {PSUBS[idx_hopf+1]} dPa")
        print(f"Real eigenvalue components are {omegas_real}")

        xhopf_0 = libhopf.gen_hopf_initial_guess(
            RES_HOPF,
            ([PSUBS[idx_hopf]], [PSUBS[idx_hopf+1]]),
            ([omegas_real[idx_hopf]], [omegas_real[idx_hopf+1]])
        )
        xhopf_n, info = libhopf.solve_hopf_newton(RES_HOPF, xhopf_0)

        with h5py.File(fpath, mode='a') as f:
            bh5utils.create_resizable_block_vector_group(f.require_group('state'), xhopf_n.labels, xhopf_n.bshape)
            bh5utils.append_block_vector_to_group(f['state'], xhopf_n)

            bh5utils.create_resizable_block_vector_group(f.require_group('props'), props.labels, props.bshape)
            bh5utils.append_block_vector_to_group(f['props'], props)

@_ignore_existing_path
def run_large_amp_model(fpath, emod_cov, emod_bod):
    """
    Run a non-linear/large amplitude (transient) oscillation model
    """
    # Set the cover/body layer properties
    _forms = RES_LAMP.solid.forms
    celllabel_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(_forms, _forms['fspace.scalar'])
    props = set_props(RES_LAMP.props, celllabel_to_dofs, emod_cov, emod_bod)
    RES_LAMP.set_props(props)

    # Load the onset pressure from the Hopf simulation
    h_fname = f'Hopf_ecov{emod_cov:.2e}_ebody{emod_bod:.2e}'
    h_fpath = f'out/stress_test/{h_fname}.h5'
    with h5py.File(h_fpath, mode='r') as f_hopf:
        if 'state' in f_hopf:
            xhopf = bh5utils.read_block_vector_from_group(f_hopf['state'])
            ponset = xhopf['psub'][0]
            xfp = xhopf[:4]
            # apriori known that the fixed point has 4 blocks (u, v, q, p)
        else:
            ponset = None
            xfp = None

    # Run a large amp. simulation at 100 Pa above the onset pressure, if applicable
    if ponset is not None:
        # Integrate the forward model in time
        ini_state = RES_LAMP.state0.copy()
        ini_state[['u', 'v', 'q', 'p']] = xfp
        ini_state['a'][:] = 0.0

        dt = 5e-5
        _times = dt*np.arange(0, int(round(0.5/dt))+1)
        times = bv.BlockVector([_times], labels=(('times',),))

        control = RES_LAMP.control.copy()
        control['psub'][:] = ponset + 100.0*10
        with sf.StateFile(RES_LAMP, fpath, mode='a') as f:
            forward.integrate(RES_LAMP, f, ini_state, [control], RES_LAMP.props, times, use_tqdm=True)
    else:
        print(f"Skipping large amplitude simulation of {fname} because no Hopf bifurcation is detected")

def postproc_gw(fpath, emods_cov, emods_bod):
    """
    Compute glottal width and time data from large amp. simulations
    """
    proc_gw = sigsl.make_sig_glottal_width_sharp(RES_LAMP)
    def proc_time(f):
        return f.get_times()

    signal_to_proc = {
        'gw': proc_gw, 'time': proc_time
    }

    in_names = [
        f'LargeAmp_ecov{emod_cov:.2e}_ebody{emod_bod:.2e}'
        for emod_cov, emod_bod in zip(emods_cov, emods_bod)
    ]
    in_paths = [f'out/stress_test/{name}.h5' for name in in_names]

    with h5py.File(fpath, mode='a') as f:
        return postprocutils.postprocess_case_to_signal(f, in_paths, RES_LAMP, signal_to_proc)

@_ignore_existing_path
def run_inv_opt(fpath, emod_cov, emod_bod, gw_ref, omega_ref, alpha=0.0, opt_options=None):

    ## Set the Hopf system properties
    _forms = RES_HOPF.res.solid.forms
    celllabel_to_dofs = meshutils.process_celllabel_to_dofs_from_forms(_forms, _forms['fspace.scalar'])
    props = set_props(RES_HOPF.props, celllabel_to_dofs, emod_cov, emod_bod)
    RES_HOPF.set_props(props)

    ## Form the log posterior functional
    std_omega = 10.0
    # uncertainty in glottal width (GW) is assumed to be 0.02 cm for the maximum
    # glottal width and increases inversely with GW magnitude; a zero glottal width
    # has infinite uncertainty
    std_gw = (0.1/5) / (np.maximum(gw_ref, 0.0) / gw_ref.max())

    func_omega = libfuncs.AbsOnsetFrequencyFunctional(RES_HOPF)
    func_gw_err = libfuncs.GlottalWidthErrorFunctional(RES_HOPF, gw_ref=gw_ref, weights=1/std_gw)
    func_egrad_norm = alpha * libfuncs.ModulusGradientNormSqr(RES_HOPF)

    func_freq_err = 1/std_omega * (func_omega - 2*np.pi*omega_ref) ** 2
    func = func_gw_err + func_freq_err + func_egrad_norm
    redu_grad = libhopf.ReducedGradient(func, RES_HOPF)

    redu_grad.set_props(props)

    ## Run the optimizer
    # Form the initial guess
    camp0 = optimize_comp_amp(func_gw_err)
    x0 = bv.concatenate_vec([props, camp0])

    def opt_callback(xk):
        print("In callback")

    scale = 1e3*np.ones(x0.mshape)
    scale[-1] = 1.0 # make sure the phase argument has no scaling
    def assem_sc_grad(sc_x, scale, assem_grad):
        x = sc_x * scale
        obj, grad = assem_grad(x)

        sc_obj = obj
        sc_grad = scale * grad
        return sc_obj, sc_grad

    with h5py.File(fpath, mode='a') as f:
         ## Record the reference gw and freq
        f['gw_ref'] = gw_ref
        f['omega_ref'] = omega_ref

        grad_manager = libhopf.OptGradManager(redu_grad, f)
        opt_res = optimize.minimize(
            assem_sc_grad,
            x0.to_mono_ndarray()/scale,
            args=(scale, grad_manager.grad),
            method='L-BFGS-B',
            jac=True,
            options=opt_options,
            callback=opt_callback
        )

        pprint(opt_res)

def _run_inv_opt(emod_cov, emod_bod, alpha, gt_fname):
    fname = f'OptInv_emod{emod_cov:.2e}_ebody{emod_bod:.2e}_alpha_{alpha:.2e}_gt{gt_fname}'
    fpath = f'out/stress_test/{fname}.h5'
    print(f'Optimizing case {fname}')
    try:
        run_inv_opt(
            fpath,
            emod_cov, emod_bod,
            gw_ref, omega_ref,
            alpha=alpha,
            opt_options=opt_options
        )
    except petsc4py.PETSc.Error as err:
        print(f"Case failed with error {err}")

def segment_last_period(y, dt):
    # Segment the final cycle from the glottal width and estimate an uncertainty
    fund_freq, *_ = modalsig.estimate_fundamental_mode(y)
    n_period = int(round(1/fund_freq)) # the number of samples in a period

    # The segmented glottal width makes up the simulated measurement
    ret_y = y[-n_period:]
    ret_omega = fund_freq / dt
    return ret_y, ret_omega

def optimize_comp_amp(func_gw_err):
    ## Compute an initial guess for the complex amplitude
    # Note that the initial guess for amplitude might be negative; this is fine
    # as a negative amplitude is equivalent to pi radian phase shift
    camp0 = func_gw_err.camp.copy()
    def _f(x):
        _camp = func_gw_err.camp
        _camp.set_vec(x)

        func_gw_err.set_camp(_camp)
        return func_gw_err.assem_g(), func_gw_err.assem_dg_dcamp().to_mono_ndarray()
    opt_res = optimize.minimize(_f, np.array([0.0, 0.0]), jac=True)
    camp0.set_vec(opt_res['x'])
    return camp0



if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--numproc', type=int, default=1)
    args = parser.parse_args()

    emods = [
        (ecov, ebod) for ecov, ebod in itertools.product(EMODS, EMODS)
        if ecov <= ebod
    ]
    emods_cov, emods_bod = [e[0] for e in emods], [e[1] for e in emods]

    ## Run linear stability analysis to check if Hopf bifurcations occur or not
    for emod_cov, emod_bod in zip(emods_cov, emods_bod):
        fname = f'LSA_ecov{emod_cov:.2e}_ebody{emod_bod:.2e}'
        fpath = f'out/stress_test/{fname}.h5'
        ans = run_lsa(fpath, emod_cov, emod_bod)

        if ans is not None:
            psubs, omegas_real = ans
            if max(omegas_real) >= 0.0 and min(omegas_real) < 0.0:
                is_hopf = [a < 0 and b >=0 for a, b in zip(omegas_real[:-1], omegas_real[1:])]
                ii = is_hopf.index(True)
                print(f"{fname} has a Hopf bifurcation between {psubs[ii]:.2f} Ba and {psubs[ii+1]:.2f} Ba")
            else:
                print(f"{fname} does not have a Hopf bifurcation between {psubs[0]:.2f} Ba and {psubs[-1]:.2f} Ba")

    ## Solve the Hopf bifurcation system for each case
    for emod_cov, emod_bod in zip(emods_cov, emods_bod):
        fname = f'Hopf_ecov{emod_cov:.2e}_ebody{emod_bod:.2e}'
        fpath = f'out/stress_test/{fname}.h5'
        run_solve_hopf(fpath, emod_cov, emod_bod)

    ## Run a transient simulation for each Hopf bifurcation
    for emod_cov, emod_bod in zip(emods_cov, emods_bod):
        fname = f'LargeAmp_ecov{emod_cov:.2e}_ebody{emod_bod:.2e}'
        fpath = f'out/stress_test/{fname}.h5'
        run_large_amp_model(fpath, emod_cov, emod_bod)

    ## Post process the glottal width and time from each transient simulation
    fpath = 'out/stress_test/signals.h5'
    SIGNALS = postproc_gw(fpath, emods_cov, emods_bod)

    ## Run the inverse analysis studies
    # determine cover/body combinations that self-oscillate
    _emods = [
        (ecov, ebod) for ecov, ebod in zip(emods_cov, emods_bod)
        if len(SIGNALS[f'LargeAmp_ecov{ecov:.2e}_ebody{ebod:.2e}/gw']) != 0
    ]
    _emods_cov = [x[0] for x in _emods]
    _emods_bod = [x[1] for x in _emods]
    ## Load the reference glottal width and omega
    for emod_cov, emod_bod in zip(_emods_cov, _emods_bod):
        gt_fname = f'LargeAmp_ecov{emod_cov:.2e}_ebody{emod_bod:.2e}'
        gw_ref = SIGNALS[f'{gt_fname}/gw']
        t_ref = SIGNALS[f'{gt_fname}/time']
        dt = t_ref[1]-t_ref[0]

        # Segment the last period
        gw_ref, omega_ref = segment_last_period(gw_ref, dt)

        ## Try to optimize to the target data from all starting points
        opt_options = {
            'disp': 99,
            'maxiter': 150
        }
        alphas = 10**np.array([-np.inf] + [-6, -4, -2])

        # for emod_cov, emod_bod in zip(_emods_cov, _emods_bod):
        #     alpha = 0.0
        #     for alpha in alphas:
        #         fname = f'OptInv_emod{emod_cov:.2e}_ebody{emod_bod:.2e}_alpha_{alpha:.2e}_gt{_name}'
        #         fpath = f'out/stress_test/{fname}.h5'
        #         print(f'Optimizing case {fname}')
        #         try:
        #             run_inv_opt(
        #                 fpath,
        #                 emod_cov, emod_bod,
        #                 gw_ref, omega_ref,
        #                 alpha=alpha,
        #                 opt_options=opt_options
        #             )
        #         except petsc4py.PETSc.Error as err:
        #             print(f"Case failed with error {err}")


        _args = [(emod[0], emod[1], alpha) for emod, alpha in itertools.product(_emods, alphas)]

        def _run(ecov, ebod, alpha):
            return _run_inv_opt(ecov, ebod, alpha, gt_fname)

        with multiprocessing.Pool(args.numproc) as p:
            p.starmap(_run, _args)