"""
Solve a simple test optimization problem
"""

import os.path as path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy import optimize
from blockarray import blockvec  as bvec
from femvf import statefile as sf
from femvf.postprocess import solid as solidsig
from femvf.postprocess.base import TimeSeries
from vfsig import modal as modalsig

from lib_main_transient import case_config
import libhopf, libsignal, libfunctionals as libfuncs
import libsetup

# This code specifies the type of non-linear transient simulation to use as
# a simulated measurement
INIT_STATE_TYPE = 'static'
R_SEP = 1.0
ZETA = 1e-4
ECOV = 5e3*10
EBODY = 5e3*10
Y_GAP = 1e-2

OUT_DIR = f'out/zeta{ZETA:.2e}_rsep{R_SEP:.1f}_ygap{Y_GAP:.2e}_init{INIT_STATE_TYPE}_fixed_rsep'

if __name__ == '__main__':
    ## Load the Hopf system and set the initial state
    # The loaded initial state is known apriori to be a good initial guess
    # because the measurement here is generated from a similar model
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('./mesh', mesh_name+'.xml')
    hopf, res, dres = libsetup.load_hopf_model(
        mesh_path, sep_method='smoothmin', sep_vert_label='separation-inf'
    )

    props = hopf.props.copy()
    props = libsetup.set_default_props(props, hopf.res.solid.forms['mesh.mesh'])
    hopf.set_props(props)

    PSUBS = np.linspace(0, 1000, 11)*10
    xhopf_0 = libhopf.gen_xhopf_0(hopf, PSUBS, tol=100.0)
    xhopf_0, _info = libhopf.solve_hopf_by_newton(hopf, xhopf_0)
    hopf.set_state(xhopf_0)

    ## Load the synthetic measured glottal width + frequency
    model_trans = libsetup.load_transient_model(mesh_path, sep_method='smoothmin')

    proc_gw = TimeSeries(solidsig.MinGlottalWidth(model_trans))
    psub = 5.7e2 * 10
    fname = case_config(mesh_name, psub, ECOV, EBODY)
    fpath = f'{OUT_DIR}/{fname}.h5'
    with sf.StateFile(model_trans, fpath, mode='r') as f:
        _gw_ref = proc_gw(f)
        _time_ref = f.get_times()

    # Segment the final cycle from the glottal width and estimate an uncertainty
    dt = _time_ref[1] - _time_ref[0]
    fund_freq, *_ = modalsig.estimate_fundamental_mode(_gw_ref)
    n_period = int(round(1/fund_freq)) # the number of samples in a period

    # The segmented glottal width makes up the simulated measurement
    gw_ref = _gw_ref[-n_period:]
    omega_ref = fund_freq / dt

    # Estimate uncertainties in simulated measurements
    # arbitrary guess of 10 Hz uncertainty in omega
    std_omega = 10.0
    # uncertainty in glottal width (GW) is assumed to be 0.02 cm for the maximum
    # glottal width and increasing inversely with GW magnitude; a zero glottal width
    # has infinite uncertainty
    std_gw = (0.1/5) / (np.maximum(gw_ref, 0.0) / gw_ref.max())

    ## Create the objective function and gradient needed for optimization

    # Form the log-posterior functional
    func_omega = libfuncs.AbsOnsetFrequencyFunctional(hopf)
    func_gw_err = libfuncs.GlottalWidthErrorFunctional(hopf, gw_ref=gw_ref, weights=1/std_gw)
    func_egrad_norm = libfuncs.ModulusGradientNormSqr(hopf)
    func_egrad_norm.assem_g()

    func_freq_err = 1/std_omega * (func_omega - 2*np.pi*omega_ref) ** 2
    func = func_gw_err + func_freq_err

    redu_grad = libhopf.ReducedFunctional(func, hopf)

    ## Compute an initial guess for the complex amplitude
    # Note that the initial guess for amplitude might be negative; this is fine
    # as a negative amplitude is equivalent to pi radian phase shift
    camp0 = redu_grad.camp.copy()
    def _f(x):
        _camp = func_gw_err.camp
        _camp.set_vec(x)

        func_gw_err.set_camp(_camp)
        return func_gw_err.assem_g(), func_gw_err.assem_dg_dcamp().to_mono_ndarray()
    opt_res = optimize.minimize(_f, np.array([0.0, 0.0]), jac=True)
    camp0.set_vec(opt_res['x'])

    # As a sanity check, plot the initial guess glottal width and the reference glottal width
    camp = camp0
    proc_gw_hopf = libsignal.make_glottal_width(hopf, gw_ref.size)
    gw_hopf = proc_gw_hopf(xhopf.to_mono_ndarray(), camp.to_mono_ndarray())

    plt.plot(gw_hopf, label='hopf')
    plt.plot(gw_ref, label='ref')
    plt.savefig('compare.png')
    # breakpoint()

    x0 = bvec.concatenate_vec([props0, camp0])
    # x0['amp'].set(1.0)

    ## Run the optimizer
    opt_options = {
        'disp': 1,
        'maxiter': 100
    }
    def opt_callback(xk):
        print("In callback")

    with h5py.File(f"out/opt_hist_{fname}.h5", mode='w') as f:
        grad_manager = libhopf.OptGradManager(redu_grad, f)
        opt_res = optimize.minimize(
            grad_manager.grad, x0.to_mono_ndarray(),
            method='L-BFGS-B',
            jac=True,
            options=opt_options,
            callback=opt_callback
        )

    pprint(opt_res)
