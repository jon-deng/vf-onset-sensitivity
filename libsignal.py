"""
Contains functions that return time-varying signals from a Hopf system
"""

import numpy as np
from jax import numpy as jnp

# from femvf.models.dynamical.fluid import smooth_min_weight, wavg
from femvf.models.equations.bernoulli import (
    smooth_min_weight, wavg
)

def _split_mono_hopf_state(state, sizes):
    """
    Split a monolithic Hopf state vector into components

    The Hopf state vector consists of 5 components (the fixed-point,
    real/imag mode shapes, psub, and omega).

    Parameters
    ----------
    state : array_like
        A monolothic vector representing the Hopf system state
    sizes : tuple of length 5
        The sizes of each block in the Hopf state
    """
    assert len(sizes) == 5
    idx_bounds = np.cumsum((0,)+sizes)
    return tuple([
        state[idxa:idxb]
        for idxa, idxb in zip(idx_bounds[:-1], idx_bounds[1:])
        ])

def make_glottal_width(hopf, num_points=100):
    """
    Return a glottal width signal
    """
    XREF = np.array(hopf.res.solid.XREF)

    IDX_U = slice(0, hopf.res.state['u'].size)
    IDX_MEDIAL = hopf.res.fsimap.dofs_solid

    YMID = float(hopf.res.prop['ymid'][0])
    if 'zeta_min' in hopf.res.prop:
        ZETA = float(hopf.res.prop['zeta_min'][0])
    else:
        ZETA = 1e-4

    S = np.array(hopf.dres.fluid.s) # can also be res

    HOPF_COMPONENT_SIZES = tuple([
        hopf.state[labels].mshape[0]
        for labels in hopf.labels_hopf_components
    ])

    def glottal_width(state, camp):
        (xfp,
            mode_real,
            mode_imag,
            psub,
            omega) = _split_mono_hopf_state(state, HOPF_COMPONENT_SIZES)
        ampl, phase = camp[-2], camp[-1]

        # get the reference position of the surface
        ucur = (xfp[IDX_U] + XREF)
        ymedial_ref = ucur[1::2][IDX_MEDIAL]

        u_mode_real = mode_real[IDX_U]
        u_mode_imag = mode_imag[IDX_U]
        u_mode = u_mode_real + 1j*u_mode_imag
        ymedial_mode = u_mode[1::2][IDX_MEDIAL]

        ymedial_signal = jnp.real(
            ymedial_ref
            + ampl * ymedial_mode
            * jnp.exp(
                1j* 2 * jnp.pi * jnp.sign(omega)
                * jnp.arange(num_points)[:, None]/(num_points+1)
                + 1j * phase
            )
            )

        area = 2*(YMID-ymedial_signal)
        wmin = smooth_min_weight(area, ZETA, axis=-1)

        # Compute the minimum area as a smooth approximation of the min
        # min_area = jnp.min(area, axis=-1)
        min_area = wavg(S, area, wmin, axis=-1)

        return min_area

    return glottal_width
