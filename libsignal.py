"""
Contains functions that return time-varying signals from a Hopf system
"""

from jax import numpy as jnp

def make_glottal_width(res, dres, num_points=100):
    """
    Return a glottal width signal
    """
    XREF = res.solid.XREF.vector()

    IDX_U = slice(0, res.state.bsize[0])
    IDX_SURFACE = res.fsimap.dofs_solid

    YMID = res.properties['ymid'][0]

    def glottal_width(xfp, mode_real, mode_imag, psub, ampl):
        # get the reference position of the surface
        uref = xfp[IDX_U][IDX_SURFACE] + XREF[IDX_SURFACE]

        umode_real = mode_real[IDX_U][IDX_SURFACE]
        umode_imag = mode_imag[IDX_U][IDX_SURFACE]

        umode = umode_real + 1j*umode_imag

        usignal = jnp.real(
            uref
            + jnp.exp(
                1j*2*jnp.pi*jnp.arange(num_points)/(num_points+1))[:, None]
            * umode)

        ys = jnp.max(usignal, axis=-1)

        return ys

    return glottal_width
