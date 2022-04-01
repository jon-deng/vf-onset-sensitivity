"""
Contains functions that return time-varying signals from a Hopf system
"""

from jax import numpy as jnp

def make_glottal_width(res, dres, num_points=100):
    """
    Return a glottal width signal
    """
    XREF = res.solid.XREF.vector()

    IDX_U = slice(0, res.state['u'].size)
    IDX_MEDIAL = res.fsimap.dofs_solid

    YMID = res.properties['ymid'][0]

    def glottal_width(xfp, mode_real, mode_imag, psub, ampl):
        # get the reference position of the surface
        u_ref = (xfp[IDX_U] + XREF)
        ymedial_ref = u_ref[1::2][IDX_MEDIAL]

        u_mode_real = mode_real[IDX_U]
        u_mode_imag = mode_imag[IDX_U]
        u_mode = u_mode_real + 1j*u_mode_imag
        ymedial_mode = u_mode[1::2][IDX_MEDIAL]

        ysignal = jnp.real(
            ymedial_ref
            + ampl * ymedial_mode
            * jnp.exp(
                2j*jnp.pi*jnp.arange(num_points)/(num_points+1)
                )[:, None]
            )
        ys = jnp.min(YMID-ysignal, axis=-1)

        return 2*ys

    return glottal_width
