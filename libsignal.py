"""
Contains functions that return time-varying signals from a Hopf system
"""

from jax import numpy as jnp

from femvf.dynamicalmodels.fluid import smooth_min_weight, wavg

def make_glottal_width(res, dres, num_points=100):
    """
    Return a glottal width signal
    """
    XREF = res.solid.XREF.vector()

    IDX_U = slice(0, res.state['u'].size)
    IDX_MEDIAL = res.fsimap.dofs_solid

    YMID = res.properties['ymid'][0]
    ZETA = res.properties['zeta_min'][0]
    S = res.fluid.s

    def glottal_width(xfp, mode_real, mode_imag, psub, ampl, phase):
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
                1j* 2 * jnp.pi
                * jnp.arange(num_points)[:, None]/(num_points+1)
                + 1j * phase
            )
            )

        breakpoint()
        area = 2*(YMID-ysignal)
        wmin = smooth_min_weight(area, ZETA, axis=-1)
        min_area = wavg(S, area, wmin, axis=-1)
        # min_area = jnp.min(YMID-ysignal, axis=-1)

        return min_area

    return glottal_width
