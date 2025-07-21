"""
Module with common functionality used for testing
"""

import numpy as np
from blockarray.blockarray import BlockArray
import blockarray.linalg as bla


def copy_vector(vector):
    """
    Copy a vector or float
    """
    if isinstance(vector, BlockArray):
        return vector.copy()
    else:
        return vector


def taylor_convergence(x0, dx, res, jac, norm=None):
    """
    Test that the Taylor convergence order is 2
    """
    if norm is None:
        norm = bla.norm

    # Compute the linearized residual from `jac`
    # and exact residual from finite-differences
    dres_linear = jac(x0, dx)

    # Step sizes go from largest to smallest
    alphas = 2 ** np.arange(4)[::-1]
    res_ns = [copy_vector(res(x0 + alpha * dx)) for alpha in alphas]
    res_0 = copy_vector(res(x0))

    dres_exacts = [res_n - res_0 for res_n in res_ns]

    abs_errs = np.array(
        [
            norm(dres_exact - alpha * dres_linear)
            for dres_exact, alpha in zip(dres_exacts, alphas)
        ]
    )
    magnitudes = np.array(
        [
            1 / 2 * norm(dres_exact + alpha * dres_linear)
            for dres_exact, alpha in zip(dres_exacts, alphas)
        ]
    )
    with np.errstate(invalid='ignore'):
        conv_rates = np.log(abs_errs[:-1] / abs_errs[1:]) / np.log(
            alphas[:-1] / alphas[1:]
        )
        rel_errs = abs_errs / magnitudes

    print(
        f"||dres_linear||, ||dres_exact|| = {norm(dres_linear)}, {norm(dres_exacts[-1])}"
    )
    print("Absolute error norms: ", abs_errs)
    print("Relative error norms: ", rel_errs)
    print("Convergence rates: ", np.array(conv_rates))
    return alphas, abs_errs, magnitudes, conv_rates
