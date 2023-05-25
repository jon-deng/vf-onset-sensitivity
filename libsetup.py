"""
This modules sets up a 'standard' Hopf model to test
"""

from femvf.models.transient import solid as tsmd, fluid as tfmd
from femvf.models.dynamical import solid as dsmd, fluid as dfmd
from femvf import load

import libhopf

def transient_fluidtype_from_sep_method(sep_method):
    if sep_method == 'fixed':
        return tfmd.BernoulliFixedSep
    elif sep_method == 'smoothmin':
        return tfmd.BernoulliSmoothMinSep
    elif sep_method == 'arearatio':
        return tfmd.BernoulliAreaRatioSep
    else:
        raise ValueError("")

def dynamical_fluidtype_from_sep_method(sep_method, flow_driven=False):
    if sep_method == 'fixed':
        if flow_driven:
            return dfmd.BernoulliFlowFixedSep, dfmd.LinearizedBernoulliFlowFixedSep
        else:
            return dfmd.BernoulliFixedSep, dfmd.LinearizedBernoulliFixedSep
    elif sep_method == 'smoothmin':
        return dfmd.BernoulliSmoothMinSep, dfmd.LinearizedBernoulliSmoothMinSep
    elif sep_method == 'arearatio':
        return dfmd.BernoulliAreaRatioSep, dfmd.LinearizedBernoulliAreaRatioSep
    else:
        raise ValueError("")

def load_hopf_model(
        mesh_path,
        sep_method='fixed',
        sep_vert_label='separation',
        flow_driven=False
    ):
    FluidType, LinFluidType = dynamical_fluidtype_from_sep_method(sep_method, flow_driven=flow_driven)

    kwargs = {
        'fsi_facet_labels': ('pressure',),
        'fixed_facet_labels': ('fixed',),
        'separation_vertex_label': sep_vert_label
    }
    res = load.load_dynamical_fsi_model(
        mesh_path,
        None,
        SolidType=dsmd.KelvinVoigt,
        FluidType=FluidType,
        **kwargs
    )

    dres = load.load_dynamical_fsi_model(
        mesh_path,
        None,
        SolidType=dsmd.LinearizedKelvinVoigt,
        FluidType=LinFluidType,
        **kwargs
    )

    res_hopf = libhopf.HopfModel(res, dres)
    return res_hopf, res, dres

def load_transient_model(
        mesh_path,
        sep_method='fixed',
        sep_vert_label='separation'
    ):
    FluidType = transient_fluidtype_from_sep_method(sep_method)

    return load.load_transient_fsi_model(
        mesh_path, None,
        SolidType=tsmd.KelvinVoigt,
        FluidType=FluidType,
        coupling='explicit',
        separation_vertex_label=sep_vert_label
    )


ECOV = 5e3*10
EBODY = 5e3*10
PSUB = 450 * 10

def set_default_props(prop, mesh):
    """
    Set the model properties
    """
    # VF material prop
    prop['emod'][:] =  ECOV
    prop['emod'][:] =  EBODY

    prop = set_constant_props(prop, mesh)

    return prop

def set_constant_props(prop, mesh):
    prop['eta'][:] =  5.0
    prop['rho'][:] =  1.0
    prop['nu'][:] =  0.45

    # Fluid separation smoothing prop
    if all(key in prop for key in ['zeta_min', 'zeta_sep']):
        prop['zeta_min'][:] =  1.0e-4
        prop['zeta_sep'][:] =  1.0e-4

    if 'r_sep' in prop:
        prop['r_sep'][:] =  1.0

    # Contact and midline symmetry properties
    # y_gap = 0.5 / 10 # Set y gap to 0.5 mm
    # y_gap = 1.0
    y_gap = 0.01
    y_contact_offset = 1/10*y_gap
    y_max = mesh.coordinates()[:, 1].max()
    y_mid = y_max + y_gap
    y_contact = y_mid - y_contact_offset
    prop['ycontact'][:] =  y_contact
    prop['kcontact'][:] =  1e16
    prop['ymid'][:] =  y_mid

    prop['rho_air'][:] =  1.293e-3

    return prop
