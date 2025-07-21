"""
This modules sets up a 'standard' Hopf model to test
"""

import dolfin as dfn

from femvf.models import transient
from femvf.residuals import fluid as tfmd, solid as tsmd
from femvf.models.dynamical import solid as dsmd, fluid as dfmd
from femvf import load

from . import hopf


def transient_fluidtype_from_sep_method(sep_method):
    if sep_method == 'fixed':
        return tfmd.BernoulliFixedSep
    elif sep_method == 'smoothmin':
        return tfmd.BernoulliSmoothMinSep
    elif sep_method == 'arearatio':
        return tfmd.BernoulliAreaRatioSep
    else:
        raise ValueError("")


def dynamical_fluidtype_from_sep_method(sep_method, bifparam_key='psub'):
    if sep_method == 'fixed':
        if bifparam_key == 'qsub':
            return dfmd.BernoulliFlowFixedSep
        elif bifparam_key == 'psub':
            return dfmd.BernoulliFixedSep
        else:
            raise ValueError("")
    elif sep_method == 'smoothmin':
        return dfmd.BernoulliSmoothMinSep
    elif sep_method == 'arearatio':
        return dfmd.BernoulliAreaRatioSep
    else:
        raise ValueError("")


def load_hopf_model(
    mesh_path,
    sep_method='fixed',
    sep_vert_label='separation',
    bifparam_key='psub',
    zs=None,
):
    FluidResidual = dynamical_fluidtype_from_sep_method(
        sep_method, bifparam_key=bifparam_key
    )

    # First load a dummy solid model so you can find a separation point
    if zs is None:
        mesh_dim = 2
    else:
        mesh_dim = 3
    solid_kwargs = {
        'dirichlet_bcs': {
            'state/u1': [(dfn.Constant(mesh_dim*[0]), 'facet', 'fixed')]
        }
        # 'fsi_facet_labels': ('pressure',),
        # 'fixed_facet_labels': ('fixed',),
    }
    if sep_method == 'fixed':
        _solid_model = load.load_fenics_model(
            mesh_path,
            tsmd.KelvinVoigtWShape,
            **solid_kwargs,
        )
        _idx_sep = load.locate_separation_vertex(
            _solid_model.residual, separation_vertex_label=sep_vert_label
        )
        fluid_kwargs = {
            'idx_sep': _idx_sep
        }
    else:
        fluid_kwargs = {}

    model_kwargs = {
        'zs': zs,
    }
    res = load.load_fsi_model(
        mesh_path,
        dsmd.KelvinVoigtWShape,
        FluidResidual,
        solid_kwargs,
        fluid_kwargs,
        model_type='dynamical',
        **model_kwargs
    )

    dres = load.load_fsi_model(
        mesh_path,
        dsmd.KelvinVoigtWShape,
        FluidResidual,
        solid_kwargs,
        fluid_kwargs,
        model_type='linearized_dynamical',
        **model_kwargs,
    )

    res_hopf = hopf.HopfModel(res, dres)
    return res_hopf, res, dres


def load_transient_model(mesh_path, sep_method='fixed', sep_vert_label='separation', zs=None):
    FluidResidual = transient_fluidtype_from_sep_method(sep_method)

    solid_kwargs = {
        'fsi_facet_labels': ('pressure',),
        'fixed_facet_labels': ('fixed',),
    }
    _solid_model = load.load_fenics_model(
        mesh_path,
        tsmd.KelvinVoigtWShape,
        dirichlet_bcs,
    )
    _idx_sep = load.locate_separation_vertex(
        _solid_model.residual, separation_vertex_label=sep_vert_label
    )
    fluid_kwargs = {
        'idx_sep': _idx_sep
    }

    model_kwargs = {
        'zs': zs,
    }

    return load.load_fsi_model(
        mesh_path,
        tsmd.KelvinVoigt,
        FluidResidual,
        solid_kwargs,
        fluid_kwargs,
        model_type='transient',
        coupling='explicit',
        **model_kwargs
    )


ECOV = 5e3 * 10
EBODY = 5e3 * 10
PSUB = 450 * 10


def set_default_props(prop, mesh):
    """
    Set the model properties
    """
    # VF material prop
    prop['emod'][:] = ECOV
    prop['emod'][:] = EBODY

    prop = set_constant_props(prop, mesh)

    return prop


def set_constant_props(prop, mesh):
    prop['eta'][:] = 5.0
    prop['rho'][:] = 1.0
    prop['nu'][:] = 0.45

    # Fluid separation smoothing prop
    fluid_values = {
        'zeta_min': 1e-4,
        'zeta_sep': 1e-4,
        'r_sep': 1.0,
        'rho_air': 1.293e-3,
    }
    for key, value in fluid_values.items():
        if key in prop:
            prop[key][:] = value

    # Contact and midline symmetry properties
    # y_gap = 0.5 / 10 # Set y gap to 0.5 mm
    # y_gap = 1.0
    y_gap = 0.01
    y_contact_offset = 1 / 10 * y_gap
    y_max = mesh.coordinates()[:, 1].max()
    y_mid = y_max + y_gap
    y_contact = y_mid - y_contact_offset
    prop['ycontact'][:] = y_contact
    prop['kcontact'][:] = 1e16
    prop['ymid'][:] = y_mid

    return prop
