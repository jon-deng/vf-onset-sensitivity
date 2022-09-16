"""
Contains definitions of functionals
"""
import numbers
import numpy as np
from jax import numpy as jnp
import jax
import dolfin as dfn
import ufl

from femvf.models.assemblyutils import CachedFormAssembler
from blockarray import blockvec  as bvec

import libsignal

# pylint: disable=abstract-method

class GenericFunctional:
    """
    All functionals have to supply the below functions
    """
    def assem_g(self):
        raise NotImplementedError()

    def assem_dg_dstate(self):
        raise NotImplementedError()

    def assem_dg_dprops(self):
        raise NotImplementedError()

    def assem_dg_dcamp(self):
        raise NotImplementedError()


    def __pos__(self):
        return self

    def __neg__(self):
        return ScalarProduct(self, scalar=-1)

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarSum(self, scalar=other)
        elif isinstance(other, GenericFunctional):
            return Sum(self, other)
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarSum(self, scalar=other)
        elif isinstance(other, GenericFunctional):
            return Sum(other, self)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarSum(self, scalar=-other)
        elif isinstance(other, GenericFunctional):
            return Sum(self, -other)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarSum(-self, scalar=other)
        elif isinstance(other, GenericFunctional):
            return Sum(other, -self)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarProduct(self, scalar=other)
        elif isinstance(other, GenericFunctional):
            return Product(self, other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarProduct(self, scalar=other)
        elif isinstance(other, GenericFunctional):
            return Product(other, self)
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarProduct(self, scalar=other**-1)
        elif isinstance(other, GenericFunctional):
            return Product(self, other**-1)
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarProduct(self**-1, scalar=other)
        elif isinstance(other, GenericFunctional):
            return Product(other, self**-1)
        else:
            return NotImplemented

    def __pow__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarPower(self, scalar=other)
        else:
            return NotImplemented


class DerivedFunctional(GenericFunctional):
    """
    Derived functional are computed from other functionals
    """
    def __init__(self, funcs):
        self.funcs = tuple(funcs)

    @property
    def state(self):
        return self.funcs[0].state

    @property
    def camp(self):
        return self.funcs[0].camp

    def set_state(self, state):
        for func in self.funcs:
            func.set_state(state)

    def set_props(self, props):
        for func in self.funcs:
            func.set_props(props)

    def set_camp(self, camp):
        for func in self.funcs:
            func.set_camp(camp)

class BinaryFunctional(DerivedFunctional):
    def __init__(self, a: GenericFunctional, b: GenericFunctional):
        self.a = a
        self.b = b

        super().__init__((a, b))

class UnaryFunctional(DerivedFunctional):
    def __init__(self, a: GenericFunctional, scalar=0.0):
        self.a = a
        self.C = scalar

        super().__init__((a,))


class Sum(BinaryFunctional):
    def assem_g(self):
        return self.a.assem_g() + self.b.assem_g()

    def assem_dg_dstate(self):
        return self.a.assem_dg_dstate() + self.b.assem_dg_dstate()

    def assem_dg_dprops(self):
        return self.a.assem_dg_dprops() + self.b.assem_dg_dprops()

    def assem_dg_dcamp(self):
        return self.a.assem_dg_dcamp() + self.b.assem_dg_dcamp()

class Product(BinaryFunctional):
    def assem_g(self):
        return self.a.assem_g() * self.b.assem_g()

    def assem_dg_dstate(self):
        return self.a.assem_g()*self.b.assem_dg_dstate() + self.a.assem_dg_dstate()*self.b.assem_g()

    def assem_dg_dprops(self):
        return self.a.assem_g()*self.b.assem_dg_dprops() + self.a.assem_dg_dprops()*self.b.assem_g()

    def assem_dg_dcamp(self):
        return self.a.assem_g()*self.b.assem_dg_dcamp() + self.a.assem_dg_dcamp()*self.b.assem_g()

class ScalarSum(UnaryFunctional):
    def assem_g(self):
        return self.a.assem_g() + self.C

    def assem_dg_dstate(self):
        return self.a.assem_dg_dstate()

    def assem_dg_dprops(self):
        return self.a.assem_dg_dprops()

    def assem_dg_dcamp(self):
        return self.a.assem_dg_dcamp()

class ScalarPower(UnaryFunctional):
    def assem_g(self):
        return self.a.assem_g() ** self.C

    def assem_dg_dstate(self):
        return self.C * self.a.assem_g()**(self.C-1) * self.a.assem_dg_dstate()

    def assem_dg_dprops(self):
        return self.C * self.a.assem_g()**(self.C-1) * self.a.assem_dg_dprops()

    def assem_dg_dcamp(self):
        return self.C * self.a.assem_g()**(self.C-1) * self.a.assem_dg_dcamp()

class ScalarProduct(UnaryFunctional):
    def assem_g(self):
        return self.C * self.a.assem_g()

    def assem_dg_dstate(self):
        return self.C * self.a.assem_dg_dstate()

    def assem_dg_dprops(self):
        return self.C * self.a.assem_dg_dprops()

    def assem_dg_dcamp(self):
        return self.C * self.a.assem_dg_dcamp()


class BaseFunctional(GenericFunctional):
    """
    Represents a functional object acting on the Hopf state

    The functional is represented by
        f(x, p, camp)
    where x is the Hopf state vector, p are the model properties and camp is the
    complex amplitude.
    """
    def __init__(self, model: 'libhopf.HopfModel'):
        self.model = model

        self.state = self.model.state
        self.props = self.model.props
        self.camp = bvec.convert_subtype_to_petsc(
            bvec.BlockVector([np.zeros(1), np.zeros(1)], (2,), (('amp', 'phase'),))
            )

    def set_state(self, state):
        self.model.set_state(state)

    def set_props(self, props):
        self.model.set_props(props)

    def set_camp(self, camp):
        self.camp[:] = camp

class OnsetPressureFunctional(BaseFunctional):
    """
    Represents a functional returning the onset pressure

    The functional is represented by
        f(x, p, camp)
    where x is the Hopf state vector, p are the model properties and camp is the
    complex amplitude.
    """

    def assem_g(self):
        return self.state['psub'][0]

    def assem_dg_dstate(self):
        dg_dstate = self.state.copy()
        dg_dstate[:] = 0.0
        dg_dstate['psub'][0] = 1.0
        return dg_dstate

    def assem_dg_dprops(self):
        dg_dprops = self.props.copy()
        dg_dprops[:] = 0
        return dg_dprops

    def assem_dg_dcamp(self):
        dg_dcamp = self.camp.copy()
        dg_dcamp[:] = 0
        return dg_dcamp

class OnsetFrequencyFunctional(BaseFunctional):
    """
    Represents a functional returning the onset frequency
    """

    def assem_g(self):
        return self.state['omega'][0]

    def assem_dg_dstate(self):
        dg_dstate = self.state.copy()
        dg_dstate[:] = 0.0
        dg_dstate['omega'][0] = 1.0
        return dg_dstate

    def assem_dg_dprops(self):
        dg_dprops = self.props.copy()
        dg_dprops[:] = 0
        return dg_dprops

    def assem_dg_dcamp(self):
        dg_dcamp = self.camp.copy()
        dg_dcamp[:] = 0
        return dg_dcamp

class AbsOnsetFrequencyFunctional(BaseFunctional):
    """
    Represents a functional returning the onset frequency
    """

    def assem_g(self):
        return abs(self.state['omega'][0])

    def assem_dg_dstate(self):
        dg_dstate = self.state.copy()
        dg_dstate[:] = 0.0
        dg_dstate['omega'][0] = np.sign(self.state['omega'][0])
        return dg_dstate

    def assem_dg_dprops(self):
        dg_dprops = self.props.copy()
        dg_dprops[:] = 0
        return dg_dprops

    def assem_dg_dcamp(self):
        dg_dcamp = self.camp.copy()
        dg_dcamp[:] = 0
        return dg_dcamp

class GlottalWidthErrorFunctional(BaseFunctional):
    """
    Return a weighted square error between model and reference glottal width
    """

    def __init__(self, model, gw_ref=None, weights=None):
        super().__init__(model)

        if gw_ref is None:
            gw_ref = np.zeros((100,))

        if weights is None:
            weights = np.ones((100,))

        assert weights.size == gw_ref.size

        eval_gw = libsignal.make_glottal_width(model, gw_ref.size)

        def _err(state, camp):
            gw_hopf = eval_gw(state, camp)
            return jnp.sum(weights*(gw_ref - gw_hopf)**2)

        self._err = _err
        self._grad_state_err = jax.grad(_err, argnums=0)
        self._grad_camp_err = jax.grad(_err, argnums=1)

    def assem_g(self):
        return self._err(self.state.to_mono_ndarray(), self.camp.to_mono_ndarray())

    def assem_dg_dstate(self):
        _dg_dstate = self._grad_state_err(self.state.to_mono_ndarray(), self.camp.to_mono_ndarray())
        dg_dstate = self.state.copy()
        dg_dstate.set_mono(_dg_dstate)
        return dg_dstate

    def assem_dg_dprops(self):
        dg_dprops = self.props.copy()
        dg_dprops[:] = 0
        return dg_dprops

    def assem_dg_dcamp(self):
        _dg_dcamp = self._grad_camp_err(self.state.to_mono_ndarray(), self.camp.to_mono_ndarray())
        dg_dcamp = self.camp.copy()
        dg_dcamp.set_mono(_dg_dcamp)
        return dg_dcamp

class StrainEnergyFunctional(BaseFunctional):

    def __init__(self, model: 'libhopf.HopfModel'):
        super().__init__(model)

        from femvf.models.equations.solid.solidforms import form_inf_strain
        forms = model.res.solid.forms
        dis = forms['coeff.state.u1']
        inf_strain = form_inf_strain(dis)

        emod = forms['coeff.prop.emod']
        cauchy_stress = forms['expr.stress_elastic']
        dx = forms['measure.dx']

        strain_energy = ufl.inner(cauchy_stress, inf_strain)*dx
        self.assem_strain_energy = CachedFormAssembler(strain_energy)
        dstrain_energy_du = dfn.derivative(strain_energy, dis)
        self.assem_dstrain_energy_du = CachedFormAssembler(dstrain_energy_du)
        dstrain_energy_demod = dfn.derivative(strain_energy, emod)
        self.assem_dstrain_energy_demod = CachedFormAssembler(dstrain_energy_demod)

    def assem_g(self):
        return np.array(self.assem_strain_energy.assemble())

    def assem_dg_dstate(self):
        dg_dstate = self.state.copy()
        dg_dstate[:] = 0.0
        dstrain_energy_du = self.assem_dstrain_energy_du.assemble()
        dg_dstate['u'] = dstrain_energy_du
        return dg_dstate

    def assem_dg_dprops(self):
        dg_dprops = self.props.copy()
        dg_dprops[:] = 0
        dg_demod = self.assem_dstrain_energy_demod.assemble()
        dg_dprops['emod'] = dg_demod
        return dg_dprops

    def assem_dg_dcamp(self):
        dg_dcamp = self.camp.copy()
        dg_dcamp[:] = 0
        return dg_dcamp

class ModulusGradientNormSqr(BaseFunctional):
    """
    Returns the L2 norm (^2) of the gradient of the modulus
    """
    def __init__(self, model):
        super().__init__(model)

        # Define the modulus gradient
        forms = model.res.solid.forms
        E = forms['coeff.prop.emod']
        dx = forms['measure.dx']
        # TODO: This doesn't work if E is from a function space without a gradient! (DG0)
        grad_E = ufl.grad(E)
        self._functional = ufl.inner(grad_E, grad_E) * dx
        self._dfunctional_demod = dfn.derivative(self._functional, E)

    def assem_g(self):
        return dfn.assemble(self._functional)

    def assem_dg_dstate(self):
        dg_dstate = self.state.copy()
        dg_dstate[:] = 0.0
        return dg_dstate

    def assem_dg_dprops(self):
        dg_dprops = self.props.copy()
        dg_dprops[:] = 0
        dg_dprops['emod'][:] = dfn.assemble(
            self._dfunctional_demod, tensor=dfn.PETScVector()
        )[:]
        return dg_dprops

    def assem_dg_dcamp(self):
        dg_dcamp = self.camp.copy()
        dg_dcamp[:] = 0
        return dg_dcamp
