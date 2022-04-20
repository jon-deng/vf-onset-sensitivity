"""
Contains definitions of functionals
"""
import numbers
import numpy as np
from jax import numpy as jnp
import jax

from libhopf import HopfModel
import libsignal

from blocktensor import vec as bvec

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
        return Sum(self, other)

    def __radd__(self, other):
        return Sum(other, self)

    def __sub__(self, other):
        return Sum(self, -other)

    def __rsub__(self, other):
        return Sum(other, -self)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarProduct(self, scalar=other)
        else:
            return Product(self, other)

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarProduct(self, scalar=other)
        else:
            return Product(other, self)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarProduct(self, scalar=other**-1)
        else:
            return Product(self, other**-1)

    def __rtruediv__(self, other):
        if isinstance(other, numbers.Number):
            return ScalarProduct(self**-1, scalar=other)
        else:
            return Product(other, self**-1)

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
    def __init__(self, model: HopfModel):
        self.model = model

        self.state = self.model.state
        self.props = self.model.props
        self.camp = bvec.convert_bvec_to_petsc(
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
        dg_dstate.set(0.0)
        dg_dstate['psub'][0] = 1.0
        return dg_dstate

    def assem_dg_dprops(self):
        dg_dprops = self.props.copy()
        dg_dprops.set(0)
        return dg_dprops

    def assem_dg_dcamp(self):
        dg_dcamp = self.camp.copy()
        dg_dcamp.set(0)
        return dg_dcamp

class OnsetFrequencyFunctional(BaseFunctional):
    """
    Represents a functional returning the onset frequency
    """

    def assem_g(self):
        return self.state['omega'][0]

    def assem_dg_dstate(self):
        dg_dstate = self.state.copy()
        dg_dstate.set(0.0)
        dg_dstate['omega'][0] = 1.0
        return dg_dstate

    def assem_dg_dprops(self):
        dg_dprops = self.props.copy()
        dg_dprops.set(0)
        return dg_dprops

    def assem_dg_dcamp(self):
        dg_dcamp = self.camp.copy()
        dg_dcamp.set(0)
        return dg_dcamp

class GlottalWidthErrorFunctional(BaseFunctional):
    """
    Return a weighted square error between model and reference glottal width
    """

    def __init__(self, model, gw_ref=None):
        super().__init__(model)

        if gw_ref is None:
            gw_ref = np.zeros((100,))
        eval_gw = libsignal.make_glottal_width(model, gw_ref.size)

        def _err(state, camp):
            gw_hopf = eval_gw(state, camp)
            return jnp.sum((gw_ref - gw_hopf)**2)

        self._err = _err
        self._grad_state_err = jax.grad(_err, argnums=0)
        self._grad_camp_err = jax.grad(_err, argnums=1)

    def assem_g(self):
        return self._err(self.state.to_ndarray(), self.camp.to_ndarray())

    def assem_dg_dstate(self):
        _dg_dstate = self._grad_state_err(self.state.to_ndarray(), self.camp.to_ndarray())
        dg_dstate = self.state.copy()
        dg_dstate.set_vec(_dg_dstate)
        return dg_dstate

    def assem_dg_dprops(self):
        dg_dprops = self.props.copy()
        dg_dprops.set(0)
        return dg_dprops

    def assem_dg_dcamp(self):
        _dg_dcamp = self._grad_camp_err(self.state.to_ndarray(), self.camp.to_ndarray())
        dg_dcamp = self.camp.copy()
        dg_dcamp.set_vec(_dg_dcamp)
        return dg_dcamp

