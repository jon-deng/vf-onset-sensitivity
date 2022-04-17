"""
Contains definitions of functionals
"""

import numpy as np

from libhopf import HopfModel

from blocktensor import vec as bvec

class GenericFunctional:
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

    def assem_g(self):
        raise NotImplementedError()

    def assem_dg_dstate(self):
        raise NotImplementedError()

    def assem_dg_dprops(self):
        raise NotImplementedError()

    def assem_dg_dcamp(self):
        raise NotImplementedError()

class BinaryFunctional(GenericFunctional):
    def __init__(self, a: GenericFunctional, b: GenericFunctional):
        self.a = a
        self.b = b

class UnaryFunctional(GenericFunctional):
    def __init__(self, a: GenericFunctional, scalar=0.0):
        self.a = a
        self.C = scalar

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


class OnsetPressureFunctional(GenericFunctional):
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

