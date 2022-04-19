import libfunctionals as libfuncs
from test_libhopf import setup_hopf_state


if __name__ == '__main__':
    # Load the Hopf system
    mesh_name = 'BC-dcov5.00e-02-cl1.00'
    mesh_path = path.join('./mesh', mesh_name+'.xml')

    hopf, xhopf, props0 = setup_hopf_state(mesh_path)

    # Load the measurement glottal width


    # func = libfuncs.OnsetPressureFunctional(hopf)
    func = libfuncs.GlottalWidthErrorFunctional(hopf)
