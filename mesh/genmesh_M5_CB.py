
import sys
import gmsh

gmsh.initialize(sys.argv)

MSH_VER = 2.0
SIZE_FACTOR = 8

def proc_M5(medial_angle):
    """
    Generate a mesh for the M5_CB_GA*.STEP geometry
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'stp/M5_CB_GA{medial_angle:d}.STEP')

    gmsh.model.add_physical_group(2, [2], name='body')
    gmsh.model.add_physical_group(2, [1], name='cover')

    gmsh.model.add_physical_group(1, [11, 10, 9, 8, 12], name='pressure')
    gmsh.model.add_physical_group(1, [13, 7, 1], name='fixed')

    gmsh.model.add_physical_group(0, [10], name='separation-inf')
    gmsh.model.add_physical_group(0, [9], name='separation-sup')

def proc_M5_split(medial_angle):
    """
    Generate a mesh for the M5_CB_GA*_split.STEP geometry
    """
    gmsh.clear()
    gmsh.model.add('main')

    gmsh.option.set_string('Geometry.OCCTargetUnit', 'CM')
    gmsh.merge(f'stp/M5_CB_GA{medial_angle:d}_split.STEP')

    gmsh.model.add_physical_group(2, [3], name='body')
    gmsh.model.add_physical_group(2, [1, 2], name='cover')

    gmsh.model.add_physical_group(1, [10, 9, 8, 7, 1, 6], name='pressure')
    gmsh.model.add_physical_group(1, [11, 16, 5], name='fixed')

    gmsh.model.add_physical_group(0, [7], name='separation-inf')
    gmsh.model.add_physical_group(0, [2], name='separation-mid')
    gmsh.model.add_physical_group(0, [1], name='separation-sup')

if __name__ == '__main__':
    clscale = gmsh.option.get_number('Mesh.MeshSizeFactor')
    for medial_angle in [0, 1, 2, 3]:
        proc_M5(medial_angle)
        gmsh.model.mesh.generate(2)
        gmsh.write(f'M5_CB_GA{medial_angle:d}_CL{clscale:.2f}.msh')

        proc_M5_split(medial_angle)
        gmsh.model.mesh.generate(2)
        gmsh.write(f'M5_CB_GA{medial_angle:d}_CL{clscale:.2f}_split.msh')