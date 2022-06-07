
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

    # gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    # gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(f'M5_CB_GA{medial_angle:d}.msh')

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

    gmsh.model.add_physical_group(1, [6, 5, 4, 3, 15, 14], name='pressure')
    gmsh.model.add_physical_group(1, [13, 7, 16], name='fixed')

    gmsh.model.add_physical_group(0, [4], name='separation-inf')
    gmsh.model.add_physical_group(0, [3], name='separation-mid')
    gmsh.model.add_physical_group(0, [14], name='separation-sup')

    # gmsh.option.set_number('Mesh.MshFileVersion', MSH_VER)
    # gmsh.option.set_number('Mesh.MeshSizeFactor', SIZE_FACTOR)

    gmsh.model.mesh.generate(2)
    gmsh.write(f'M5_CB_GA{medial_angle:d}_split.msh')

if __name__ == '__main__':
    for medial_angle in [0, 1, 2, 3]:
        proc_M5(medial_angle)
        proc_M5_split(medial_angle)