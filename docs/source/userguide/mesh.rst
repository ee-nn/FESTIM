.. _mesh_guide:

====
Mesh
====

.. testsetup:: mesh

    import festim as F

Meshes are required to discretise the geometrical domain of the simulation.
As FESTIM is not a meshing tool, its meshing capabilities are limited to simple 1D meshes.
Higher-dimensional meshes are built with `DOLFINx <https://docs.fenicsproject.org/dolfinx/main/python/>`_ or with external meshing software and read into FESTIM.

Regardless of how it is defined, the mesh is passed to the :code:`mesh` attribute of the problem.

---------
1D meshes
---------

The easiest way to define a 1D mesh in FESTIM is to define it from a list of vertices (see :class:`festim.Mesh1D`):

.. testcode:: mesh

    mesh = F.Mesh1D(vertices=[0, 1, 2, 4, 5, 10])

For bigger meshes, use the numpy library to generate an array of vertices.

.. testcode:: mesh

    import numpy as np

    mesh = F.Mesh1D(vertices=np.linspace(0, 10, num=1000))

Numpy arrays can be combined to have local refinements:

.. testcode:: mesh

    import numpy as np

    vertices = np.concatenate(
        [
            np.linspace(0, 1e-6, num=100),  # 99 cells between 0 and 1 micron
            np.linspace(1e-6, 1e-4, num=100),  # 99 cells between 1 micron and 0.1 mm
            np.linspace(1e-4, 1e-2, num=10)  # 9 cells between 0.1 mm and 1 cm
        ]
    )
    mesh = F.Mesh1D(vertices=vertices)

Duplicated vertices (here at 1e-6 and 1e-4) are removed automatically.

Several disconnected solids can be represented with one mesh by giving the vertices as a list of lists, one per solid.
No cell is created between the last vertex of a block and the first vertex of the next one, leaving a gap which can then be coupled through an enclosure:

.. testcode:: mesh

    mesh = F.Mesh1D(vertices=[[0, 0.1, 0.2, 0.3], [1, 1.1, 1.2]])

Volume subdomains of 1D meshes are defined with :class:`festim.VolumeSubdomain1D` and surfaces with :class:`festim.SurfaceSubdomain1D` (see :doc:`Subdomains & Materials <subdomains>`).

-------------------
Meshes from DOLFINx
-------------------

Any :code:`dolfinx.mesh.Mesh` object can be used in FESTIM by wrapping it in :class:`festim.Mesh`.
This is the simplest way to obtain 2D and 3D meshes of simple geometries, using the `built-in meshes of DOLFINx <https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.mesh.html>`_:

.. testcode:: mesh

    from mpi4py import MPI
    from dolfinx.mesh import create_unit_square

    dolfinx_mesh = create_unit_square(MPI.COMM_WORLD, 50, 50)

    mesh = F.Mesh(mesh=dolfinx_mesh)

Similarly, :code:`create_rectangle`, :code:`create_box`, :code:`create_unit_cube`... can be used.

Since such meshes do not carry any tag, the volume and surface subdomains must be located with a ``locator`` function (see :ref:`Surface Subdomains` and :ref:`Volume Subdomains`):

.. testcode:: mesh

    import numpy as np

    my_mat = F.Material(D_0=1, E_D=0.1)

    left_half = F.VolumeSubdomain(
        id=1, material=my_mat, locator=lambda x: x[0] <= 0.5
    )
    right_half = F.VolumeSubdomain(
        id=2, material=my_mat, locator=lambda x: x[0] >= 0.5
    )
    left_surface = F.SurfaceSubdomain(id=1, locator=lambda x: np.isclose(x[0], 0))
    right_surface = F.SurfaceSubdomain(id=2, locator=lambda x: np.isclose(x[0], 1))

.. note::

    Alternatively, if you already have facet and cell meshtags (for instance from gmsh, see below), they can be given directly to the problem through its :code:`facet_meshtags` and :code:`volume_meshtags` attributes.
    The subdomain ids must then match the values of the meshtags.

Coordinate systems
------------------

By default, the equations are solved in cartesian coordinates.
Cylindrical (1D or 2D meshes) and spherical (1D meshes only) coordinates can be selected with the :code:`coordinate_system` argument:

.. testcode:: mesh

    mesh = F.Mesh1D(vertices=np.linspace(1e-3, 2e-3, num=100), coordinate_system="spherical")

.. testcode:: mesh

    mesh = F.Mesh(mesh=dolfinx_mesh, coordinate_system="cylindrical")

In cylindrical coordinates :code:`x[0]` is the radial coordinate :math:`r` and :code:`x[1]` is :math:`z`; in spherical coordinates :code:`x[0]` is :math:`r`.

----------------
Meshes from XDMF
----------------

More complex meshes can be read from XDMF files (see :class:`festim.MeshFromXDMF`): one file containing the mesh and the volume (cell) tags, and one containing the surface (facet) tags.

.. testsetup:: mesh

    import numpy as np
    from mpi4py import MPI
    from dolfinx.io import XDMFFile
    from dolfinx.mesh import create_unit_square, locate_entities_boundary, meshtags

    _mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    _num_cells = _mesh.topology.index_map(2).size_local
    _ct = meshtags(
        _mesh, 2, np.arange(_num_cells, dtype=np.int32), np.full(_num_cells, 1, dtype=np.int32)
    )
    _facets = locate_entities_boundary(_mesh, 1, lambda x: np.isclose(x[0], 0))
    _ft = meshtags(_mesh, 1, _facets, np.full(len(_facets), 1, dtype=np.int32))
    _mesh.topology.create_connectivity(1, 2)
    _mesh.name = _ct.name = _ft.name = "Grid"
    for _name, _tags in [("volume_mesh.xdmf", _ct), ("surface_mesh.xdmf", _ft)]:
        with XDMFFile(MPI.COMM_WORLD, _name, "w") as _f:
            _f.write_mesh(_mesh)
            _f.write_meshtags(_tags, _mesh.geometry)

.. testcode:: mesh

    mesh = F.MeshFromXDMF(volume_file="volume_mesh.xdmf", facet_file="surface_mesh.xdmf")

.. testcleanup:: mesh

    import os
    for _name in ["volume_mesh", "surface_mesh"]:
        for _ext in [".xdmf", ".h5"]:
            if os.path.exists(_name + _ext):
                os.remove(_name + _ext)

When such a mesh is used, the meshtags are read from the files and the subdomains do not need a ``locator``: their ids simply have to match the tags in the files.

The XDMF files must be readable by DOLFINx.
By default, the mesh and the meshtags are looked up under the name ``"Grid"`` in the files, which is what `meshio <https://github.com/nschloe/meshio>`_ writes.
Different names can be given with the ``mesh_name``, ``volume_meshtags_name`` and ``surface_meshtags_name`` arguments.
For instance, files written by :code:`dolfinx.io.XDMFFile` use the names of the :code:`dolfinx.mesh.Mesh` and :code:`dolfinx.mesh.MeshTags` objects (``"mesh"`` and ``"mesh_tags"`` by default).

The recommended workflow is to mesh your geometry with your favourite meshing software (`SALOME <https://www.salome-platform.org/>`_, `gmsh <https://gmsh.info/>`_...) and either read it directly with DOLFINx (gmsh) or convert the produced mesh to XDMF with meshio.

GMSH example
------------

The DOLFINx tutorial gives an `example <https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html#creating-the-mesh>`_ of mesh generation with gmsh, and additionally the GMSH reference manual can be accessed `here <https://gmsh.info/dev/doc/texinfo/gmsh.pdf>`_

The following is a workflow using the python API to make a mesh that can be directly integrated into FESTIM:

Here we will walk through GMSH's usage when creating a monoblock subsection consisting of tungsten surrounding a tube of CuCrZr

.. figure:: ../images/gmsh_tut_1.png
    :width: 400
    :align: center

Meshing the geometry with GMSH
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GMSH can be installed via the following `link <https://gmsh.info>`_.

To use the Python API, gmsh will need to be installed in the same environment as FESTIM, for instance with

.. code-block:: bash

    conda install -c conda-forge python-gmsh

Now, GMSH must be imported and initialised.

.. code-block:: python

    import gmsh

    gmsh.initialize()
    gmsh.model.add("mesh")

We can set the size of our mesh using:

.. code-block:: python

    lc = 1e-3

Models in GMSH consist of a series of:

- Points
- Lines
-  Wires / Curve Loops
   - whether we use curve loops or wires depends on whether we use the `.occ` or `.geo` geometry kernels. `.occ` allows for direct construction of more complex features such as cylinders, whereas using `.geo` requires explicit user definition of all the points, surfaces and volumes that would make up the cylinder.
-  Surfaces
-  Surface Loops
-  Volumes

We will begin by defining the points of our square of tungsten.

.. code-block:: python

    p1 = gmsh.model.occ.addPoint(-15e-3, 15e-3, 0, lc)
    p2 = gmsh.model.occ.addPoint(-15e-3, -15e-3, 0, lc)
    p3 = gmsh.model.occ.addPoint(15e-3, 15e-3, 0, lc)
    p4 = gmsh.model.occ.addPoint(15e-3, -15e-3, 0, lc)

These points can then be joined together using lines. It is important that we pay close attention to the direction that these lines are going.

.. code-block:: python

    line_1_2 = gmsh.model.occ.addLine(p1, p2)
    line_1_3 = gmsh.model.occ.addLine(p1, p3)
    line_2_4 = gmsh.model.occ.addLine(p2, p4)
    line_3_4 = gmsh.model.occ.addLine(p3, p4)

These are then used to create curve loops or wires.
Wires and curve loops must be closed loops, and the list of lines must flow in the correct direction so as to form a complete loop.

.. code-block:: python

    base_loop = gmsh.model.occ.addWire([line_1_2, line_2_4, -line_3_4, -line_1_3])

We can also define the inner and outer circles and loops for the CuCrZr tube.

.. code-block:: python

    inner_circle = gmsh.model.occ.addCircle(0, 0, 0, 5e-3)
    outer_circle = gmsh.model.occ.addCircle(0, 0, 0, 10e-3)

    inner_circle_loop = gmsh.model.occ.addWire([inner_circle])
    outer_circle_loop = gmsh.model.occ.addWire([outer_circle])

Surfaces are defined using loops, where the first loop in the list denotes the outer borders of the surface, and any others define holes within the surface.
Here `base_surface` is our tungsten layer, and so it consists of our base rectangle curve loop, with a hole defined by the outer CuCrZr loop.

.. code-block:: python

    base_surface = gmsh.model.occ.addPlaneSurface([base_loop, outer_circle_loop])
    cylinder_surface = gmsh.model.occ.addPlaneSurface([outer_circle_loop, inner_circle_loop])

While we could then define another surface above the first and join them together, it is often easier to just perform an extrusion of the surfaces.
Here we stretch both the tungsten and CuCrZr surfaces by 5e-3 in the z-direction, and 0 in the x and y.

.. code-block:: python

    outer_layer_extrusion = gmsh.model.occ.extrude(
        [(2, base_surface)], 0, 0, 5e-3, numElements=[100]
    )
    interface_layer_extrusion = gmsh.model.occ.extrude(
        [(2, cylinder_surface)], 0, 0, 5e-3, numElements=[100]
    )

Upon performing the extrusion, GMSH will define any necessary surfaces and volumes for us. However, this means that the surface of the outer cylinder will have been defined twice. Therefore it is necessary to remove any duplicate elements via

.. code-block:: python

    gmsh.model.occ.remove_all_duplicates()

It is important that all points in our model are defined using the same characteristic length. Therefore we need to define a couple of points across the mesh to have the same `lc`. Here we have used points on the inner and outer tube perimeters, on both the front and back of the mesh:

.. code-block:: python

    inner_front_perimiter_point = gmsh.model.occ.addPoint(5e-3, 0, 5e-3, lc)
    inner_back_perimiter_point = gmsh.model.occ.addPoint(5e-3, 0, 0, lc)

    outer_front_perimiter_point = gmsh.model.occ.addPoint(10e-3, 0, 5e-3, lc)
    outer_back_perimiter_point = gmsh.model.occ.addPoint(10e-3, 0, 0, lc)

The model can then be synchronized:

.. code-block:: python

    gmsh.model.occ.synchronize()

At any point, the GMSH GUI can be opened by running the line

.. code-block:: python

    gmsh.fltk.run()

after synchronizing the model.

Running this command at this stage will open the GUI, displaying something that looks like this:

.. figure:: ../images/gmsh_tut_2.png
    :width: 400
    :align: center

To be used with FESTIM, it is necessary for us to define surface and volume markers.
If the element has been defined explicitly, this is as easy as doing the following:

.. code-block:: python

    id_number = 1
    gmsh.model.addPhysicalGroup(2, [base_surface, cylinder_surface], id_number, name="surface")

where the 2 indicates that this is a 2nd dimension element, and we have listed the surfaces that we would like to assign with this ID number.

However, as we generated the surfaces using an extrusion, it can be complicated to keep track of which element corresponds to what.
GMSH assigns the surface labels cyclically when performing the extrusion, so these element IDs could be directly extracted using code. However, it may be more straightforward and intuitive to open the GUI as before and analyze the surfaces manually.

After opening the GUI, again after synchronising and using `gmsh.fltk.run()`, go into 'Tools' then 'Options', and ensure that 'Surfaces' is checked under 'Geometry'.
This will make the surfaces are visible and selectable in the visualisation.

.. figure:: ../images/gmsh_tut_3.png
    :width: 400
    :align: center

We can then hover our mouse over each surface to see its information. For example, we can see that the front tungsten surface is defined as Plane 7, and borders the volume 1.

.. figure:: ../images/gmsh_tut_4.png
    :width: 400
    :align: center

We can now look at each surface and interface and assign the necessary IDs.

.. code-block:: python

    front_id = 1
    back_id = 2
    left_id = 3
    right_id = 4
    top_id = 5
    bottom_id = 6
    outer_cylinder_surface_id = 7
    inner_cylinder_surface_id = 8

    tungsten_id = 1
    cucrzr_id = 2

    gmsh.model.addPhysicalGroup(2, [7, 10], front_id, name="front")
    gmsh.model.addPhysicalGroup(2, [6, 9], back_id, name="back")
    gmsh.model.addPhysicalGroup(2, [1], left_id, name="left")
    gmsh.model.addPhysicalGroup(2, [3], right_id, name="right")
    gmsh.model.addPhysicalGroup(2, [4], top_id, name="top")
    gmsh.model.addPhysicalGroup(2, [2], bottom_id, name="bottom")
    gmsh.model.addPhysicalGroup(
        2, [5], outer_cylinder_surface_id, name="tungsten_cucrzr_interface"
    )
    gmsh.model.addPhysicalGroup(
        2, [8], inner_cylinder_surface_id, name="cucrzr_coolant_interface"
    )

    gmsh.model.addPhysicalGroup(3, [1], tungsten_id, name="tungsten")
    gmsh.model.addPhysicalGroup(3, [2], cucrzr_id, name="cucrzr")

.. note::

    Surface ids and volume ids are stored in separate meshtags, so a surface and a volume can share the same id.

The model must then be resynchronized before generating the mesh.

.. code-block:: python

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)

We have now created our mesh!

Reading the mesh with DOLFINx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DOLFINx can convert the gmsh model directly into a mesh and meshtags, without writing any intermediate file, with :code:`dolfinx.io.gmsh.model_to_mesh`.
The mesh is then wrapped in :class:`festim.Mesh` and the tags are given to the problem:

.. code-block:: python

    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio
    import festim as F

    mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3)
    gmsh.finalize()

    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh(mesh=mesh_data.mesh)
    my_model.volume_meshtags = mesh_data.cell_tags
    my_model.facet_meshtags = mesh_data.facet_tags

Alternatively, the mesh can be written to a ``.msh`` file with :code:`gmsh.write("my_mesh.msh")` and read later with :code:`gmshio.read_from_msh("my_mesh.msh", MPI.COMM_WORLD, rank=0, gdim=3)`, which returns the same kind of object.

Using the mesh in FESTIM
^^^^^^^^^^^^^^^^^^^^^^^^^

The subdomains are then defined with the ids of the physical groups, and a FESTIM simulation can be run:

.. code-block:: python

    tungsten = F.VolumeSubdomain(id=tungsten_id, material=F.Material(D_0=1, E_D=0))
    cucrzr = F.VolumeSubdomain(id=cucrzr_id, material=F.Material(D_0=5, E_D=0))
    top = F.SurfaceSubdomain(id=top_id)
    coolant = F.SurfaceSubdomain(id=inner_cylinder_surface_id)

    H = F.Species("H")

    my_model.subdomains = [tungsten, cucrzr, top, coolant]
    my_model.species = [H]
    my_model.temperature = 800
    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=top, value=1, species=H),
        F.FixedConcentrationBC(subdomain=coolant, value=0, species=H),
    ]
    my_model.exports = [F.SpeciesExport("mobile.bp", field=[H], subdomain=tungsten)]
    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    my_model.initialise()
    my_model.run()

This produces the following visualisation in Paraview:

.. figure:: ../images/gmsh_tut_5.png
    :width: 400
    :align: center

SALOME example
--------------

This is a step-by-step guide to meshing with `SALOME 9.12.0 <https://www.salome-platform.org/>`_.

Building the geometry in SALOME
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Open SALOME and create a new study.
2. Activate the Geometry module

.. figure:: ../images/salome_guide_1.png
    :width: 400
    :align: center

3. Create a first square by clicking "Create rectangular face". Keep the default parameters. Click "Apply and Close"

.. figure:: ../images/salome_guide_2.png
    :width: 400
    :align: center

4. Repeat the operation to create a second square
5. Translate the second square by clicking "Operations/Transformation/Translation"

.. figure:: ../images/salome_guide_3.png
    :width: 400
    :align: center

6. Make sure Face 2 is selected. Enter 100 for the Dx value. Click "Apply and Close"

.. figure:: ../images/salome_guide_4.png
    :width: 400
    :align: center

7. Create a compound by clicking "New Entity/Build/Compound" make sure Face_1 and Translation_1 are selected then click "Apply and Close".

.. figure:: ../images/salome_guide_5.png
    :width: 400
    :align: center

8. Create a group "New Entity/Group/Create group". In Shape Type, select the 2D surface. Name the group "left_volume". Make sure Compound_1 is selected.
Click on the left square and click "Add" (2 should appear in the white window). Click "Apply and Close".

.. figure:: ../images/salome_guide_6.png
    :width: 400
    :align: center

9. Repeat the operation to create a group "right_volume" with the right square (12 should appear in the white window).

10. Create another group "left_boundary" but this time in Shape Type select the 1D curve. Click on the left edge of the left square and click "Add". Click "Apply and Close".

.. figure:: ../images/salome_guide_7.png
    :width: 400
    :align: center

11. Repeat the operation to create a group "right_boundary" with the right edge of the right square. Your study should look like:

.. figure:: ../images/salome_guide_8.png
    :width: 400
    :align: center

12. Click on "Mesh" to activate the mesh module.

.. figure:: ../images/salome_guide_9.png
    :width: 400
    :align: center

13. Create a mesh by clicking "Mesh/Create Mesh".

14. Make sure Compound_1 is selected in "Geometry". Under the 2D tab, select "NETGEN 1D-2D" as algorithm.

.. figure:: ../images/salome_guide_10.png
    :width: 400
    :align: center

15. Next to "Hypothesis" click on the gear symbol. Select "NETGEN 2D Simple Parameters". Click Ok. Click "Apply and Close".

.. figure:: ../images/salome_guide_11.png
    :width: 400
    :align: center

    In the Objet Browser, under Mesh_1 you should see Groups of Edges and Groups of Faces, containing left_boundary, right_boundary, left_volume and right_volume.

16. Export the mesh to MED by right clicking on Mesh_1 in the Object Browser, then Export/MED file. Choose a location where you want to write your MED file and click Save.

.. figure:: ../images/salome_guide_12.png
    :width: 400
    :align: center

17. Convert mesh with meshio (at the time of writing we are using meshio 5.3)

.. code-block:: bash

    python convert_mesh.py

The script `convert_mesh.py` is:

.. code-block:: python

    import meshio
    import numpy as np


    def convert_med_to_xdmf(
        med_file,
        cell_file="mesh_domains.xdmf",
        facet_file="mesh_boundaries.xdmf",
        cell_type="tetra",
        facet_type="triangle",
    ):
        """Converts a MED mesh to XDMF files readable by FESTIM

        Args:
            med_file (str): the name of the MED file
            cell_file (str, optional): the name of the file containing the
                volume markers. Defaults to "mesh_domains.xdmf".
            facet_file (str, optional): the name of the file containing the
                surface markers. Defaults to "mesh_boundaries.xdmf".
            cell_type (str, optional): The topology of the cells. Defaults to "tetra".
            facet_type (str, optional): The topology of the facets. Defaults to "triangle".

        Returns:
            dict: the correspondance between markers and group names
        """
        msh = meshio.read(med_file)

        correspondance_dict = msh.cell_tags

        points = msh.points
        if cell_type in ("triangle", "quad"):
            # DOLFINx expects 2D points for a 2D mesh
            points = points[:, :2]

        for block_type, filename in [(cell_type, cell_file), (facet_type, facet_file)]:
            cells = np.concatenate(
                [block.data for block in msh.cells if block.type == block_type]
            )
            # SALOME writes negative tags: flip the sign so that ids are positive
            tags = -1 * np.concatenate(
                [
                    data
                    for block, data in zip(msh.cells, msh.cell_data["cell_tags"])
                    if block.type == block_type
                ]
            )
            meshio.write(
                filename,
                meshio.Mesh(
                    points=points,
                    cells=[(block_type, cells)],
                    cell_data={"tags": [tags]},
                ),
            )

        return correspondance_dict


    if __name__ == "__main__":
        filename = "Mesh_1.med"
        correspondance_dict = convert_med_to_xdmf(
            filename, cell_type="triangle", facet_type="line"
        )
        print(correspondance_dict)

Running this script produces mesh_domains.xdmf, mesh_boundaries.xdmf, mesh_domains.h5, mesh_boundaries.h5 and a dictionary of correspondance between the markers and the mesh entities:

.. code-block:: bash

    {-6: ['left_volume'], -7: ['right_volume'], -8: ['left_boundary'], -9: ['right_boundary']}

The correspondance dictionary can be used to assign the correct markers to the mesh.
Since the script flips the sign of the tags, the left volume is tagged with ID 6 and the right boundary is tagged with ID 9.

18. Inspect the produced XDMF files with Paraview using the XDMF3 S reader. The file mesh_domains.xdmf should look like:

.. figure:: ../images/salome_guide_13.png
    :width: 400
    :align: center


19. Test the mesh in FESTIM by running:

.. code-block:: python

    import festim as F

    my_model = F.HydrogenTransportProblem()

    my_model.mesh = F.MeshFromXDMF(
        volume_file="mesh_domains.xdmf", facet_file="mesh_boundaries.xdmf"
    )

    left_volume = F.VolumeSubdomain(id=6, material=F.Material(D_0=1, E_D=0))
    right_volume = F.VolumeSubdomain(id=7, material=F.Material(D_0=5, E_D=0))
    left_boundary = F.SurfaceSubdomain(id=8)
    right_boundary = F.SurfaceSubdomain(id=9)
    my_model.subdomains = [left_volume, right_volume, left_boundary, right_boundary]

    H = F.Species("H")
    my_model.species = [H]

    my_model.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left_boundary, value=1, species=H),
        F.FixedConcentrationBC(subdomain=right_boundary, value=0, species=H),
    ]

    my_model.temperature = 823

    my_model.exports = [F.SpeciesExport("mobile.bp", field=[H], subdomain=left_volume)]

    my_model.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        transient=False,
    )

    my_model.initialise()
    my_model.run()

20. The simulation should run without errors. The solute field can be visualised with Paraview.

.. figure:: ../images/salome_guide_14.png
    :width: 400
    :align: center

Meshing CAD files in SALOME
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have a CAD model, you can export it to a mesh with SALOME.

1. Create a new study
2. Activate the Geometry module
3. Import STEP file by clicking "File/Import/STEP"

.. figure:: ../images/salome_guide_cad_1.png
    :width: 400
    :align: center

4. By clicking "Fit to selection" you can see the imported geometry:

.. figure:: ../images/salome_guide_cad_2.png
    :width: 400
    :align: center

5. Create a partition just like in the previous example
6. Create groups of volumes and faces
7. Mesh the geometry
8. Export the mesh to MED
9. Convert the mesh to XDMF (don't forget to change the cell and facet types in the script)
