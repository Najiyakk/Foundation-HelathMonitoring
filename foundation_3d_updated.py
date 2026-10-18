"""
GEOFLUX: 3D soil + L-shaped concrete footing + 3 columns
FEniCSx 0.11 + Gmsh

Geometry (SI units):
  Soil block: 0.60 x 0.40 x 0.30 m
  L footing: 0.30 m long arm, 0.20 m short arm
  Footing width: 0.06 m
  Footing thickness: 0.04 m
  Columns: 3 x (0.05 x 0.05 x 0.10 m)

Outputs:
  geoflux_foundation_3d.msh
  geoflux_displacement.xdmf

Note:
  Material parameters and loads are illustrative.
  Calibrate against experimental data before interpreting
  results as real foundation predictions.
"""

import gmsh
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, io, mesh
from dolfinx.io import gmsh as dolfinx_gmsh
from dolfinx.fem.petsc import LinearProblem

# -------------------- MPI --------------------

comm = MPI.COMM_WORLD
rank = comm.rank

# -------------------- DIMENSIONS --------------------

SOIL_X, SOIL_Y, SOIL_Z = 0.60, 0.40, 0.30

ARM_LONG = 0.30
ARM_SHORT = 0.20
FOOT_W = 0.06
FOOT_T = 0.04

COL_W = 0.05
COL_H = 0.10

FX, FY = 0.10, 0.10
FZ = SOIL_Z

COLUMN_POSITIONS = {
    "A": (0.115, 0.105),
    "B": (0.325, 0.105),
    "C": (0.105, 0.235),
}

# -------------------- MATERIAL PROPERTIES --------------------

# Illustrative values only
E_SOIL = 20.0e6
NU_SOIL = 0.30

E_CONC = 25.0e9
NU_CONC = 0.20

# Downward loads in Newtons
LOAD_A = 10.0
LOAD_B = 10.0
LOAD_C = 10.0

# Mesh sizes in metres
MESH_SIZE_SOIL = 0.035
MESH_SIZE_CONC = 0.018

# -------------------- VALIDATE GEOMETRY --------------------

if not (
    0 < FOOT_W < ARM_SHORT <= SOIL_Y
    and ARM_LONG <= SOIL_X
):
    raise ValueError(
        "Footing dimensions do not fit inside the soil block."
    )

for label, (cx, cy) in COLUMN_POSITIONS.items():

    horizontal_ok = (
        FX - 1e-9 <= cx
        and cx + COL_W <= FX + ARM_LONG + 1e-9
        and FY - 1e-9 <= cy
        and cy + COL_W <= FY + FOOT_W + 1e-9
    )

    vertical_ok = (
        FX - 1e-9 <= cx
        and cx + COL_W <= FX + FOOT_W + 1e-9
        and FY - 1e-9 <= cy
        and cy + COL_W <= FY + ARM_SHORT + 1e-9
    )

    if not (horizontal_ok or vertical_ok):
        raise ValueError(
            f"Column {label} is not fully supported by the L footing."
        )

# ============================================================
# BUILD GEOMETRY AND MESH USING GMSH
# ============================================================

if rank == 0:

    gmsh.initialize()
    gmsh.model.add("GEOFLUX_3D_foundation")

    occ = gmsh.model.occ

    # Soil block
    soil_tag = occ.addBox(
        0.0, 0.0, 0.0,
        SOIL_X, SOIL_Y, SOIL_Z
    )

    # L-shaped footing strips
    long_tag = occ.addBox(
        FX, FY, FZ,
        ARM_LONG, FOOT_W, FOOT_T
    )

    short_tag = occ.addBox(
        FX, FY, FZ,
        FOOT_W, ARM_SHORT, FOOT_T
    )

    # Fuse the two footing strips
    fused, _ = occ.fuse(
        [(3, long_tag)],
        [(3, short_tag)],
        removeObject=True,
        removeTool=True
    )

    footing_vols = [
        tag for dim, tag in fused if dim == 3
    ]

    # Three columns
    column_vols = []

    for label, (cx, cy) in COLUMN_POSITIONS.items():

        tag = occ.addBox(
            cx,
            cy,
            FZ + FOOT_T,
            COL_W,
            COL_W,
            COL_H
        )

        column_vols.append(tag)

    # Fragment soil and concrete volumes to create
    # a conforming mesh at the interfaces.
    concrete_entities = (
        [(3, v) for v in footing_vols]
        + [(3, v) for v in column_vols]
    )

    occ.fragment(
        [(3, soil_tag)],
        concrete_entities,
        removeObject=True,
        removeTool=True
    )

    occ.synchronize()

    # Identify soil and concrete volumes
    volumes = gmsh.model.getEntities(3)

    soil_volumes = []
    concrete_volumes = []

    soil_tol = 1e-7

    for dim, tag in volumes:

        cx, cy, cz = occ.getCenterOfMass(dim, tag)

        if cz < SOIL_Z - soil_tol:
            soil_volumes.append(tag)
        else:
            concrete_volumes.append(tag)

    if not soil_volumes or not concrete_volumes:

        gmsh.finalize()

        raise RuntimeError(
            "Could not identify soil and concrete volumes."
        )

    # Physical volume groups
    gmsh.model.addPhysicalGroup(
        3, soil_volumes, 1
    )
    gmsh.model.setPhysicalName(
        3, 1, "Soil"
    )

    gmsh.model.addPhysicalGroup(
        3, concrete_volumes, 2
    )
    gmsh.model.setPhysicalName(
        3, 2, "Concrete"
    )

    # --------------------------------------------------------
    # LOAD SURFACES: TOP OF EACH COLUMN
    # Tags: 11, 12, 13
    # --------------------------------------------------------

    for i, (label, (cx, cy)) in enumerate(
        COLUMN_POSITIONS.items(),
        start=11
    ):

        ztop = FZ + FOOT_T + COL_H

        surfaces = gmsh.model.getEntitiesInBoundingBox(
            cx - 1e-7,
            cy - 1e-7,
            ztop - 1e-7,
            cx + COL_W + 1e-7,
            cy + COL_W + 1e-7,
            ztop + 1e-7,
            2
        )

        valid = []

        for _, surface_tag in surfaces:

            bb = gmsh.model.getBoundingBox(
                2, surface_tag
            )

            is_top_face = (
                abs(bb[2] - ztop) < 1e-6
                and abs(bb[5] - ztop) < 1e-6
            )

            fits_column = (
                bb[0] >= cx - 1e-6
                and bb[1] >= cy - 1e-6
                and bb[3] <= cx + COL_W + 1e-6
                and bb[4] <= cy + COL_W + 1e-6
            )

            if is_top_face and fits_column:
                valid.append(surface_tag)

        if not valid:

            gmsh.finalize()

            raise RuntimeError(
                f"Could not find top load surface for column {label}."
            )

        gmsh.model.addPhysicalGroup(
            2, valid, i
        )

        gmsh.model.setPhysicalName(
            2, i, f"Load_{label}"
        )

    # --------------------------------------------------------
    # BOTTOM AND OUTER SIDE SURFACES
    # Bottom tag = 21
    # Side tag = 22
    # --------------------------------------------------------

    bottom = []
    sides = []

    for _, surface_tag in gmsh.model.getEntities(2):

        bb = gmsh.model.getBoundingBox(
            2, surface_tag
        )

        # Bottom plane: z = 0
        if (
            abs(bb[2]) < 1e-6
            and abs(bb[5]) < 1e-6
        ):
            bottom.append(surface_tag)

        # Outer x and y side planes
        is_outer_side = (
            (
                abs(bb[0]) < 1e-6
                and abs(bb[3]) < 1e-6
            )
            or (
                abs(bb[0] - SOIL_X) < 1e-6
                and abs(bb[3] - SOIL_X) < 1e-6
            )
            or (
                abs(bb[1]) < 1e-6
                and abs(bb[4]) < 1e-6
            )
            or (
                abs(bb[1] - SOIL_Y) < 1e-6
                and abs(bb[4] - SOIL_Y) < 1e-6
            )
        )

        if is_outer_side:
            sides.append(surface_tag)

    gmsh.model.addPhysicalGroup(
        2, bottom, 21
    )
    gmsh.model.setPhysicalName(
        2, 21, "Bottom"
    )

    gmsh.model.addPhysicalGroup(
        2, sides, 22
    )
    gmsh.model.setPhysicalName(
        2, 22, "Sides"
    )

    # --------------------------------------------------------
    # MESH GENERATION
    # --------------------------------------------------------

    for dim, tag in gmsh.model.getEntities(0):

        x, y, z = gmsh.model.getValue(
            0, tag, []
        )

        if z >= SOIL_Z - 1e-7:
            gmsh.model.mesh.setSize(
                [(0, tag)], MESH_SIZE_CONC
            )
        else:
            gmsh.model.mesh.setSize(
                [(0, tag)], MESH_SIZE_SOIL
            )

    gmsh.option.setNumber(
        "Mesh.CharacteristicLengthMin",
        min(MESH_SIZE_SOIL, MESH_SIZE_CONC)
    )

    gmsh.option.setNumber(
        "Mesh.CharacteristicLengthMax",
        max(MESH_SIZE_SOIL, MESH_SIZE_CONC)
    )

    gmsh.option.setNumber(
        "Mesh.ElementOrder", 1
    )

    gmsh.model.mesh.generate(3)

    gmsh.write(
        "geoflux_foundation_3d.msh"
    )

    gmsh.finalize()

# ============================================================
# READ GMSH MESH INTO FENICSX
# ============================================================

comm.barrier()

mesh_data = dolfinx_gmsh.read_from_msh(
    "geoflux_foundation_3d.msh",
    comm,
    rank=0,
    gdim=3
)

domain = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

tdim = domain.topology.dim
fdim = tdim - 1

# Three-component displacement space
V = fem.functionspace(
    domain,
    ("Lagrange", 1, (3,))
)

# ============================================================
# MATERIAL PROPERTIES
# ============================================================

Q0 = fem.functionspace(
    domain,
    ("DG", 0)
)

E_field = fem.Function(
    Q0, name="Youngs_modulus"
)

nu_field = fem.Function(
    Q0, name="Poissons_ratio"
)

if cell_tags is None:
    raise RuntimeError(
        "No physical cell tags found. "
        "Check Gmsh physical volume groups."
    )

# Assign properties to soil and concrete cells
for cell, tag in zip(
    cell_tags.indices,
    cell_tags.values
):

    dof = Q0.dofmap.cell_dofs(
        int(cell)
    )[0]

    if tag == 1:
        E_field.x.array[dof] = E_SOIL
        nu_field.x.array[dof] = NU_SOIL

    elif tag == 2:
        E_field.x.array[dof] = E_CONC
        nu_field.x.array[dof] = NU_CONC

# Lamé parameters
mu = E_field / (
    2.0 * (1.0 + nu_field)
)

lmbda = (
    E_field * nu_field
    / (
        (1.0 + nu_field)
        * (1.0 - 2.0 * nu_field)
    )
)

def eps(u):
    """Small-strain tensor."""
    return ufl.sym(ufl.grad(u))


def sigma(u):
    """Linear-elastic stress tensor."""
    return (
        2.0 * mu * eps(u)
        + lmbda
        * ufl.tr(eps(u))
        * ufl.Identity(3)
    )

# ============================================================
# BOUNDARY CONDITIONS
# ============================================================

if facet_tags is None:
    raise RuntimeError(
        "No physical facet tags found in the mesh."
    )

bottom_facets = facet_tags.find(21)
side_facets = facet_tags.find(22)

if len(bottom_facets) == 0:
    raise RuntimeError(
        "Bottom boundary tag 21 is empty."
    )

if len(side_facets) == 0:
    raise RuntimeError(
        "Side boundary tag 22 is empty."
    )

# Fix the bottom in all three directions
u_zero = fem.Function(V)
u_zero.x.array[:] = 0.0

bc_bottom = fem.dirichletbc(
    u_zero,
    fem.locate_dofs_topological(
        V, fdim, bottom_facets
    )
)

# Restrain normal displacement at the outer sides
x0f = mesh.locate_entities_boundary(
    domain,
    fdim,
    lambda x: np.isclose(x[0], 0.0)
)

xXf = mesh.locate_entities_boundary(
    domain,
    fdim,
    lambda x: np.isclose(x[0], SOIL_X)
)

y0f = mesh.locate_entities_boundary(
    domain,
    fdim,
    lambda x: np.isclose(x[1], 0.0)
)

yYf = mesh.locate_entities_boundary(
    domain,
    fdim,
    lambda x: np.isclose(x[1], SOIL_Y)
)

bc_x0 = fem.dirichletbc(
    PETSc.ScalarType(0.0),
    fem.locate_dofs_topological(
        V.sub(0), fdim, x0f
    ),
    V.sub(0)
)

bc_xX = fem.dirichletbc(
    PETSc.ScalarType(0.0),
    fem.locate_dofs_topological(
        V.sub(0), fdim, xXf
    ),
    V.sub(0)
)

bc_y0 = fem.dirichletbc(
    PETSc.ScalarType(0.0),
    fem.locate_dofs_topological(
        V.sub(1), fdim, y0f
    ),
    V.sub(1)
)

bc_yY = fem.dirichletbc(
    PETSc.ScalarType(0.0),
    fem.locate_dofs_topological(
        V.sub(1), fdim, yYf
    ),
    V.sub(1)
)

bcs = [
    bc_bottom,
    bc_x0,
    bc_xX,
    bc_y0,
    bc_yY
]

# ============================================================
# APPLY LOADS TO COLUMN TOPS
# ============================================================

ds = ufl.Measure(
    "ds",
    domain=domain,
    subdomain_data=facet_tags
)

v = ufl.TestFunction(V)
u = ufl.TrialFunction(V)

load_map = {
    11: LOAD_A,
    12: LOAD_B,
    13: LOAD_C
}

L = 0

for tag, force in load_map.items():

    # Pressure = force / column top area
    pressure = force / (
        COL_W * COL_W
    )

    # Downward traction in the z direction
    traction = fem.Constant(
        domain,
        PETSc.ScalarType(
            (0.0, 0.0, -pressure)
        )
    )

    L += ufl.dot(
        traction, v
    ) * ds(tag)

# ============================================================
# SOLVE LINEAR ELASTICITY
# ============================================================

a = ufl.inner(
    sigma(u), eps(v)
) * ufl.dx

problem = LinearProblem(
    a,
    L,
    bcs=bcs,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu"
    },
    petsc_options_prefix="geoflux3d_"
)

uh = problem.solve()
uh.name = "Displacement"

# ============================================================
# SAVE DISPLACEMENT RESULTS
# ============================================================

with io.XDMFFile(
    comm,
    "geoflux_displacement.xdmf",
    "w"
) as xdmf:

    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

# ============================================================
# CALCULATE DISPLACEMENT
# ============================================================

u_local = uh.x.array.reshape(
    (-1, 3)
)

max_settlement = max(
    0.0,
    -float(np.min(u_local[:, 2])) * 1000.0
)

max_horizontal = float(
    np.max(
        np.sqrt(
            u_local[:, 0] ** 2
            + u_local[:, 1] ** 2
        )
    )
) * 1000.0

# ============================================================
# PRINT RESULTS
# ============================================================

if rank == 0:

    print("\nGEOFLUX 3D FEM completed.")

    print(
        f"Soil block: {SOIL_X:.2f} x "
        f"{SOIL_Y:.2f} x {SOIL_Z:.2f} m"
    )

    print(
        f"L footing arms: {ARM_LONG:.2f} m "
        f"and {ARM_SHORT:.2f} m; "
        f"width={FOOT_W:.2f} m; "
        f"thickness={FOOT_T:.2f} m"
    )

    print(
        f"Columns: {COL_W:.2f} x "
        f"{COL_W:.2f} x {COL_H:.2f} m"
    )

    print(
        f"Maximum vertical settlement: "
        f"{max_settlement:.6f} mm"
    )

    print(
        f"Maximum horizontal displacement magnitude: "
        f"{max_horizontal:.6f} mm"
    )

    print(
        "WARNING: Results depend on the assumed "
        "material properties, boundary conditions, "
        "mesh and applied loads."
    )