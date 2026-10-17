"""
2D plane-strain linear-elastic soil model under a strip footing (FEniCSx).

Assumed (illustrative) inputs - NOT experimentally validated:
    E, nu, load q, and the domain/footing size.

Examples (run from the project folder):
    python3 fenics_simulation/foundation_fem.py
    # The script prompts for model inputs; press Enter to accept each default.
    python3 fenics_simulation/foundation_fem.py --NX 200 --NY 100
    python3 fenics_simulation/foundation_fem.py --W 20 --H 10 --NX 200 --NY 100 --tag big
    python3 fenics_simulation/foundation_fem.py --show      # interactive PyVista window
"""
import argparse
import inspect
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem

# ---------------- Command-line parameters (SI units) ----------------
ap = argparse.ArgumentParser(description="Strip footing on elastic soil (FEniCSx)")
ap.add_argument("--W", type=float, help="soil width [m]")
ap.add_argument("--H", type=float, help="soil depth [m]")
ap.add_argument("--B", type=float, help="footing width [m], centred on top")
ap.add_argument("--NX", type=int, help="mesh divisions in x")
ap.add_argument("--NY", type=int, help="mesh divisions in y")
ap.add_argument("--E", type=float, help="Young's modulus [Pa]")
ap.add_argument("--NU", type=float, help="Poisson's ratio [-]")
ap.add_argument("--Q", type=float, help="footing pressure [Pa], downward")
ap.add_argument("--tag", type=str, default="", help="suffix for output files, e.g. 'fine'")
ap.add_argument("--show", action="store_true",
                help="open the interactive PyVista window (blocks until closed)")
args = ap.parse_args()

comm = MPI.COMM_WORLD


def prompt_float(label, default, minimum=None, maximum=None, max_inclusive=True):
    while True:
        raw = input(f"{label} [{default:g}]: ").strip()
        try:
            value = default if not raw else float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if not np.isfinite(value):
            print("Please enter a finite number.")
            continue
        if minimum is not None and value <= minimum:
            print(f"Value must be greater than {minimum}.")
            continue
        if maximum is not None and (value > maximum or (not max_inclusive and value == maximum)):
            relation = "at most" if max_inclusive else "less than"
            print(f"Value must be {relation} {maximum}.")
            continue
        return value


def prompt_int(label, default):
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        try:
            value = default if not raw else int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value < 1:
            print("Value must be at least 1.")
            continue
        return value


if comm.rank == 0:
    print("Enter simulation inputs (press Enter to use the default).")
    W = args.W if args.W is not None else prompt_float("Soil width W [m]", 10.0, minimum=0)
    H = args.H if args.H is not None else prompt_float("Soil depth H [m]", 5.0, minimum=0)
    B = args.B if args.B is not None else prompt_float(
        "Footing width B [m]", 2.0, minimum=0, maximum=W, max_inclusive=False)
    NX = args.NX if args.NX is not None else prompt_int("Mesh divisions NX", 100)
    NY = args.NY if args.NY is not None else prompt_int("Mesh divisions NY", 50)
    E = args.E if args.E is not None else prompt_float("Young's modulus E [MPa]", 20.0, minimum=0) * 1e6
    NU = args.NU if args.NU is not None else prompt_float(
        "Poisson's ratio nu", 0.3, minimum=0, maximum=0.5, max_inclusive=False)
    Q = args.Q if args.Q is not None else prompt_float("Footing pressure Q [kPa]", 100.0, minimum=0) * 1e3
    parameters = (W, H, B, NX, NY, E, NU, Q)
else:
    parameters = None

W, H, B, NX, NY, E, NU, Q = comm.bcast(parameters, root=0)
suffix = f"_{args.tag}" if args.tag else ""

if not np.isfinite([W, H, B, E, NU, Q]).all():
    raise SystemExit("All model inputs must be finite numbers.")
if W <= 0:
    raise SystemExit("Soil width W must be greater than 0.")
if H <= 0:
    raise SystemExit("Soil depth H must be greater than 0.")
if B <= 0 or B >= W:
    raise SystemExit("Footing width B must be greater than 0 and smaller than soil width W.")
if NX < 1 or NY < 1:
    raise SystemExit("Mesh divisions NX and NY must each be at least 1.")
if E <= 0:
    raise SystemExit("Young's modulus E must be greater than 0.")
if not (0.0 < NU < 0.5):
    raise SystemExit("Poisson's ratio NU must be between 0 and 0.5 (plane strain).")
if Q < 0:
    raise SystemExit("Footing pressure Q cannot be negative.")
# --------------------------------------------------------------------

msh = mesh.create_rectangle(
    comm, [np.array([0.0, 0.0]), np.array([W, H])], [NX, NY],
    cell_type=mesh.CellType.triangle,
)
tdim = msh.topology.dim
fdim = tdim - 1

# Vector P1 function space (handles API differences between versions)
try:
    V = fem.functionspace(msh, ("Lagrange", 1, (tdim,)))
except Exception:
    V = fem.VectorFunctionSpace(msh, ("Lagrange", 1))

# ---------------- Boundary identification ----------------
tol = 1e-8
msh.topology.create_connectivity(fdim, tdim)

bottom_facets = mesh.locate_entities_boundary(
    msh, fdim, lambda x: np.isclose(x[1], 0.0, atol=tol))
side_facets = mesh.locate_entities_boundary(
    msh, fdim, lambda x: np.isclose(x[0], 0.0, atol=tol) | np.isclose(x[0], W, atol=tol))
footing_facets = mesh.locate_entities_boundary(
    msh, fdim,
    lambda x: np.isclose(x[1], H, atol=tol) & (np.abs(x[0] - W / 2) <= B / 2 + tol))

# Mark footing facets so the load is applied only there
order = np.argsort(footing_facets)
facet_tags = mesh.meshtags(
    msh, fdim, footing_facets[order],
    np.full(len(footing_facets), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

# ---------------- Boundary conditions ----------------
# Bottom: fixed in x and y
u_zero = fem.Function(V)
u_zero.x.array[:] = 0.0
bc_bottom = fem.dirichletbc(
    u_zero, fem.locate_dofs_topological(V, fdim, bottom_facets))

# Sides: ux = 0 (roller), uy free
Vx = V.sub(0)
bc_sides = fem.dirichletbc(
    default_scalar_type(0.0),
    fem.locate_dofs_topological(Vx, fdim, side_facets), Vx)

bcs = [bc_bottom, bc_sides]

# ---------------- Variational problem (plane strain) ----------------
mu = E / (2.0 * (1.0 + NU))
lmbda = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u):
    return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(tdim)


u_ = ufl.TrialFunction(V)
v_ = ufl.TestFunction(V)
traction = fem.Constant(msh, default_scalar_type((0.0, -Q)))

a = ufl.inner(sigma(u_), epsilon(v_)) * ufl.dx
L = ufl.dot(traction, v_) * ds(1)

opts = {"ksp_type": "preonly", "pc_type": "lu"}
kwargs = {"bcs": bcs, "petsc_options": opts}
if "petsc_options_prefix" in inspect.signature(LinearProblem.__init__).parameters:
    kwargs["petsc_options_prefix"] = "foundation_"
problem = LinearProblem(a, L, **kwargs)
uh = problem.solve()
uh.name = "Displacement"

# ---------------- Results ----------------
u_arr = uh.x.array.reshape(-1, tdim)
max_settle = -u_arr[:, 1].min()
max_ux = np.abs(u_arr[:, 0]).max()
print(f"Domain: W={W} m, H={H} m | mesh: {NX}x{NY} (cell ~{W/NX:.3f} x {H/NY:.3f} m)")
print(f"Footing width: {B} m | pressure: {Q/1e3:.1f} kPa | E: {E/1e6:.1f} MPa | nu: {NU}")
print(f"Max vertical settlement : {max_settle * 1000:.3f} mm")
print(f"Max horizontal movement : {max_ux * 1000:.3f} mm")
print("NOTE: simulated values from assumed soil properties, not measured data.")

with io.XDMFFile(comm, f"foundation_displacement{suffix}.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(uh)

# ---------------- Settlement contour plot (always saved) ----------------
import matplotlib
if not args.show:
    matplotlib.use("Agg")  # no window, so loops and scripts never block
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

xy = V.tabulate_dof_coordinates()
tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
settle_mm = -u_arr[:, 1] * 1000

fig, ax = plt.subplots(figsize=(9, 5))
cs = ax.tricontourf(tri, settle_mm, levels=20, cmap="viridis")
fig.colorbar(cs, label="Vertical settlement [mm]")
ax.plot([W / 2 - B / 2, W / 2 + B / 2], [H, H], "r-", lw=4, label="Footing")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m] (top surface at y = H)")
ax.set_title(f"Soil settlement under strip footing (simulated), max {max_settle*1000:.2f} mm")
ax.set_aspect("equal")
ax.legend()
out_png = f"foundation_settlement{suffix}.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"Saved {out_png}")

# ---------------- Interactive PyVista plot (only with --show) ----------------
if args.show:
    try:
        import pyvista
        from dolfinx import plot
    except ImportError:
        pyvista = None
        print("PyVista is unavailable; showing the Matplotlib contour instead.")
        plt.show()

    if pyvista is not None:
        cells, types, x = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        u3 = np.zeros((x.shape[0], 3))
        u3[:, :tdim] = u_arr
        grid.point_data["u"] = u3
        grid.point_data["|u| [mm]"] = np.linalg.norm(u3, axis=1) * 1000
        factor = 0.1 * H / max(np.linalg.norm(u3, axis=1).max(), 1e-12)  # exaggerated
        warped = grid.warp_by_vector("u", factor=factor)

        p = pyvista.Plotter(off_screen=False)
        p.add_mesh(grid, style="wireframe", color="gray", opacity=0.3)
        p.add_mesh(warped, scalars="|u| [mm]", show_edges=False)
        p.view_xy()
        p.add_text(f"Deformed shape (x{factor:.0f} exaggeration)", font_size=10)
        p.show(screenshot=f"foundation_displacement{suffix}.png")
        print(f"Saved foundation_displacement{suffix}.png")
