# Module 2: Numerical Simulation and Synthetic Data (FEM)

## Current implementation

GEOFLUX includes a three-dimensional, linear-elastic foundation simulation built
with FEniCSx and Gmsh. The model contains a soil block, an L-shaped concrete
footing, and three columns. It applies downward loads to the column tops and
solves for displacement using isotropic elastic material properties
(Young's modulus and Poisson's ratio) for soil and concrete.

The simulation writes the generated mesh to `geoflux_foundation_3d.msh` and
the displacement field to `geoflux_displacement.xdmf` (with its HDF5 data
file). It also reports maximum vertical settlement and horizontal
displacement. The model's geometry, material properties, loads, and boundary
conditions are currently fixed in the simulation script.

The Streamlit dashboard in `geoflux_web.py` runs locally and displays the
foundation, load areas, settlement comparisons, and—when available—the saved
FEM deformation field. Its editable load and soil inputs drive a fast
elastic settlement estimate; they do not rerun the saved 3D FEM simulation.
The dashboard calculates bulk and shear moduli from Young's modulus and
Poisson's ratio. Sample moisture can be entered for density conversion, but
moisture is not currently a parameter in the FEM material model.

## Scope and limitations

- The implemented FEM model is a single fixed-configuration run, not an
  automated sweep over load vectors or moisture levels.
- The dashboard's estimated settlement and strain proxy are not local
  stress/strain fields extracted from the FEM solution.
- There is currently no batch-generated CSV/NumPy training dataset or
  dataset-management view.
- Material parameters and loads are illustrative. The model is not
  experimentally calibrated or suitable as a substitute for engineering
  design or review.

## Running the current implementation

Run the 3D simulation in an environment with FEniCSx, Gmsh, MPI, PETSc, and
their Python dependencies:

```powershell
python foundation_3d_updated.py
```

Run the local dashboard in an environment with the packages listed in
`requirements-web.txt`:

```powershell
python -m pip install -r requirements-web.txt
streamlit run geoflux_web.py
```
