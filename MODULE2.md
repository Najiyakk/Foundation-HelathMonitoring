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
conditions are fixed in the simulation script, apart from optional measured
soil density and water-content inputs for self-weight.

The Streamlit dashboard in `geoflux_web.py` runs locally and displays the
foundation, load areas, settlement comparisons, and—when available—the saved
FEM deformation field. Its editable load and soil inputs drive a fast
elastic settlement estimate; they do not rerun the saved 3D FEM simulation.
The dashboard calculates bulk and shear moduli from Young's modulus and
Poisson's ratio. Its sample moisture input is used for density conversion and
does not change the dashboard's settlement estimate. The 3D FEM can optionally
include soil self-weight using measured dry density and gravimetric water
content; moisture does not change the soil stiffness or elastic material
model. Dashboard inputs are not automatically passed to the FEM script.

## Scope and limitations

- The implemented FEM model is a single fixed-configuration run, not an
  automated sweep over load vectors or moisture levels.
- The optional FEM self-weight uses uniform soil bulk density calculated as
  dry density multiplied by `1 + water_content / 100`. It does not model
  pore pressure, saturation effects, or moisture-dependent stiffness, and is
  illustrative rather than calibrated.
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

To include soil self-weight, provide both measured dry density in kg/m³ and
gravimetric water content in percent:

```powershell
python foundation_3d_updated.py --dry-density 1600 --water-content 12
```

Without both options, the simulation retains its existing behavior and omits
soil self-weight. This moisture input changes the soil body force only; it
does not change Young's modulus or the dashboard's settlement estimate.

Run the local dashboard in an environment with the packages listed in
`requirements-web.txt`:

```powershell
python -m pip install -r requirements-web.txt
streamlit run geoflux_web.py
```
