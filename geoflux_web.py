from pathlib import Path
from collections.abc import Mapping
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import plotly.graph_objects as go
import streamlit as st


FOOT_X = 0.10
FOOT_Y = 0.10
ARM_LONG = 0.30
ARM_SHORT = 0.20
FOOT_WIDTH = 0.06
SOIL_SURFACE_Z = 0.30
FOOTING_TOP_Z = SOIL_SURFACE_Z + 0.04
SOIL_MODULUS_MPA = 20.0
POISSON_RATIO_DEFAULT = 0.30
STANDARD_GRAVITY_M_S2 = 9.80665
LOAD_DEFAULTS = (
    ("Load A", 0.125, 0.13),
    ("Load B", 0.275, 0.13),
    ("Load C", 0.13, 0.225),
)
QUADRATURE_NODES, QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(10)


def _hdf_reference(element: ET.Element | None) -> tuple[str, str]:
    if element is None or element.text is None:
        raise ValueError("The XDMF file is missing a required data item.")

    if element.get("Format", "").upper() != "HDF":
        raise ValueError("Only HDF-backed XDMF data is supported.")

    reference = element.text.strip()
    try:
        hdf_name, dataset_name = reference.rsplit(":", 1)
    except ValueError as error:
        raise ValueError(f"Invalid HDF data reference: {reference}") from error

    if not hdf_name or not dataset_name:
        raise ValueError(f"Invalid HDF data reference: {reference}")

    return hdf_name, dataset_name


def load_mesh(
    xdmf_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    xdmf = ET.parse(xdmf_path).getroot()
    topology_item = xdmf.find(".//Grid[@Name='mesh']/Topology/DataItem")
    geometry_item = xdmf.find(".//Grid[@Name='mesh']/Geometry/DataItem")

    topology_file, topology_dataset = _hdf_reference(topology_item)
    geometry_file, geometry_dataset = _hdf_reference(geometry_item)
    displacement_item = xdmf.find(
        ".//Grid[@Name='Displacement']/Grid/Attribute[@Name='Displacement']/DataItem"
    )
    displacement_reference = (
        _hdf_reference(displacement_item)
        if displacement_item is not None
        else None
    )
    topology_path = (xdmf_path.parent / topology_file).resolve()
    geometry_path = (xdmf_path.parent / geometry_file).resolve()

    if topology_path != geometry_path:
        raise ValueError("The XDMF mesh must use one HDF5 file.")
    if displacement_reference is not None:
        displacement_file, _ = displacement_reference
        displacement_path = (xdmf_path.parent / displacement_file).resolve()
        if displacement_path != topology_path:
            raise ValueError("The XDMF displacement must use the mesh HDF5 file.")

    with h5py.File(topology_path, "r") as hdf:
        tetrahedra = np.asarray(hdf[topology_dataset], dtype=np.int64)
        points = np.asarray(hdf[geometry_dataset], dtype=np.float64)
        displacement = (
            np.asarray(hdf[displacement_reference[1]], dtype=np.float64)
            if displacement_reference is not None
            else None
        )

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("The XDMF geometry must contain 3D points.")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise ValueError("The foundation mesh must contain tetrahedral cells.")
    if tetrahedra.size == 0 or len(points) == 0:
        raise ValueError("The XDMF file contains an empty foundation mesh.")
    if tetrahedra.min() < 0 or tetrahedra.max() >= len(points):
        raise ValueError("The foundation mesh contains invalid point indices.")
    if displacement is not None and (
        displacement.shape != points.shape
        or not np.isfinite(displacement).all()
    ):
        raise ValueError(
            "The saved FEM displacement must contain a finite 3D vector "
            "for every mesh point."
        )

    return points[:, :3], tetrahedra, displacement


def foundation_surface_triangles(tetrahedra: np.ndarray, points: np.ndarray) -> np.ndarray:
    faces = np.concatenate(
        (
            tetrahedra[:, [0, 1, 2]],
            tetrahedra[:, [0, 1, 3]],
            tetrahedra[:, [0, 2, 3]],
            tetrahedra[:, [1, 2, 3]],
        )
    )
    sorted_faces = np.sort(faces, axis=1)
    _, first_indices, counts = np.unique(
        sorted_faces, axis=0, return_index=True, return_counts=True
    )
    exterior_faces = faces[first_indices[counts == 1]]
    face_heights = points[exterior_faces, 2].mean(axis=1)

    # The saved mesh has old columns; omit their exposed faces from this view.
    return exterior_faces[face_heights <= FOOTING_TOP_Z + 1e-8]


def foundation_face_mask(
    surface_triangles: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    triangle_points = points[surface_triangles]
    centers = triangle_points.mean(axis=1)
    x, y = centers[:, 0], centers[:, 1]
    on_footing_plan = (
        (
            (x >= FOOT_X - 1e-8)
            & (x <= FOOT_X + ARM_LONG + 1e-8)
            & (y >= FOOT_Y - 1e-8)
            & (y <= FOOT_Y + FOOT_WIDTH + 1e-8)
        )
        | (
            (x >= FOOT_X - 1e-8)
            & (x <= FOOT_X + FOOT_WIDTH + 1e-8)
            & (y >= FOOT_Y - 1e-8)
            & (y <= FOOT_Y + ARM_SHORT + 1e-8)
        )
    )
    above_soil_surface = np.all(
        triangle_points[:, :, 2] >= SOIL_SURFACE_Z - 1e-8,
        axis=1,
    )
    return on_footing_plan & above_soil_surface


def patch_fits_footing(
    center_x: float,
    center_y: float,
    width_x: float,
    width_y: float,
) -> bool:
    x_min = center_x - width_x / 2
    x_max = center_x + width_x / 2
    y_min = center_y - width_y / 2
    y_max = center_y + width_y / 2
    tolerance = 1e-9
    within_bounds = (
        x_min >= FOOT_X - tolerance
        and x_max <= FOOT_X + ARM_LONG + tolerance
        and y_min >= FOOT_Y - tolerance
        and y_max <= FOOT_Y + ARM_SHORT + tolerance
    )
    on_long_arm = (
        y_min >= FOOT_Y - tolerance
        and y_max <= FOOT_Y + FOOT_WIDTH + tolerance
    )
    on_short_arm = (
        x_min >= FOOT_X - tolerance
        and x_max <= FOOT_X + FOOT_WIDTH + tolerance
    )
    return within_bounds and (on_long_arm or on_short_arm)


def settlement_at_points(
    points: np.ndarray,
    loads: list[dict[str, float | str]],
    soil_modulus_mpa: float,
    poisson_ratio: float,
) -> np.ndarray:
    soil_modulus_pa = soil_modulus_mpa * 1_000_000
    depth = np.maximum(0.0, SOIL_SURFACE_Z - points[:, 2])
    settlement_m = np.zeros(len(points), dtype=np.float64)

    for load in loads:
        width_x = float(load["width_x"])
        width_y = float(load["width_y"])
        center_x = float(load["x"])
        center_y = float(load["y"])
        pressure_pa = (
            float(load["force_kn"]) * 1000 / (width_x * width_y)
        )
        source_x = (
            center_x
            + QUADRATURE_NODES * width_x / 2
        )
        source_y = (
            center_y
            + QUADRATURE_NODES * width_y / 2
        )
        source_weights = (
            QUADRATURE_WEIGHTS[:, None]
            * QUADRATURE_WEIGHTS[None, :]
            * width_x
            * width_y
            / 4
        )
        source_x_grid, source_y_grid = np.meshgrid(source_x, source_y)
        source_weight_grid = source_weights.ravel()
        dx = points[:, 0, None] - source_x_grid.ravel()[None, :]
        dy = points[:, 1, None] - source_y_grid.ravel()[None, :]
        radius = np.sqrt(
            dx**2 + dy**2 + depth[:, None] ** 2
        )
        radius = np.maximum(radius, 1e-12)
        kernel = (
            2 * (1 - poisson_ratio) / radius
            + depth[:, None] ** 2 / radius**3
        )
        settlement_m += (
            (1 + poisson_ratio)
            * pressure_pa
            / (2 * np.pi * soil_modulus_pa)
            * (kernel @ source_weight_grid)
        )

    return settlement_m


def settlement_at_load_centers(
    loads: list[dict[str, float | str]],
    soil_modulus_mpa: float,
    poisson_ratio: float,
) -> np.ndarray:
    centers = np.array(
        [
            [float(load["x"]), float(load["y"]), FOOTING_TOP_Z]
            for load in loads
        ],
        dtype=np.float64,
    )
    return settlement_at_points(
        centers,
        loads,
        soil_modulus_mpa,
        poisson_ratio,
    ) * 1000


def load_outline(load: dict[str, float | str]) -> tuple[list[float], list[float]]:
    half_x = float(load["width_x"]) / 2
    half_y = float(load["width_y"]) / 2
    center_x = float(load["x"])
    center_y = float(load["y"])
    return (
        [
            center_x - half_x,
            center_x + half_x,
            center_x + half_x,
            center_x - half_x,
            center_x - half_x,
        ],
        [
            center_y - half_y,
            center_y - half_y,
            center_y + half_y,
            center_y + half_y,
            center_y - half_y,
        ],
    )


st.set_page_config(
    page_title="GEOFLUX Digital Twin",
    page_icon="🏗️",
    layout="wide",
)
st.title("🏗️ GEOFLUX — Foundation Health Monitoring")
st.write("Place named loads on the footing and compare estimated settlement.")

with st.sidebar:
    st.header("Foundation loads")
    load_count = st.number_input(
        "Number of load areas",
        min_value=1,
        max_value=12,
        value=3,
        step=1,
    )
    soil_modulus_mpa = st.number_input(
        "Soil stiffness E (MPa)",
        min_value=0.1,
        max_value=1000.0,
        value=SOIL_MODULUS_MPA,
        step=1.0,
        help="Illustrative soil stiffness. Use a measured value for engineering analysis.",
    )
    poisson_ratio = st.number_input(
        "Poisson's ratio",
        min_value=0.0,
        max_value=0.49,
        value=POISSON_RATIO_DEFAULT,
        step=0.01,
        help="Used with Young's modulus for elastic estimates.",
    )
    show_deformation_cage = st.checkbox(
        "Show saved FEM deformation cage",
        value=True,
        help="Overlay the original FEM mesh deformed by its saved displacement field.",
    )
    deformation_scale = st.number_input(
        "Deformation display scale",
        min_value=0.0,
        max_value=10000.0,
        value=1000.0,
        step=100.0,
        help="Visual exaggeration only. A value of 1000 magnifies saved displacement 1000 times.",
    )
    with st.expander("Project and soil measurements", expanded=False):
        record_sample = st.checkbox(
            "Enter sample mass and volume to calculate density",
            key="record_sample",
        )
        sample_mass_kg = 0.0
        sample_volume_l = 0.0
        mass_basis = "Wet mass"
        water_content_pct: float | None = None
        if record_sample:
            sample_mass_kg = st.number_input(
                "Sample mass (kg)",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key="sample_mass_kg",
            )
            sample_volume_l = st.number_input(
                "Sample volume (L)",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key="sample_volume_l",
            )
            mass_basis = st.selectbox(
                "Mass basis",
                ("Wet mass", "Dry mass"),
                key="sample_mass_basis",
                help="Select whether the recorded sample mass includes its water.",
            )
            water_content_measured = st.checkbox(
                "Water content is measured",
                key="water_content_measured",
            )
            if water_content_measured:
                water_content_pct = st.number_input(
                    "Water content (%)",
                    min_value=0.0,
                    max_value=500.0,
                    value=0.0,
                    step=0.5,
                    key="water_content_pct",
                    help="Gravimetric water content: water mass divided by dry soil mass.",
                )
    st.caption(
        "Footing plan: X = 0.10–0.40 m, Y = 0.10–0.30 m. "
        "Click a point on either footing arm to place a load."
    )

if "load_positions" not in st.session_state:
    st.session_state.load_positions = [
        (x, y) for _, x, y in LOAD_DEFAULTS
    ]
while len(st.session_state.load_positions) < int(load_count):
    index = len(st.session_state.load_positions)
    _, x, y = LOAD_DEFAULTS[index % len(LOAD_DEFAULTS)]
    st.session_state.load_positions.append((x, y))
st.session_state.load_positions = st.session_state.load_positions[
    : int(load_count)
]

loads: list[dict[str, float | str]] = []
for index in range(int(load_count)):
    default_name, _, _ = LOAD_DEFAULTS[index % len(LOAD_DEFAULTS)]
    letter = chr(ord("A") + index)
    with st.sidebar.expander(f"Load {letter}", expanded=index == 0):
        name = st.text_input(
            "Name",
            value=default_name if index < len(LOAD_DEFAULTS) else f"Load {letter}",
            key=f"load_name_{index}",
        ).strip()
        load_input_type = st.selectbox(
            "Enter load as",
            ("Force (kN)", "Mass (kg)"),
            key=f"load_input_type_{index}",
            help=(
                "Mass is converted to downward force using standard gravity "
                f"({STANDARD_GRAVITY_M_S2:g} m/s²)."
            ),
        )
        if load_input_type == "Mass (kg)":
            entered_load = st.number_input(
                "Mass / weight (kg)",
                min_value=0.0,
                max_value=1_000_000_000.0,
                value=1000.0,
                step=100.0,
                key=f"load_mass_{index}",
                help="Mass in kilograms; converted to force using standard gravity.",
            )
            force_kn = entered_load * STANDARD_GRAVITY_M_S2 / 1000
        else:
            entered_load = st.number_input(
                "Downward load (kN)",
                min_value=0.0,
                max_value=10000.0,
                value=10.0,
                step=1.0,
                key=f"load_force_{index}",
            )
            force_kn = entered_load
        col_width, col_depth = st.columns(2)
        width_x = col_width.number_input(
            "Area width X (m)",
            min_value=0.005,
            max_value=ARM_LONG,
            value=0.05,
            step=0.005,
            format="%.3f",
            key=f"load_width_{index}",
        )
        width_y = col_depth.number_input(
            "Area width Y (m)",
            min_value=0.005,
            max_value=ARM_SHORT,
            value=0.04,
            step=0.005,
            format="%.3f",
            key=f"load_depth_{index}",
        )
        loads.append(
            {
                "name": name or f"Load {letter}",
                "force_kn": force_kn,
                "input_type": load_input_type,
                "entered_load": entered_load,
                "x": st.session_state.load_positions[index][0],
                "y": st.session_state.load_positions[index][1],
                "width_x": width_x,
                "width_y": width_y,
            }
        )

validation_errors: list[str] = []
names = [str(load["name"]).casefold() for load in loads]
if len(set(names)) != len(names):
    validation_errors.append("Each load needs a unique name.")
for load in loads:
    if not patch_fits_footing(
        float(load["x"]),
        float(load["y"]),
        float(load["width_x"]),
        float(load["width_y"]),
    ):
        validation_errors.append(
            f"{load['name']}: the full load area must fit on one of the L-shaped footing arms."
        )

if validation_errors:
    for message in validation_errors:
        st.error(message)
    st.stop()

st.subheader("Click the footing to place a load")
load_colors = (
    "#ffffff",
    "#ff5a36",
    "#ffe14a",
    "#40e0d0",
    "#ff69b4",
    "#a78bfa",
)
active_load_index = st.selectbox(
    "Place click for load",
    options=range(len(loads)),
    format_func=lambda index: str(loads[index]["name"]),
    key="active_load_index",
)
active_load = loads[active_load_index]
grid_x = np.arange(
    FOOT_X + float(active_load["width_x"]) / 2,
    FOOT_X + ARM_LONG - float(active_load["width_x"]) / 2 + 1e-9,
    0.01,
)
grid_y = np.arange(
    FOOT_Y + float(active_load["width_y"]) / 2,
    FOOT_Y + ARM_SHORT - float(active_load["width_y"]) / 2 + 1e-9,
    0.01,
)
candidate_centers = [
    (float(x), float(y))
    for x in grid_x
    for y in grid_y
    if patch_fits_footing(
        float(x),
        float(y),
        float(active_load["width_x"]),
        float(active_load["width_y"]),
    )
]
plan_fig = go.Figure()
plan_fig.add_trace(
    go.Scatter(
        x=[FOOT_X, FOOT_X + ARM_LONG, FOOT_X + ARM_LONG,
           FOOT_X + FOOT_WIDTH, FOOT_X + FOOT_WIDTH, FOOT_X],
        y=[FOOT_Y, FOOT_Y, FOOT_Y + FOOT_WIDTH,
           FOOT_Y + FOOT_WIDTH, FOOT_Y + ARM_SHORT, FOOT_Y + ARM_SHORT],
        fill="toself",
        fillcolor="#b8a27a",
        line=dict(color="#665537", width=3),
        mode="lines",
        name="Concrete footing",
        hoverinfo="skip",
    )
)
plan_fig.add_trace(
    go.Scatter(
        x=[x for x, _ in candidate_centers],
        y=[y for _, y in candidate_centers],
        mode="markers",
        marker=dict(size=8, color="#4c566a", opacity=0.3),
        name="Click to place load",
        hovertemplate="Place load at X=%{x:.3f} m, Y=%{y:.3f} m<extra></extra>",
    )
)
for index, load in enumerate(loads):
    color = load_colors[index % len(load_colors)]
    plan_fig.add_trace(
        go.Scatter(
            x=[float(load["x"])],
            y=[float(load["y"])],
            mode="markers+text",
            marker=dict(size=14, color=color, line=dict(color="#222222", width=2)),
            text=[str(load["name"])],
            textposition="top center",
            name=str(load["name"]),
            hovertemplate=(
                f"{load['name']}<br>"
                f"Load: {float(load['force_kn']):.3f} kN"
                "<extra></extra>"
            ),
        )
    )
plan_fig.update_layout(
    height=440,
    clickmode="event+select",
    showlegend=True,
    xaxis=dict(title="X (m)", range=[FOOT_X - 0.02, FOOT_X + ARM_LONG + 0.02]),
    yaxis=dict(
        title="Y (m)",
        range=[FOOT_Y - 0.02, FOOT_Y + ARM_SHORT + 0.02],
        scaleanchor="x",
        scaleratio=1,
    ),
    margin=dict(l=0, r=0, t=20, b=0),
)
plan_event = st.plotly_chart(
    plan_fig,
    key="load_placement_plan",
    on_select="rerun",
    selection_mode="points",
    width="stretch",
)
if isinstance(plan_event, Mapping):
    selection = plan_event.get("selection", {})
    selected_points = (
        selection.get("points", [])
        if isinstance(selection, Mapping)
        else []
    )
else:
    selection = getattr(plan_event, "selection", None)
    selected_points = selection.points if selection is not None else []
if selected_points:
    selected_point = selected_points[-1]
    if "x" in selected_point and "y" in selected_point:
        new_position = (
            float(selected_point["x"]),
            float(selected_point["y"]),
        )
        if patch_fits_footing(
            new_position[0],
            new_position[1],
            float(active_load["width_x"]),
            float(active_load["width_y"]),
        ) and new_position != st.session_state.load_positions[active_load_index]:
            st.session_state.load_positions[active_load_index] = new_position
            st.rerun()

mesh_file = Path(__file__).resolve().parent / "geoflux_displacement.xdmf"
try:
    points, tetrahedra, fem_displacement = load_mesh(mesh_file)
    surface = foundation_surface_triangles(tetrahedra, points)
except (ET.ParseError, OSError, KeyError, ValueError) as error:
    st.error(f"Could not load the foundation mesh: {error}")
    st.info(
        "Keep geoflux_displacement.xdmf and its referenced HDF5 file "
        "together in the project folder."
    )
    st.stop()

if show_deformation_cage and fem_displacement is None:
    st.warning(
        "The saved mesh does not contain a FEM displacement field, so the "
        "deformation cage cannot be displayed."
    )

foundation_mask = foundation_face_mask(surface, points)
foundation_surface = surface[foundation_mask]
soil_surface = surface[~foundation_mask]
if not len(foundation_surface) or not len(soil_surface):
    st.error(
        "Could not separate the concrete footing from the surrounding soil "
        "in the saved mesh."
    )
    st.stop()

st.subheader("Project parameters")
st.caption(
    "Edit Value and Unit / format cells directly. Length units accept mm, cm, "
    "or m. Blank source, classification, reference, and notes are optional. "
    "Container dimensions scale the soil view; foundation dimensions scale "
    "the footing view. This visual scaling does not remesh or rerun FEM."
)
parameter_editor_data = [
    {"Parameter": "Project", "Value": "GEOFLUX — Foundation Health Monitoring", "Unit / format": "Project information"},
    {"Parameter": "Soil source / location", "Value": "", "Unit / format": "Supplier, site, sample/batch"},
    {"Parameter": "Soil classification", "Value": "", "Unit / format": "Test-based classification"},
    {"Parameter": "Soil condition", "Value": "Unknown", "Unit / format": "Dry / moist / saturated"},
    {"Parameter": "Foundation long dimension", "Value": f"{ARM_LONG * 1000:g}", "Unit / format": "mm"},
    {"Parameter": "Foundation short dimension", "Value": f"{ARM_SHORT * 1000:g}", "Unit / format": "mm"},
    {"Parameter": "Foundation thickness", "Value": "40", "Unit / format": "mm"},
    {"Parameter": "Embedment depth", "Value": "0", "Unit / format": "mm"},
    {"Parameter": "Container length", "Value": "600", "Unit / format": "mm"},
    {"Parameter": "Container width", "Value": "400", "Unit / format": "mm"},
    {"Parameter": "Soil fill depth", "Value": f"{SOIL_SURFACE_Z * 1000:g}", "Unit / format": "mm"},
    {"Parameter": "Soil property reference", "Value": "", "Unit / format": "Text / URL"},
    {"Parameter": "Assumptions / notes", "Value": "", "Unit / format": "Text"},
]
parameter_editor = st.data_editor(
    parameter_editor_data,
    key="project_parameters_editor",
    hide_index=True,
    width="stretch",
    num_rows="fixed",
    column_config={
        "Parameter": st.column_config.TextColumn(disabled=True),
        "Value": st.column_config.TextColumn(),
        "Unit / format": st.column_config.TextColumn(),
    },
)
parameter_rows = (
    parameter_editor
    if isinstance(parameter_editor, list)
    else parameter_editor.to_dict("records")
)
parameter_values = {
    str(row["Parameter"]): str(row["Value"]).strip()
    for row in parameter_rows
}
parameter_units = {
    str(row["Parameter"]): str(row["Unit / format"]).strip()
    for row in parameter_rows
}


def parameter_number(label: str, *, minimum: float = 0.0) -> float:
    raw_value = parameter_values[label]
    try:
        value = float(raw_value)
    except ValueError as error:
        st.error(f"{label} must be a number; received {raw_value!r}.")
        st.stop()
        raise error
    if not np.isfinite(value) or value < minimum:
        st.error(f"{label} must be a finite value of at least {minimum:g}.")
        st.stop()
    return value


def parameter_length_mm(label: str, *, minimum: float = 0.0) -> float:
    unit = parameter_units[label].casefold()
    unit_to_mm = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
    if unit not in unit_to_mm:
        st.error(f"{label}: use mm, cm, or m in the Unit / format cell.")
        st.stop()
    return parameter_number(label, minimum=minimum) * unit_to_mm[unit]


foundation_length_mm = parameter_length_mm("Foundation long dimension", minimum=0.001)
foundation_width_mm = parameter_length_mm("Foundation short dimension", minimum=0.001)
foundation_thickness_mm = parameter_length_mm("Foundation thickness", minimum=0.001)
embedment_depth_mm = parameter_length_mm("Embedment depth")
container_length_mm = parameter_length_mm("Container length", minimum=0.001)
container_width_mm = parameter_length_mm("Container width", minimum=0.001)
soil_fill_depth_mm = parameter_length_mm("Soil fill depth", minimum=0.001)
if embedment_depth_mm > soil_fill_depth_mm:
    st.error("Embedment depth cannot exceed the entered soil fill depth.")
    st.stop()
if foundation_length_mm + FOOT_X * 1000 > container_length_mm:
    st.error("Foundation long dimension must fit inside the container length.")
    st.stop()
if foundation_width_mm + FOOT_Y * 1000 > container_width_mm:
    st.error("Foundation short dimension must fit inside the container width.")
    st.stop()
container_scale_x = container_length_mm / 600.0
container_scale_y = container_width_mm / 400.0
soil_depth_scale = soil_fill_depth_mm / (SOIL_SURFACE_Z * 1000)
foundation_scale_x = foundation_length_mm / (ARM_LONG * 1000)
foundation_scale_y = foundation_width_mm / (ARM_SHORT * 1000)
scene_points = points.copy()
scene_points[:, 0] *= container_scale_x
scene_points[:, 1] *= container_scale_y
scene_points[:, 2] *= soil_depth_scale
foundation_points = scene_points.copy()
foundation_origin_x = FOOT_X * container_scale_x
foundation_origin_y = FOOT_Y * container_scale_y
foundation_points[:, 0] = foundation_origin_x + (
    scene_points[:, 0] - foundation_origin_x
) * foundation_scale_x
foundation_points[:, 1] = foundation_origin_y + (
    scene_points[:, 1] - foundation_origin_y
) * foundation_scale_y
foundation_base_z = (soil_fill_depth_mm - embedment_depth_mm) / 1000
foundation_points[:, 2] = foundation_base_z + (
    points[:, 2] - SOIL_SURFACE_Z
) * (foundation_thickness_mm / 1000) / (
    FOOTING_TOP_Z - SOIL_SURFACE_Z
)
project_name = parameter_values["Project"]
soil_source = parameter_values["Soil source / location"]
soil_classification = parameter_values["Soil classification"]
soil_condition = parameter_values["Soil condition"]
property_reference = parameter_values["Soil property reference"]
project_notes = parameter_values["Assumptions / notes"]

settlement_m = settlement_at_points(
    points,
    loads,
    float(soil_modulus_mpa),
    float(poisson_ratio),
)
settlement_mm = settlement_m * 1000
center_settlement_mm = settlement_at_load_centers(
    loads,
    float(soil_modulus_mpa),
    float(poisson_ratio),
)
mean_settlement_mm = float(np.mean(center_settlement_mm))
contact_pressures_kpa = np.array(
    [
        float(load["force_kn"])
        / (float(load["width_x"]) * float(load["width_y"]))
        for load in loads
    ],
    dtype=np.float64,
)
maximum_contact_pressure_kpa = float(np.max(contact_pressures_kpa))
maximum_strain_proxy_microstrain = float(
    np.max(np.abs(center_settlement_mm))
    / float(soil_fill_depth_mm)
    * 1_000_000
)
shear_modulus_mpa = float(soil_modulus_mpa) / (
    2 * (1 + float(poisson_ratio))
)
bulk_modulus_mpa = float(soil_modulus_mpa) / (
    3 * (1 - 2 * float(poisson_ratio))
)

bulk_density_kg_m3: float | None = None
dry_density_kg_m3: float | None = None
if record_sample and sample_mass_kg > 0 and sample_volume_l > 0:
    measured_density_kg_m3 = sample_mass_kg / sample_volume_l * 1000
    moisture_ratio = (
        water_content_pct / 100
        if water_content_pct is not None
        else None
    )
    if mass_basis == "Wet mass":
        bulk_density_kg_m3 = measured_density_kg_m3
        if moisture_ratio is not None:
            dry_density_kg_m3 = bulk_density_kg_m3 / (1 + moisture_ratio)
    else:
        dry_density_kg_m3 = measured_density_kg_m3
        if moisture_ratio is not None:
            bulk_density_kg_m3 = dry_density_kg_m3 * (1 + moisture_ratio)

st.warning(
    "Settlement is a fast elastic half-space estimate under uniformly "
    "distributed rectangular loads. It is not a rerun of the saved 3D FEM "
    "simulation and is not a substitute for engineering design."
)

st.subheader("Project parameters and calculated properties")
st.caption(
    f"Project: {project_name or 'Not entered'} | "
    f"Soil source: {soil_source or 'Not entered'} | "
    f"Classification: {soil_classification or 'Not entered'} | "
    f"Condition: {soil_condition}"
)
st.caption(
    f"Sample mass: {sample_mass_kg:g} kg | "
    f"Sample volume: {sample_volume_l:g} L | "
    f"Mass basis: {mass_basis if record_sample else 'Not entered'} | "
    f"Water content: "
    f"{f'{water_content_pct:g}%' if water_content_pct is not None else 'Not entered'}"
)
property_columns = st.columns(4)
property_columns[0].metric(
    "Young's modulus",
    f"{float(soil_modulus_mpa):.3f} MPa",
)
property_columns[1].metric(
    "Poisson's ratio",
    f"{float(poisson_ratio):.3f}",
)
property_columns[2].metric(
    "Shear modulus (calculated)",
    f"{shear_modulus_mpa:.3f} MPa",
)
property_columns[3].metric(
    "Bulk modulus (calculated)",
    f"{bulk_modulus_mpa:.3f} MPa",
)
density_columns = st.columns(2)
density_columns[0].metric(
    "Bulk density (calculated)",
    (
        f"{bulk_density_kg_m3:.1f} kg/m³"
        if bulk_density_kg_m3 is not None
        else "Not available"
    ),
)
density_columns[1].metric(
    "Dry density (calculated)",
    (
        f"{dry_density_kg_m3:.1f} kg/m³"
        if dry_density_kg_m3 is not None
        else "Not available"
    ),
)
if record_sample and (sample_mass_kg <= 0 or sample_volume_l <= 0):
    st.info(
        "Enter positive sample mass and volume to calculate density."
    )
st.caption(
    "Elastic moduli use isotropic linear-elastic formulas. Density is "
    "calculated from sample mass and volume; dry/bulk conversion requires "
    "measured water content. Source, reference, and notes are recorded only. "
    "Edited dimensions scale the 3D display, not the saved FEM solution."
)

summary_columns = st.columns(3)
summary_columns[0].metric(
    "Maximum estimated settlement",
    f"{settlement_mm.max():.3f} mm",
)
summary_columns[1].metric(
    "Settlement at load points",
    f"{center_settlement_mm.max():.3f} mm",
)
summary_columns[2].metric(
    "Difference between load points",
    f"{np.ptp(center_settlement_mm):.3f} mm",
)
response_columns = st.columns(2)
response_columns[0].metric(
    "Maximum contact stress",
    f"{maximum_contact_pressure_kpa:.2f} kPa",
)
response_columns[1].metric(
    "Maximum settlement/depth strain proxy",
    f"{maximum_strain_proxy_microstrain:.1f} με",
)
st.caption(
    "Contact stress is load divided by its loaded area. The strain proxy is "
    "load-center settlement divided by the entered soil fill depth; it is an "
    "average estimate, not a local strain extracted from an FEM solution."
)

st.subheader("Foundation and load areas")
st.caption(
    "The foundation is shown in an undeformed shape. Soil color shows "
    "estimated settlement; settlement values remain in millimetres. The "
    "deformation cage, when enabled, shows the saved FEM displacement and "
    "is visually scaled. Edited dimensions resize this view only; the "
    "settlement estimate and saved FEM field are not recalculated."
)

fig = go.Figure(
    data=[
        go.Mesh3d(
            x=scene_points[:, 0],
            y=scene_points[:, 1],
            z=scene_points[:, 2],
            i=soil_surface[:, 0],
            j=soil_surface[:, 1],
            k=soil_surface[:, 2],
            intensity=settlement_mm,
            intensitymode="vertex",
            colorscale="Turbo",
            colorbar=dict(title="Settlement (mm)"),
            customdata=settlement_mm,
            hovertemplate=(
                "Soil<br>"
                "X: %{x:.4f} m<br>"
                "Y: %{y:.4f} m<br>"
                "Settlement: %{customdata:.4f} mm"
                "<extra></extra>"
            ),
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
            flatshading=True,
            name="Soil (settlement)",
        ),
        go.Mesh3d(
            x=foundation_points[:, 0],
            y=foundation_points[:, 1],
            z=foundation_points[:, 2],
            i=foundation_surface[:, 0],
            j=foundation_surface[:, 1],
            k=foundation_surface[:, 2],
            color="#e2e2e2",
            customdata=settlement_mm,
            hovertemplate=(
                "Concrete foundation<br>"
                "X: %{x:.4f} m<br>"
                "Y: %{y:.4f} m<br>"
                "Settlement: %{customdata:.4f} mm"
                "<extra></extra>"
            ),
            lighting=dict(ambient=0.85, diffuse=0.9, specular=0.25),
            flatshading=True,
            name="Concrete foundation",
        ),
    ]
)
if show_deformation_cage and fem_displacement is not None:
    display_displacement = fem_displacement * np.array(
        [container_scale_x, container_scale_y, soil_depth_scale],
        dtype=np.float64,
    )
    deformed_points = scene_points + display_displacement * float(
        deformation_scale
    )
    footing_vertices = np.unique(foundation_surface)
    deformed_points[footing_vertices, 0] = foundation_origin_x + (
        deformed_points[footing_vertices, 0] - foundation_origin_x
    ) * foundation_scale_x
    deformed_points[footing_vertices, 1] = foundation_origin_y + (
        deformed_points[footing_vertices, 1] - foundation_origin_y
    ) * foundation_scale_y
    deformed_points[footing_vertices, 2] = foundation_points[
        footing_vertices, 2
    ] + display_displacement[footing_vertices, 2] * float(
        deformation_scale
    )
    fig.add_trace(
        go.Mesh3d(
            x=deformed_points[:, 0],
            y=deformed_points[:, 1],
            z=deformed_points[:, 2],
            i=surface[:, 0],
            j=surface[:, 1],
            k=surface[:, 2],
            color="#ff8c42",
            opacity=0.2,
            flatshading=True,
            name="Saved FEM deformation",
            hovertemplate=(
                "Deformed FEM surface<br>"
                "Display scale: "
                f"{float(deformation_scale):g}×"
                "<extra></extra>"
            ),
        )
    )
    edges = np.unique(
        np.sort(
            np.concatenate(
                (
                    surface[:, [0, 1]],
                    surface[:, [1, 2]],
                    surface[:, [2, 0]],
                ),
                axis=0,
            ),
            axis=1,
        ),
        axis=0,
    )
    edge_coordinates = deformed_points[edges]
    edge_x = np.column_stack(
        (edge_coordinates[:, 0, 0], edge_coordinates[:, 1, 0], np.full(len(edges), np.nan))
    ).ravel()
    edge_y = np.column_stack(
        (edge_coordinates[:, 0, 1], edge_coordinates[:, 1, 1], np.full(len(edges), np.nan))
    ).ravel()
    edge_z = np.column_stack(
        (edge_coordinates[:, 0, 2], edge_coordinates[:, 1, 2], np.full(len(edges), np.nan))
    ).ravel()
    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="#8b1a1a", width=2),
            name="Deformed FEM mesh cage",
            hoverinfo="skip",
        )
    )
for index, (load, load_settlement_mm) in enumerate(
    zip(loads, center_settlement_mm)
):
    load_x, load_y = load_outline(load)
    load_x = [
        foundation_origin_x
        + (x * container_scale_x - foundation_origin_x) * foundation_scale_x
        for x in load_x
    ]
    load_y = [
        foundation_origin_y
        + (y * container_scale_y - foundation_origin_y) * foundation_scale_y
        for y in load_y
    ]
    load_z = foundation_base_z + foundation_thickness_mm / 1000 + 0.002
    load_center_x = foundation_origin_x + (
        float(load["x"]) * container_scale_x - foundation_origin_x
    ) * foundation_scale_x
    load_center_y = foundation_origin_y + (
        float(load["y"]) * container_scale_y - foundation_origin_y
    ) * foundation_scale_y
    color = load_colors[index % len(load_colors)]
    fig.add_trace(
        go.Scatter3d(
            x=load_x,
            y=load_y,
            z=[load_z] * len(load_x),
            mode="lines",
            line=dict(color=color, width=7),
            name=str(load["name"]),
            hovertemplate=(
                f"{load['name']}<br>"
                f"Load: {float(load['force_kn']):.3f} kN"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[load_center_x],
            y=[load_center_y],
            z=[load_z],
            mode="markers+text",
            marker=dict(size=6, color=color, line=dict(color="#222222", width=1)),
            text=[str(load["name"])],
            textposition="top center",
            name=f"{load['name']} center",
            showlegend=False,
            hovertemplate=(
                f"{load['name']}<br>"
                f"Load: {float(load['force_kn']):.3f} kN<br>"
                f"Estimated settlement: {load_settlement_mm:.4f} mm"
                "<extra></extra>"
            ),
        )
    )

fig.update_layout(
    height=700,
    scene=dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        aspectmode="data",
    ),
    margin=dict(l=0, r=0, t=20, b=0),
)
st.plotly_chart(fig, width="stretch")
if fem_displacement is not None:
    saved_max_displacement_mm = float(
        np.max(np.linalg.norm(fem_displacement, axis=1)) * 1000
    )
    st.caption(
        f"Saved FEM maximum displacement: {saved_max_displacement_mm:.6f} mm. "
        f"Displayed cage scale: {float(deformation_scale):g}×. "
        "This saved result is from its original FEM run; changing current "
        "loads or material inputs does not rerun that simulation."
    )

st.subheader("Settlement comparison at each load")
results = []
for load, load_settlement_mm in zip(loads, center_settlement_mm):
    area = float(load["width_x"]) * float(load["width_y"])
    results.append(
        {
            "Load name": str(load["name"]),
            "Entered as": str(load["input_type"]),
            "Entered value": float(load["entered_load"]),
            "Equivalent mass (kg)": (
                float(load["force_kn"]) * 1000 / STANDARD_GRAVITY_M_S2
            ),
            "Load (kN)": float(load["force_kn"]),
            "Center X (m)": float(load["x"]),
            "Center Y (m)": float(load["y"]),
            "Area (m²)": area,
            "Pressure (kPa)": float(load["force_kn"]) / area,
            "Settlement/depth strain proxy (με)": float(
                load_settlement_mm / float(soil_fill_depth_mm) * 1_000_000
            ),
            "Settlement (mm)": float(load_settlement_mm),
            "Difference from average (mm)": float(
                load_settlement_mm - mean_settlement_mm
            ),
        }
    )
st.dataframe(results, hide_index=True, width="stretch")
st.caption(
    "Soil stiffness and loads are illustrative inputs. Use calibrated "
    "properties and a verified FEM analysis for engineering decisions."
)
