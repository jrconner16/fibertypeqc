"""Open FiberTypeQC's guided Phase 4.5 image and object review workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.generate_review_qc import load_manual_selections
from src.review.dashboard import build_dashboard_model, load_dashboard_tables
from src.review.image_review import ImageReviewController
from src.review.nuclear_review import NuclearReviewController
from src.review.project import load_project
from src.review.schemas import Domain, Scope
from src.review.section_selection import SelectionStrategy
from src.review.session import ReviewSession
from src.review.storage import load_session


def downsample_review_data(
    raw: np.ndarray,
    labels: np.ndarray | None,
    factor: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Downsample raw and label display layers together without changing IDs."""
    if factor < 1:
        raise ValueError("display_downsample must be at least 1")
    if raw.ndim != 3:
        raise ValueError(f"Expected CHW raw image, got shape {raw.shape}")
    if labels is not None and labels.shape != raw.shape[1:]:
        raise ValueError(
            f"Fiber-label shape {labels.shape} does not match raw image {raw.shape[1:]}"
        )
    if factor == 1:
        return raw, labels
    return raw[:, ::factor, ::factor], None if labels is None else labels[::factor, ::factor]


def selected_fiber_outline(labels: np.ndarray, fiber_id: int) -> np.ndarray:
    """Return a thick, non-obscuring outline for the selected fiber."""
    from skimage.segmentation import find_boundaries

    if labels.ndim != 2:
        raise ValueError(f"Expected 2D fiber labels, got shape {labels.shape}")
    return find_boundaries(labels == fiber_id, mode="thick").astype(np.uint8)


def selected_fiber_outline_rgba(labels: np.ndarray, fiber_id: int) -> np.ndarray:
    """Render the selected-fiber outline as an opaque cyan RGBA overlay."""
    outline = selected_fiber_outline(labels, fiber_id)
    rgba = np.zeros((*outline.shape, 4), dtype=np.float32)
    rgba[..., 1] = outline
    rgba[..., 2] = outline
    rgba[..., 3] = outline
    return rgba


def downsample_label_data(labels: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2D labels layer for display without renumbering IDs."""
    if factor < 1:
        raise ValueError("display_downsample must be at least 1")
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels, got shape {labels.shape}")
    return labels[::factor, ::factor]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--qc-dir",
        type=Path,
        help="Phase 2A QC directory; defaults to <project>/qc.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=[strategy.value for strategy in SelectionStrategy],
        default=SelectionStrategy.ALL_PASSING.value,
    )
    parser.add_argument(
        "--manual-selection",
        type=Path,
        help="Optional manual mouse/domain/image selection YAML.",
    )
    parser.add_argument("--reviewer", default="", help="Optional reviewer name stored in events.")
    parser.add_argument(
        "--display-downsample",
        type=int,
        default=1,
        help="Display-only raw/label downsample factor (1, 2, or 4 are typical).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.display_downsample < 1:
        raise ValueError("--display-downsample must be at least 1")
    project = load_project(args.project)
    tables = load_dashboard_tables(project, args.qc_dir)
    manual = load_manual_selections(args.manual_selection)
    session = (
        load_session(project.review_state_path, expected_project_id=project.project_id)
        if project.review_state_path.is_file()
        else ReviewSession(
            project_id=project.project_id,
            model_version=project.model_version,
            qc_version=project.qc_version,
            reviewer=args.reviewer,
        )
    )
    if args.reviewer:
        session.reviewer = args.reviewer
    model = build_dashboard_model(
        project,
        tables,
        strategy=args.selection_strategy,
        manual_selections=manual,
        session=session,
    )

    import napari

    from src.io_utils import load_multichannel_image
    from src.review.channel_display import channel_displays, stain_composite
    from src.review.channel_map_widget import ChannelMapWidget
    from src.review.dashboard_widget import CohortDashboardWidget
    from src.review.fiber_type_review import FiberTypeReviewController
    from src.review.guided_review_widget import GuidedReviewWidget
    from src.review.image_review_widget import ImageReviewWidget
    from src.review.nuclear_review_widget import NuclearReviewWidget
    from src.review.region_review import RegionReviewController
    from src.review.region_review_widget import RegionReviewWidget
    from src.review.schemas import RegionKind
    from src.typing_display import normalize_for_display

    viewer = napari.Viewer(title=f"FiberTypeQC project: {project.project_name}")
    widget = CohortDashboardWidget(
        project,
        tables,
        strategy=model.strategy,
        manual_selections=manual,
        session=session,
    )
    dashboard_dock = viewer.window.add_dock_widget(widget, area="right", name="Cohort Dashboard")
    channel_map_widget = ChannelMapWidget()
    channel_map_dock = viewer.window.add_dock_widget(
        channel_map_widget,
        area="right",
        name="Channel Map",
    )
    controller = ImageReviewController(project, tables.image_qc, session)
    nuclear_controller = NuclearReviewController(project, session)
    region_controller = RegionReviewController(project, session)
    loaded_image_id: str | None = None
    nuclear_widget: NuclearReviewWidget | None = None

    def _region_shape_data(image_id: str, kind: RegionKind) -> list[np.ndarray]:
        data: list[np.ndarray] = []
        for region in region_controller.regions_for_image(image_id):
            if region.kind is not kind:
                continue
            if region.geometry.get("type") != "Polygon":
                continue
            rings = region.geometry.get("coordinates", [])
            if not rings:
                continue
            # GeoJSON is x/y; Napari image coordinates are row/y then column/x.
            data.append(
                np.asarray(
                    [
                        [y / args.display_downsample, x / args.display_downsample]
                        for x, y in rings[0]
                    ],
                    dtype=float,
                )
            )
        return data

    def _refresh_region_shapes() -> None:
        try:
            shapes = viewer.layers["review_region_shapes"]
        except KeyError:
            return
        shapes.data = _region_shape_data(controller.current_image_id, RegionKind.REVIEW)
        try:
            rois = viewer.layers["review_analysis_rois"]
        except KeyError:
            rois = None
        if rois is not None:
            rois.data = _region_shape_data(controller.current_image_id, RegionKind.ANALYSIS_ROI)
        try:
            coverage = viewer.layers["review_region_coverage"]
        except KeyError:
            return
            coverage.data = region_controller.coverage_heatmap(
                controller.current_image_id,
                coverage.data.shape,
                coordinate_scale=args.display_downsample,
                kind=RegionKind.REVIEW,
            )

    def _selected_region_geometry() -> dict | None:
        try:
            shapes = viewer.layers["review_region_shapes"]
        except KeyError:
            return None
        selected = sorted(shapes.selected_data)
        if not selected:
            return None
        points = np.asarray(shapes.data[selected[-1]])
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            return None
        coordinates = [
            [float(x * args.display_downsample), float(y * args.display_downsample)]
            for y, x in points
        ]
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
        return {"type": "Polygon", "coordinates": [coordinates]}

    def show_image(image_id: str) -> None:
        nonlocal loaded_image_id
        for layer in list(viewer.layers):
            if layer.name.startswith("review_"):
                viewer.layers.remove(layer)
        image = project.image(image_id)
        raw = load_multichannel_image(image.raw_image_path)
        labels_path = image.outputs.get("fiber_labels")
        nuclei_path = image.outputs.get("nuclei_labels")
        labels = None
        nuclei = None
        if labels_path is not None:
            import tifffile

            labels = np.asarray(tifffile.imread(labels_path), dtype=np.int32)
        if nuclei_path is not None:
            nuclei = nuclear_controller.load_nuclei_labels(image_id)
            if nuclei.shape != raw.shape[1:]:
                raise ValueError(
                    f"Nuclei-label shape {nuclei.shape} does not match raw image {raw.shape[1:]}"
                )
        raw, labels = downsample_review_data(raw, labels, args.display_downsample)
        displays = channel_displays(project.panel_manifest, raw.shape[0])
        channel_map_widget.set_displays(displays)
        viewer.add_image(
            stain_composite(raw, displays),
            name="review_stain_composite",
            rgb=True,
        )
        for display in displays:
            viewer.add_image(
                normalize_for_display(raw[display.channel]),
                name=f"review_raw_{display.role}_ch{display.channel}",
                colormap=display.colormap,
                blending="additive",
                visible=False,
            )
        if labels is not None:
            viewer.add_labels(labels, name="review_fiber_labels", opacity=0.25)
            viewer.add_image(
                np.zeros((*labels.shape, 4), dtype=np.float32),
                name="review_fiber_selected_outline",
                opacity=1.0,
                blending="additive",
                rgb=True,
            )
        if nuclei is not None:
            displayed_nuclei = downsample_label_data(nuclei, args.display_downsample)
            nuclei_layer = viewer.add_labels(
                displayed_nuclei,
                name="review_nuclei_labels",
                opacity=0.45,
            )
            try:
                nuclei_layer.editable = False
            except AttributeError:
                pass
            viewer.add_labels(
                np.zeros_like(displayed_nuclei, dtype=np.int32),
                name="review_new_nucleus_draft",
                opacity=0.7,
            )
        viewer.add_shapes(
            _region_shape_data(image_id, RegionKind.REVIEW),
            name="review_region_shapes",
            shape_type="polygon",
            edge_color="yellow",
            face_color=[1.0, 1.0, 0.0, 0.08],
            edge_width=2,
        )
        viewer.add_shapes(
            _region_shape_data(image_id, RegionKind.ANALYSIS_ROI),
            name="review_analysis_rois",
            shape_type="polygon",
            edge_color="cyan",
            face_color=[0.0, 1.0, 1.0, 0.06],
            edge_width=2,
        )
        viewer.add_image(
            region_controller.coverage_heatmap(
                image_id,
                raw.shape[1:],
                coordinate_scale=args.display_downsample,
                kind=RegionKind.REVIEW,
            ),
            name="review_region_coverage",
            colormap="yellow",
            opacity=0.18,
            visible=True,
        )
        loaded_image_id = image_id
        region_widget.refresh()
        if nuclear_widget is not None:
            nuclear_widget.refresh()

    def show_object(image_id: str, fiber_id: int) -> None:
        controller.set_image(image_id)
        if loaded_image_id != image_id:
            show_image(image_id)
        try:
            selected = viewer.layers["review_fiber_selected_outline"]
            labels = viewer.layers["review_fiber_labels"]
        except KeyError:
            selected = None
            labels = None
        if selected is not None and labels is not None:
            selected.data = selected_fiber_outline_rgba(labels.data, fiber_id)
            coordinates = (labels.data == fiber_id).nonzero()
            if len(coordinates[0]):
                viewer.camera.center = tuple(float(values.mean()) for values in coordinates)
        review_widget.refresh()

    def focus_current_object() -> None:
        item = guided_widget.controller.current_item
        if item is not None:
            show_object(item.image_id, item.fiber_id)

    region_widget = RegionReviewWidget(
        region_controller,
        selected_geometry=_selected_region_geometry,
        regions_changed=_refresh_region_shapes,
    )
    region_review_dock = viewer.window.add_dock_widget(
        region_widget,
        area="left",
        name="Region Review",
    )
    region_review_dock.hide()
    review_widget = ImageReviewWidget(controller, image_changed=show_image)
    image_review_dock = viewer.window.add_dock_widget(
        review_widget,
        area="left",
        name="Image Review (advanced)",
    )
    image_review_dock.hide()

    def _selected_nucleus_id() -> int:
        try:
            return int(viewer.layers["review_nuclei_labels"].selected_label)
        except KeyError:
            return 0

    def _draft_nucleus_pixels() -> np.ndarray:
        try:
            return np.asarray(viewer.layers["review_new_nucleus_draft"].data, dtype=bool)
        except KeyError as exc:
            raise ValueError("This image has no nuclei-label artifact") from exc

    def _reset_draft_nucleus() -> None:
        try:
            draft = viewer.layers["review_new_nucleus_draft"]
        except KeyError:
            return
        draft.data = np.zeros_like(draft.data, dtype=np.int32)

    def _refresh_nuclei_layers() -> None:
        image_id = controller.current_image_id
        try:
            layer = viewer.layers["review_nuclei_labels"]
        except KeyError:
            nuclear_widget.refresh()
            return
        layer.data = downsample_label_data(
            nuclear_controller.load_nuclei_labels(image_id), args.display_downsample
        )
        _reset_draft_nucleus()
        nuclear_widget.refresh()

    nuclear_widget = NuclearReviewWidget(
        nuclear_controller,
        selected_nucleus_id=_selected_nucleus_id,
        draft_pixels=_draft_nucleus_pixels,
        reset_draft=_reset_draft_nucleus,
        review_changed=_refresh_nuclei_layers,
        add_enabled=args.display_downsample == 1,
    )
    nuclear_review_dock = viewer.window.add_dock_widget(
        nuclear_widget,
        area="left",
        name="Nuclei Review",
    )
    nuclear_review_dock.hide()

    def open_dashboard() -> None:
        dashboard_dock.show()
        dashboard_dock.raise_()

    def open_image_review() -> None:
        image_review_dock.show()
        image_review_dock.raise_()

    def show_section() -> None:
        controller.session.active_scope = Scope.IMAGE
        show_image(controller.current_image_id)
        open_image_review()

    def open_region_review() -> None:
        show_image(controller.current_image_id)
        region_review_dock.show()
        region_review_dock.raise_()

    def open_nuclei_review() -> None:
        controller.set_domain(Domain.NUCLEI)
        show_image(controller.current_image_id)
        nuclear_review_dock.show()
        nuclear_review_dock.raise_()

    def show_domain(domain: Domain) -> None:
        controller.set_domain(domain)
        review_widget.refresh(notify=True)
        if domain is Domain.NUCLEI:
            open_nuclei_review()
        else:
            open_image_review()

    guided_widget = GuidedReviewWidget(
        project,
        FiberTypeReviewController(session),
        object_changed=show_object,
        show_dashboard=open_dashboard,
        show_section=show_section,
        show_region=open_region_review,
        show_domain=show_domain,
        focus_current_object=focus_current_object,
    )
    guided_dock = viewer.window.add_dock_widget(guided_widget, area="left", name="Guided Review")
    dashboard_dock.hide()

    qt_window = getattr(viewer.window, "_qt_window", None)
    if qt_window is not None:
        workspace_menu = qt_window.menuBar().addMenu("Workspace")
        workspace_menu.addAction("Show Guided Review", guided_dock.show)
        workspace_menu.addAction("Show Cohort QC", open_dashboard)
        workspace_menu.addAction("Show Image Controls", open_image_review)
        workspace_menu.addAction("Show Region Review", open_region_review)
        workspace_menu.addAction("Show Nuclei Review", open_nuclei_review)
        workspace_menu.addAction("Show Channel Map", channel_map_dock.show)

        def restore_workspace() -> None:
            guided_dock.show()
            guided_dock.raise_()
            channel_map_dock.show()
            channel_map_dock.raise_()

        workspace_menu.addAction("Restore review workspace", restore_workspace)
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
