"""Capture the GUI screenshots used in the paper's Figures 2 and 3.

Each window is opened by FiberTypeQC's own code on the public demonstrations, set to a fixed size,
and saved with Qt's window grab, so the screenshots are reproducible and crisp at the screen's
pixel ratio. Nothing is drawn over them. Outputs go to paper/figures/screens/.

Inputs (unpacked v0.5.2 demo downloads):
- the demo, with demo_project_completed finalized;
- the builder demo, after `fibertypeqc build builder_project` (its documented first step).

Run from the repository root (each window is captured in its own process):
    uv run python paper/figures/capture_screenshots.py \
        --demo path/to/fibertypeqc_demo --builder path/to/fibertypeqc_builder_demo
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCREENS = Path(__file__).resolve().parent / "screens"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _settle(app, milliseconds: int) -> None:
    from PyQt5.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec_()
    app.processEvents()


def capture_launcher(builder: Path) -> None:
    """The launcher and its Build results window on the builder demo, in the light theme."""
    os.environ["FIBERTYPEQC_THEME"] = "light"
    sys.path.insert(0, str(ROOT))
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    # The Mac's dark mode would otherwise draw the native widgets dark under a light page.
    app.setStyle("Fusion")
    app.setPalette(app.style().standardPalette())
    from src.launcher import LauncherWindow

    project = builder / "builder_project"
    window = LauncherWindow(project)
    # Shown the way a user's home folder appears, not this machine's checkout path.
    window.path_label.setText("~/fibertypeqc_builder_demo/builder_project")
    window.resize(760, 630)
    window.show()
    _settle(app, 800)
    window.grab().save(str(SCREENS / "launcher_builder.png"))

    window.show_build_results()
    dialog = window.results_dialog
    dialog.resize(820, 630)
    dialog.show()
    _settle(app, 800)
    dialog.grab().save(str(SCREENS / "build_results.png"))
    dialog.close()
    window.close()


def capture_review(demo: Path) -> None:
    """The review window on the completed demo, on the first flagged fiber."""
    sys.path.insert(0, str(ROOT))
    import napari
    from PyQt5.QtCore import QTimer

    from src.review.guided_review_widget import GuidedReviewWidget

    real_run = napari.run

    def run_and_capture() -> None:
        viewer = napari.current_viewer()
        window = viewer.window._qt_window

        def capture() -> None:
            qt_viewer = viewer.window._qt_viewer
            for name in ("dockLayerControls", "dockLayerList"):
                dock = getattr(qt_viewer, name, None)
                if dock is not None:
                    dock.hide()
            window.statusBar().hide()
            guided = window.findChildren(GuidedReviewWidget)[0]
            guided.start_flagged_review()
            window.resize(1300, 860)

            def grab_and_close() -> None:
                viewer.reset_view()
                window.grab().save(str(SCREENS / "review.png"))
                viewer.close()

            QTimer.singleShot(1500, grab_and_close)

        QTimer.singleShot(1500, capture)
        real_run()

    napari.run = run_and_capture
    from src import review_project_napari

    # The same arguments `fibertypeqc review` passes (src/cli.py, command_review).
    project = demo / "demo_project_completed" / "review" / "project.yaml"
    review_project_napari.main(["--project", str(project), "--display-downsample", "1"])


def capture_report(demo: Path) -> None:
    """The cohort report (light), cropped to the provenance header and the composition chart."""
    from PIL import Image

    report = demo / "demo_project_completed" / "results" / "cohort_report.html"
    with tempfile.TemporaryDirectory() as tmp:
        shot = Path(tmp) / "report.png"
        subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                "--window-size=1300,1100",  # wide, to sit under the review window
                "--blink-settings=preferredColorScheme=1",
                f"--screenshot={shot}",
                report.resolve().as_uri(),
            ],
            check=True,
            capture_output=True,
        )
        # The provenance header and the composition chart, with the page's own margins.
        Image.open(shot).crop((150, 20, 2420, 1170)).save(SCREENS / "cohort_report.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--only", choices=("launcher", "review", "report"))
    args = parser.parse_args(argv)
    SCREENS.mkdir(exist_ok=True)
    if args.only == "launcher":
        capture_launcher(args.builder.resolve())
    elif args.only == "review":
        capture_review(args.demo.resolve())
    elif args.only == "report":
        capture_report(args.demo.resolve())
    else:
        for part in ("launcher", "review", "report"):
            command = [sys.executable, __file__, "--demo", str(args.demo), "--builder"]
            subprocess.run([*command, str(args.builder), "--only", part], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
