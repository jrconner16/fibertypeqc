import numpy as np

from src.nuclear_association import associate_nuclei, summarize_fiber_nuclei


def test_associate_nuclei_marks_central_boundary_and_unassigned():
    fibers = np.zeros((20, 30), dtype=np.int32)
    fibers[2:18, 2:13] = 1
    fibers[2:18, 17:28] = 2
    nuclei = np.zeros_like(fibers)
    nuclei[7:9, 6:8] = 1  # central in fiber 1
    nuclei[2:4, 7:9] = 2  # boundary-associated with fiber 1
    nuclei[8:10, 14:16] = 3  # interstitial

    table, links = associate_nuclei(
        nuclei,
        fibers,
        boundary_distance_px=2.0,
        central_normalized_radius=0.2,
    )

    categories = dict(zip(table.nucleus_id, table.association_category, strict=True))
    assert categories[1] == "central_interior"
    assert categories[2] == "boundary_associated"
    assert categories[3] == "unassigned_or_interstitial"
    assert set(links.fiber_id) == {1}


def test_summarize_fiber_nuclei_adds_zero_counts_for_empty_fibers():
    fibers = np.zeros((8, 12), dtype=np.int32)
    fibers[1:7, 1:5] = 1
    fibers[1:7, 7:11] = 2
    nuclei = np.zeros_like(fibers)
    nuclei[3:5, 2:4] = 1
    table, _ = associate_nuclei(
        nuclei,
        fibers,
        boundary_distance_px=0.0,
        central_normalized_radius=0.0,
    )
    summary = summarize_fiber_nuclei(fibers, table).set_index("fiber_id")
    assert int(summary.loc[1, "associated_nuclei_count"]) == 1
    assert int(summary.loc[2, "associated_nuclei_count"]) == 0
    assert bool(summary.loc[1, "centrally_nucleated"])
