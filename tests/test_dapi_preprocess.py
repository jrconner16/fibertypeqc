import numpy as np

from src.dapi_preprocess import tile_background_subtract, tile_percentile_normalize


def test_tile_background_subtract_removes_local_baseline():
    image = np.full((8, 8), 100, dtype=np.float32)
    image[:, 4:] += 100
    image[2, 2] += 50
    image[5, 6] += 50
    out = tile_background_subtract(image, tile_size=4, background_quantile=0.0)
    assert out[0, 0] == 0
    assert out[0, 4] == 0
    assert out[2, 2] == 50
    assert out[5, 6] == 50


def test_tile_percentile_normalize_scales_local_ranges():
    image = np.zeros((4, 8), dtype=np.float32)
    image[:, :4] = np.arange(4, dtype=np.float32)[:, None]
    image[:, 4:] = 100 + 2 * np.arange(4, dtype=np.float32)[:, None]
    out = tile_percentile_normalize(image, tile_size=4, low_percentile=0, high_percentile=100)
    assert np.isclose(out[0, 0], 0.0)
    assert np.isclose(out[-1, 0], 1.0)
    assert np.isclose(out[0, 4], 0.0)
    assert np.isclose(out[-1, 4], 1.0)
