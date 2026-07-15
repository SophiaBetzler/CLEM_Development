import numpy as np

from clem_mrc_mdoc_reader import MRCReader


def test_crop_centered_accepts_scalar_like_inputs():
    full = np.arange(100, dtype=np.float32).reshape(10, 10)
    crop = MRCReader._crop_centered(full, np.array([4.2], dtype=object), np.array([5.8], dtype=object), 3)
    assert crop.shape == (3, 3)


def test_fov_in_px_uses_numeric_spacing():
    reader = MRCReader(coord_key="AlignedPieceCoords", path=".", refine_alignment=False)
    reader.pixel_spacing_um = 0.5
    assert reader._fov_in_px("8") == 16
