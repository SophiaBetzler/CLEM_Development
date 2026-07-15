import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clem_correlation import CLEMCorrelator


def test_load_transform_from_csv_and_combine_with_fit(tmp_path):
    csv_path = tmp_path / "transform.csv"
    csv_path.write_text("1,0,10\n0,1,20\n0,0,1\n", encoding="utf-8")

    correlator = CLEMCorrelator(mrc_reader=None)
    imported = correlator.load_transform_from_csv(csv_path)

    assert imported is not None
    assert np.allclose(np.asarray(imported.params), np.array([[1.0, 0.0, 10.0],
                                                            [0.0, 1.0, 20.0],
                                                            [0.0, 0.0, 1.0]]))

    point_pairs = [
        {"tiff": (0.0, 0.0), "mrc": (10.0, 20.0)},
        {"tiff": (10.0, 0.0), "mrc": (20.0, 20.0)},
    ]

    tform, fit_info, n_pairs = correlator.fit_fixed_scale_reflection(
        point_pairs,
        tiff_shape=(100, 100),
        fixed_scale=1.0,
        initial_transform=imported,
    )

    assert n_pairs == 2
    assert tform is not None
    assert fit_info["expected_scale"] == 1.0
    assert np.asarray(tform.params).shape == (3, 3)
