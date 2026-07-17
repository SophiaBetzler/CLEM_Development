import os
import sys
import numpy as np
import mrcfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clem_mrc_mdoc_reader import MRCReader


def test_assemble_montage_accepts_explicit_pieces(tmp_path):
    mrc_path = tmp_path / "test_tiles.mrc"
    data = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    with mrcfile.new(str(mrc_path), data=data) as mrc:
        mrc.set_data(data)

    reader = MRCReader(coord_key="PieceCoordinates", path=str(tmp_path), refine_alignment=False, section=0)
    pieces = [{"ZValue": 0, "PieceCoordinates": [0, 0, 0]}]

    img, min_x, min_y = reader._assemble_montage(
        str(mrc_path),
        img_h=4,
        img_w=4,
        feather_px=1,
        key="PieceCoordinates",
        pieces=pieces,
    )

    assert img.shape == (4, 4)
    assert min_x == 0.0
    assert min_y == 0.0
