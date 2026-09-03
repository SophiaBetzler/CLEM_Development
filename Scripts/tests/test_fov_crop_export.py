"""MRCReader.write_fov_crops: the picker's 'Export crops' path.

Checks that both writers are driven correctly (this function used to raise
TypeError before writing anything), that files land in the picks folder, and
that the fast crop-warp path is used and agrees with the full-plane path.

Run with:  python tests/test_fov_crop_export.py
"""
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

# mrcfile/tifffile may not be installed; record calls instead of writing.
WRITES = {"tif": [], "mrc": []}


class _FakeMrc:
    def __init__(self, path):
        self.path = path; self.voxel_size = None
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def set_data(self, d):
        WRITES["mrc"].append((self.path, tuple(d.shape), str(d.dtype)))
    def update_header_from_data(self):
        pass


mrcfile_stub = types.ModuleType("mrcfile")
mrcfile_stub.new = lambda path, overwrite=False, **kw: _FakeMrc(path)
sys.modules.setdefault("mrcfile", mrcfile_stub)

tifffile_stub = types.ModuleType("tifffile")


def _imwrite(path, data, **kw):
    WRITES["tif"].append((path, tuple(data.shape), kw.get("compression")))


tifffile_stub.imwrite = _imwrite
sys.modules.setdefault("tifffile", tifffile_stub)

from clem_dataclasses import MRCSummary, Pick          # noqa: E402
from clem_mrc_mdoc_reader import MRCReader             # noqa: E402

FAILURES = []


def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
        return
    FAILURES.append(label)
    print(f"  FAIL  {label}\n        got  {got!r}\n        want {want!r}")


def check_navigator(summary):
    """Drive CLEMPicker.add_picks_to_navigator against a stubbed SerialEM and
    return the notes it set, in order."""
    notes = []
    idx = {"n": 0}

    class _Sem(types.ModuleType):
        def __getattr__(self, name):
            def call(*a, **k):
                if name == "AddImagePosAsNavPoint":
                    idx["n"] += 1
                    return idx["n"]
                if name == "ChangeItemNote":
                    notes.append(a[1])
                return 0
            return call

    sys.modules["serialem"] = _Sem("serialem")
    for m in ("tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox"):
        sys.modules.setdefault(m, types.ModuleType(m))

    from clem_tem_communication import TEMComm
    from clem_target_picking import CLEMPicker

    tem = TEMComm(mrc_reader=MRCReader(coord_key="AlignedPieceCoordsVS", section=0),
                  path="/tmp", offline=False)
    picker = CLEMPicker(summary, tem, site_data=None)
    picker.add_picks_to_navigator(summary, buffer="S")
    return notes


def make_summary(H=200, W=240, n_picks=3, ps=0.005):
    rng = np.random.default_rng(1)
    s = MRCSummary(mrc_path="/tmp/site_01/site_01_montage.mrc",
                   montage_id="site_01_montage",
                   image=rng.random((H, W)).astype(np.float32),
                   image_height=H, image_width=W,
                   pixel_spacing_um=ps, flip_x=False, flip_y=True)
    for i in range(n_picks):
        # CLEMPicker.make_pick_dataclass numbers picks "1", "2", ... -- keep the
        # fixture faithful so nav notes and crop filenames line up as in reality.
        s.picks.append(Pick(pick_id=str(i + 1),
                            image_coord_x=40.0 + 50 * i,
                            image_coord_y=60.0 + 30 * i))
    return s


def main():
    reader = MRCReader(coord_key="AlignedPieceCoordsVS", section=0)
    picks_dir = os.path.join(tempfile.mkdtemp(), "picks")
    summary = make_summary()
    H, W = summary.image.shape
    n_ch, n_z, fov = 2, 3, 0.25            # 0.25 um / 0.005 = 50 px crops

    calls = {"plane": 0, "crop": 0}

    def warp_slice(c, z):
        calls["plane"] += 1
        rng = np.random.default_rng(100 * c + z)
        return rng.random((H, W)).astype(np.float32)

    def warp_crop(c, z, x0, y0, cw):
        # Crop of exactly the plane warp_slice would have produced, so the two
        # paths are directly comparable.
        calls["crop"] += 1
        rng = np.random.default_rng(100 * c + z)
        full = rng.random((H, W)).astype(np.float32)
        out = np.zeros((cw, cw), np.float32)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W, x0 + cw), min(H, y0 + cw)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = full[sy0:sy1, sx0:sx1]
        return out

    print("\n--- it runs at all --------------------------------------------")
    WRITES["tif"].clear(); WRITES["mrc"].clear()
    res = reader.write_fov_crops(mrc_dataclass=summary, warp_slice=warp_slice,
                                 warp_crop=warp_crop, n_channels=n_ch, n_z=n_z,
                                 fov_um=fov, picks_dir=picks_dir)
    check("returns both lists", sorted(res), ["mrc", "tif"])
    check("one tif per pick", len(res["tif"]), 3)
    check("one mrc per pick", len(res["mrc"]), 3)

    print("\n--- files land in the picks folder ----------------------------")
    check("picks dir created", os.path.isdir(picks_dir), True)
    tif_paths = [p for p, _s, _c in WRITES["tif"]]
    mrc_paths = [p for p, _s, _d in WRITES["mrc"]]
    check("tifs in picks dir",
          all(os.path.dirname(p) == picks_dir for p in tif_paths), True)
    check("mrcs in picks dir",
          all(os.path.dirname(p) == picks_dir for p in mrc_paths), True)
    check("tif names", sorted(os.path.basename(p) for p in tif_paths),
          ["target_overlays_1.tif", "target_overlays_2.tif",
           "target_overlays_3.tif"])
    check("mrc names", sorted(os.path.basename(p) for p in mrc_paths),
          ["crop_fov_1.mrc", "crop_fov_2.mrc", "crop_fov_3.mrc"])

    print("\n--- crop geometry ---------------------------------------------")
    cw = max(2, int(round(fov / summary.pixel_spacing_um)))
    check("crop width from fov / pixel size", cw, 50)
    check("tif stacks are (z, 1+channels, cw, cw)",
          {s for _p, s, _c in WRITES["tif"]}, {(n_z, 1 + n_ch, cw, cw)})
    check("mrc crops are cw x cw", {s for _p, s, _d in WRITES["mrc"]}, {(cw, cw)})

    print("\n--- the fast path is the one used -----------------------------")
    check("crop warps used", calls["crop"], 3 * n_ch * n_z)
    check("no full planes warped", calls["plane"], 0)

    print("\n--- fast path agrees with the full-plane path -----------------")
    stacks = {}
    for mode in ("crop", "plane"):
        WRITES["tif"].clear()
        captured = []
        tifffile_stub.imwrite = lambda p, d, **kw: captured.append(d.copy())
        reader.write_fov_crops(mrc_dataclass=summary, warp_slice=warp_slice,
                               warp_crop=(warp_crop if mode == "crop" else None),
                               n_channels=n_ch, n_z=n_z, fov_um=fov,
                               picks_dir=picks_dir)
        stacks[mode] = captured
        tifffile_stub.imwrite = _imwrite
    check("same number of stacks", len(stacks["crop"]), len(stacks["plane"]))
    worst = max(float(np.abs(a - b).max())
                for a, b in zip(stacks["crop"], stacks["plane"]))
    check("crop path == plane path", worst, 0.0)

    print("\n--- falls back when no crop callback --------------------------")
    calls["plane"] = calls["crop"] = 0
    reader.write_fov_crops(mrc_dataclass=summary, warp_slice=warp_slice,
                           warp_crop=None, n_channels=n_ch, n_z=n_z,
                           fov_um=fov, picks_dir=picks_dir)
    check("full planes warped instead", calls["plane"], n_ch * n_z)

    print("\n--- pick.view_crop_path recorded ------------------------------")
    check("every pick points at its mrc crop",
          all(p.view_crop_path and p.view_crop_path.endswith(f"{p.pick_id}.mrc")
              for p in summary.picks), True)

    print("\n--- coordinates returned to SerialEM --------------------------")
    notes = check_navigator(summary)
    check("one nav point per pick", len(notes), 3)
    check("notes carry site label and pick number", notes,
          ["site_01_pick_1", "site_01_pick_2", "site_01_pick_3"])
    check("nav note number matches the crop filename",
          [n.rsplit("_", 1)[-1] for n in notes],
          [os.path.basename(p).rsplit("_", 1)[-1].removesuffix(".mrc")
           for p in sorted(mrc_paths)])

    print()
    print("=" * 62)
    print("FOV CROP EXPORT:", "all checks passed" if not FAILURES
          else f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
    print("=" * 62)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
