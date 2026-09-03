"""CropPickerWindow._export in the standalone tool.

Calls the real method against a duck-typed picker (no Tk needed) and checks it
writes the same things, with the same names, as the pipeline picker's Export --
minus the SerialEM step, which the standalone tool has no navigator for.

Run with:  python tests/test_standalone_export.py
"""
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

WRITES = {"tif": [], "mrc": [], "png": []}


class _FakeMrc:
    def __init__(self, path):
        self.path = path; self.voxel_size = None
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def set_data(self, d):
        WRITES["mrc"].append((self.path, tuple(d.shape)))
    def update_header_from_data(self):
        pass


for name, mod in (("mrcfile", types.ModuleType("mrcfile")),
                  ("tifffile", types.ModuleType("tifffile")),
                  ("skimage", types.ModuleType("skimage")),
                  ("skimage.transform", types.ModuleType("skimage.transform"))):
    sys.modules.setdefault(name, mod)
sys.modules["mrcfile"].new = lambda path, overwrite=False, **kw: _FakeMrc(path)
sys.modules["tifffile"].imwrite = lambda p, d, **kw: WRITES["tif"].append(
    (p, tuple(d.shape), kw.get("compression"), kw.get("imagej")))
sys.modules["skimage.transform"].warp = lambda *a, **k: None
sys.modules["skimage.transform"].ProjectiveTransform = object
sys.modules["skimage.transform"].AffineTransform = object
sys.modules["skimage.transform"].SimilarityTransform = object
sys.modules["skimage.transform"].EuclideanTransform = object
sys.modules["skimage.transform"].estimate_transform = lambda *a, **k: None
for m in ("tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox",
          "matplotlib", "matplotlib.pyplot", "matplotlib.figure",
          "matplotlib.patheffects", "matplotlib.backends",
          "matplotlib.backends.backend_tkagg"):
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["matplotlib"].use = lambda *a, **k: None
sys.modules["matplotlib"].rcParams = {}

FAILURES = []


def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
        return
    FAILURES.append(label)
    print(f"  FAIL  {label}\n        got  {got!r}\n        want {want!r}")


def load_export():
    """Pull CropPickerWindow._export out of the source without importing the GUI."""
    src = open(os.path.join(SCRIPTS, "clem_correlation_tool_standalone.py")).read()
    start = src.index("    def _export(self):")
    end = src.index("# Main application", start)
    body = "\n".join(l[4:] if l.startswith("    ") else l
                     for l in src[start:end].splitlines())

    msgs = []
    ns = {
        "np": np, "os": os,
        "mrcfile": sys.modules["mrcfile"],
        "tifffile": sys.modules["tifffile"],
        "tiff_write": lambda p, d, **kw: sys.modules["tifffile"].imwrite(p, d, **kw),
        "messagebox": types.SimpleNamespace(
            showwarning=lambda *a, **k: msgs.append(("warn", a)),
            showerror=lambda *a, **k: msgs.append(("error", a)),
            showinfo=lambda *a, **k: msgs.append(("info", a))),
    }
    exec(compile(body, "<_export>", "exec"), ns)
    return ns["_export"], msgs


class Picker:
    """Minimal stand-in for CropPickerWindow."""
    def __init__(self, out_dir, n_picks=3, n_ch=2, n_z=3, H=200, W=240, ps=0.005):
        rng = np.random.default_rng(2)
        self._picks = [types.SimpleNamespace(pick_id=str(i + 1),
                                             image_coord_x=40.0 + 50 * i,
                                             image_coord_y=60.0 + 30 * i,
                                             view_crop_path=None)
                       for i in range(n_picks)]
        self._fov_var = types.SimpleNamespace(get=lambda: "0.25")
        self.map = types.SimpleNamespace(pixel_spacing_um=ps, map_id="site_01_map",
                                         path="/tmp/site_01/map.mrc")
        self._out_dir = out_dir
        self._map_true = rng.random((H, W)).astype(np.float32)
        self._chan_true = [rng.random((H, W)).astype(np.float32) for _ in range(n_ch)]
        self._n_z = n_z
        self._warp_slice_true = lambda c, z: rng.random((H, W)).astype(np.float32)
        self._warp_crop_true = lambda c, z, x0, y0, cw: np.full((cw, cw), c + 1, np.float32)
        self._status = types.SimpleNamespace(set=lambda s: None)
        self._export_status = types.SimpleNamespace(set=lambda s: setattr(self, "status_text", s))
        self._fig = types.SimpleNamespace(
            savefig=lambda p, **kw: WRITES["png"].append(p),
            get_facecolor=lambda: "#000000")
        self.status_text = ""

    def update_idletasks(self):
        pass

    def _crop(self, full, px, py, cw, fill=0.0):
        H, W = full.shape[:2]
        half = cw // 2
        x0, y0 = int(round(px)) - half, int(round(py)) - half
        out = np.full((cw, cw), fill, np.float32)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W, x0 + cw), min(H, y0 + cw)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = full[sy0:sy1, sx0:sx1]
        return out


def main():
    _export, msgs = load_export()
    base = tempfile.mkdtemp()
    p = Picker(base)
    WRITES["tif"].clear(); WRITES["mrc"].clear(); WRITES["png"].clear()

    _export(p)

    picks_dir = os.path.join(base, "picks")
    print("\n--- lands in the picks folder ---------------------------------")
    check("picks dir created", os.path.isdir(picks_dir), True)
    check("tifs in picks dir",
          {os.path.dirname(x[0]) for x in WRITES["tif"]}, {picks_dir})
    check("mrcs in picks dir",
          {os.path.dirname(x[0]) for x in WRITES["mrc"]}, {picks_dir})
    check("screenshot in picks dir",
          [os.path.dirname(x) for x in WRITES["png"]], [picks_dir])

    print("\n--- same filenames as the pipeline picker ---------------------")
    check("mrc names", sorted(os.path.basename(x[0]) for x in WRITES["mrc"]),
          ["crop_fov_1.mrc", "crop_fov_2.mrc", "crop_fov_3.mrc"])
    check("tif names", sorted(os.path.basename(x[0]) for x in WRITES["tif"]),
          ["target_overlays_1.tif", "target_overlays_2.tif",
           "target_overlays_3.tif"])
    check("screenshot name", [os.path.basename(x) for x in WRITES["png"]],
          ["picker_screenshot.png"])

    print("\n--- contents --------------------------------------------------")
    cw = 50                                  # 0.25 um / 0.005 um-per-px
    check("tif stacks are (z, 1+channels, cw, cw)",
          {x[1] for x in WRITES["tif"]}, {(3, 3, cw, cw)})
    check("mrc crops are cw x cw", {x[1] for x in WRITES["mrc"]}, {(cw, cw)})
    check("tif written as ImageJ hyperstack", {x[3] for x in WRITES["tif"]}, {True})
    check("view_crop_path recorded",
          all(q.view_crop_path and q.view_crop_path.endswith(f"crop_fov_{q.pick_id}.mrc")
              for q in p._picks), True)

    print("\n--- no stage-position txt, and it reported success ------------")
    check("nothing else written to picks",
          sorted(os.listdir(picks_dir)), [])   # all writers are stubbed
    check("no txt path anywhere in the writes",
          any(str(x).endswith(".txt") for x in
              [w[0] for w in WRITES["tif"]] + [w[0] for w in WRITES["mrc"]]
              + WRITES["png"]), False)
    check("reported completion", [m[0] for m in msgs], ["info"])
    check("status line mentions the folder", picks_dir in p.status_text, True)

    print()
    print("=" * 62)
    print("STANDALONE EXPORT:", "all checks passed" if not FAILURES
          else f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
    print("=" * 62)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
