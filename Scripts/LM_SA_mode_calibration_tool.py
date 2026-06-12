"""
TEM <-> Acquired-Image Stage Calibration Tool
==============================================
A standalone GUI (inspired by the registration tool's stage picker) for
calibrating the relationship between two imaging states' stage frames.

Workflow
--------
  1. Load MRC + mdoc            Assembles the SerialEM TEM montage and fits the
                                montage-pixel -> stage (um) mapping from the
                                tile StagePositions (similarity / affine fit,
                                same approach as the picker).
  2. Pick 4 points (left pane)  Left-click four features on the TEM montage.
                                Each becomes a stage position (S1).
  3. Choose imaging state       Dropdown: LMMM, MMM, 33kx_tomo, 81kx_tomo,
                                64kx_tomo, HMMM.
  4. Acquire                    Sets the imaging state, moves the stage to the
                                CENTRE of the four S1 points, records one image,
                                and shows it in the right pane.
  5. Pick 4 points (right pane) Left-click the same four features on the
                                acquired image. Each becomes a stage position
                                (S2) in the acquired image's frame.
  6. Compute transform          A similarity transform mapping S1 -> S2 is fit
                                and appended as one line to the output window.
  7. Export output              Save all output lines to a .txt file.

Mouse controls (both panes)
---------------------------
  Scroll wheel       - zoom centred on cursor
  Middle-click drag  - pan
  Shift + left drag  - pan (laptop / trackpad friendly)
  Left-click         - place a point (max 4 per pane)
  Right-click        - remove the last point in that pane

Microscope control
------------------
  Acquiring images, moving the stage and setting imaging states require
  SerialEM's Python module (`import serialem`).  If that module is importable
  this tool drives SerialEM directly; otherwise it falls back to a MOCK
  microscope that synthesises an image so the whole UI can be exercised without
  a scope.  The real SerialEM commands live in SerialEMMicroscope and are
  marked "VERIFY" where the exact command name/signature may differ between
  SerialEM versions - check them against your installation.

Dependencies
------------
  pip install numpy matplotlib mrcfile scikit-image
  (tkinter ships with Python; on Debian/Ubuntu: apt install python3-tk)
"""

import os
import sys
import re
import math
import datetime
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
#import serialem as sem

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import mrcfile
    _HAVE_MRCFILE = True
except ImportError:
    _HAVE_MRCFILE = False

try:
    from skimage.transform import estimate_transform
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False


# ---------------------------------------------------------------------------
# Theme + constants
# ---------------------------------------------------------------------------
BG   = "#1e1e2e"; BG2 = "#313244"; BG3 = "#45475a"; FG = "#cdd6f4"
ACC  = "#89b4fa"; ACC2 = "#a6e3a1"; RED = "#f38ba8"; CYA = "#89dceb"
YEL  = "#f9e2af"; MAG = "#cba6f7"
ZOOM_FACTOR = 1.25
MAX_DISP_PX = 4000          # downsample display to at most this on the long axis
N_POINTS    = 4             # points required per pane
_SHIFT_MASK = 0x0001        # Tk event.state bit for Shift

IMAGING_STATES = ["LMMM", "MMM", "33kx_tomo", "81kx_tomo", "64kx_tomo", "HMMM"]

PT_COLORS = [CYA, YEL, ACC2, MAG]   # per-point colours (1..4)


def _shift_held(mpl_event):
    """True when Shift was held during a matplotlib mouse event (TkAgg)."""
    ge = getattr(mpl_event, "guiEvent", None)
    if ge is not None and hasattr(ge, "state"):
        try:
            return bool(int(ge.state) & _SHIFT_MASK)
        except (TypeError, ValueError):
            pass
    return getattr(mpl_event, "key", None) in ("shift", "Shift")


# ---------------------------------------------------------------------------
# Pan / Zoom (scroll = zoom, middle-drag or shift+left-drag = pan)
# ---------------------------------------------------------------------------
class PanZoom:
    def __init__(self, ax, canvas):
        self.ax, self.cv, self._pan = ax, canvas, None
        w = canvas.get_tk_widget()
        w.bind("<MouseWheel>",            self._wheel,   add="+")
        w.bind("<Button-4>",              self._sup,     add="+")
        w.bind("<Button-5>",              self._sdn,     add="+")
        w.bind("<Button-2>",              self._pstart,  add="+")
        w.bind("<B2-Motion>",             self._pmove,   add="+")
        w.bind("<ButtonRelease-2>",       self._pend,    add="+")
        w.bind("<Shift-Button-1>",        self._pstart,  add="+")
        w.bind("<Shift-B1-Motion>",       self._pmove,   add="+")
        w.bind("<Shift-ButtonRelease-1>", self._pend,    add="+")
        w.bind("<ButtonRelease-1>",       self._pend,    add="+")

    def _in(self, tx, ty):
        h = self.cv.get_tk_widget().winfo_height()
        bb = self.ax.get_window_extent()
        return bb.x0 <= tx <= bb.x1 and bb.y0 <= (h - ty) <= bb.y1

    def _data(self, tx, ty):
        h = self.cv.get_tk_widget().winfo_height()
        return self.ax.transData.inverted().transform((tx, h - ty))

    def _zoom(self, tx, ty, f):
        if not self._in(tx, ty):
            return
        cx, cy = self._data(tx, ty)
        xl, xr = self.ax.get_xlim(); yl, yr = self.ax.get_ylim()
        self.ax.set_xlim(cx + (xl - cx) * f, cx + (xr - cx) * f)
        self.ax.set_ylim(cy + (yl - cy) * f, cy + (yr - cy) * f)
        self.cv.draw_idle()

    def _wheel(self, e):
        self._zoom(e.x, e.y, 1 / ZOOM_FACTOR if e.delta > 0 else ZOOM_FACTOR)
    def _sup(self, e): self._zoom(e.x, e.y, 1 / ZOOM_FACTOR)
    def _sdn(self, e): self._zoom(e.x, e.y, ZOOM_FACTOR)

    def _pstart(self, e):
        if self._in(e.x, e.y):
            self._pan = (e.x, e.y, self.ax.get_xlim(), self.ax.get_ylim())

    def _pmove(self, e):
        if self._pan is None:
            return
        x0, y0, xl, yl = self._pan
        bb = self.ax.get_window_extent()
        if bb.width < 1 or bb.height < 1:
            return
        dx = (e.x - x0) / bb.width  * (xl[1] - xl[0])
        dy = (e.y - y0) / bb.height * (yl[1] - yl[0])
        self.ax.set_xlim(xl[0] - dx, xl[1] - dx)
        self.ax.set_ylim(yl[0] + dy, yl[1] + dy)
        self.cv.draw_idle()

    def _pend(self, e):
        self._pan = None


# ---------------------------------------------------------------------------
# mdoc parser
# ---------------------------------------------------------------------------
def _coerce(val_str):
    parts = val_str.split()
    if not parts:
        return val_str
    try:
        nums = [int(p) if re.fullmatch(r"-?\d+", p) else float(p) for p in parts]
        return nums[0] if len(nums) == 1 else nums
    except ValueError:
        return val_str.strip()


def parse_mdoc(path):
    global_info, pieces, mont_sections = {}, [], []
    current, ctype = global_info, "global"
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"\[ZValue\s*=\s*(\d+)\]", line)
            if m:
                current = {"ZValue": int(m.group(1))}
                pieces.append(current); ctype = "zvalue"; continue
            m = re.match(r"\[MontSection\s*=\s*(\d+)\]", line)
            if m:
                current = {"MontSection": int(m.group(1))}
                mont_sections.append(current); ctype = "mont"; continue
            if line.startswith("["):
                ctype = "text"; continue
            if ctype == "text":
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                current[key.strip()] = _coerce(val.strip())
    return global_info, pieces, mont_sections


# ---------------------------------------------------------------------------
# MRC montage assembler  (memory-efficient: mmap + per-tile downsample)
# ---------------------------------------------------------------------------
def _cosine_weight(h, w, feather):
    feather = max(1, int(feather))

    def ramp(n):
        r = np.ones(n, np.float32)
        f = min(feather, n // 2)
        if f > 0:
            t = np.linspace(0, math.pi / 2, f, dtype=np.float32)
            r[:f] = np.sin(t); r[-f:] = np.sin(t)[::-1]
        return r
    return np.outer(ramp(h), ramp(w))


def assemble_montage(mrc_path, pieces, img_h, img_w, feather_px,
                     max_disp=MAX_DISP_PX, status_cb=None):
    """Assemble a montage for DISPLAY at reduced resolution but report the
    FULL-resolution canvas size so coordinate maths stays in full-res px.
    Returns (display_array, (full_h, full_w), ds)."""
    max_x = max_y = 0
    for p in pieces:
        c = p.get("PieceCoordinates", [0, 0, 0])
        if isinstance(c, (list, tuple)):
            max_x = max(max_x, int(c[0]) + img_w)
            max_y = max(max_y, int(c[1]) + img_h)
    if max_x == 0:
        max_x = img_w
    if max_y == 0:
        max_y = img_h
    full_w, full_h = int(max_x), int(max_y)

    ds = max(1, int(math.ceil(max(full_w, full_h) / float(max_disp))))
    cw = -(-full_w // ds)
    ch = -(-full_h // ds)
    canvas  = np.zeros((ch, cw), np.float64)
    weights = np.zeros((ch, cw), np.float64)

    n = len(pieces)
    with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
        n_frames = mrc.data.shape[0] if mrc.data.ndim == 3 else 1
        for i, p in enumerate(pieces):
            z = p.get("ZValue", i)
            if z >= n_frames:
                continue
            tile = (mrc.data[z] if mrc.data.ndim == 3 else mrc.data).astype(np.float64)
            if ds > 1:
                tile = tile[::ds, ::ds]
            dh, dw = tile.shape
            wmap = _cosine_weight(dh, dw, max(1, feather_px // ds)).astype(np.float64)
            c  = p.get("PieceCoordinates", [0, 0, 0])
            cx = int(c[0]) if isinstance(c, (list, tuple)) else 0
            cy = int(c[1]) if isinstance(c, (list, tuple)) else 0
            ox = int(round(cx / ds)); oy = int(round(cy / ds))
            sy0 = max(0, -oy); sx0 = max(0, -ox)
            sy1 = min(dh, ch - oy); sx1 = min(dw, cw - ox)
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            dy0 = oy + sy0; dx0 = ox + sx0
            dy1 = dy0 + (sy1 - sy0); dx1 = dx0 + (sx1 - sx0)
            canvas [dy0:dy1, dx0:dx1] += tile[sy0:sy1, sx0:sx1] * wmap[sy0:sy1, sx0:sx1]
            weights[dy0:dy1, dx0:dx1] += wmap[sy0:sy1, sx0:sx1]
            if status_cb and i % max(1, n // 10) == 0:
                status_cb(f"Assembling tile {i+1}/{n} (1/{ds} res) ...")

    valid = weights > 0
    canvas[valid] /= weights[valid]
    return normalize_disp(canvas.astype(np.float32)), (full_h, full_w), ds


def normalize_disp(arr):
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi > lo:
        return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)
    return np.zeros_like(arr, np.float32)


def load_tem_montage(mrc_path, g, pieces, status_cb=None):
    """Return (display_img, (full_h, full_w), ds, info_str)."""
    if not _HAVE_MRCFILE:
        raise RuntimeError("mrcfile not installed - pip install mrcfile")
    img_size = g.get("ImageSize", None)
    with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
        shape = mrc.data.shape
    if img_size is None:
        img_w, img_h = int(shape[-1]), int(shape[-2])
    elif isinstance(img_size, (int, float)):
        img_w = img_h = int(img_size)
    else:
        img_w, img_h = int(img_size[0]), int(img_size[1])

    ps = g.get("PieceSpacing", [img_w - img_w // 10, img_h - img_h // 10])
    if isinstance(ps, (int, float)):
        ps_x = ps_y = int(ps)
    else:
        ps_x, ps_y = int(ps[0]), int(ps[1])
    feather = max(4, min(img_w - ps_x, img_h - ps_y, img_w // 8))

    disp, (fh, fw), ds = assemble_montage(mrc_path, pieces, img_h, img_w,
                                          feather, status_cb=status_cb)
    extra = f"  (display 1/{ds})" if ds > 1 else ""
    info = f"montage {fw}x{fh}{extra} from {len(pieces)} tiles"
    return disp, (fh, fw), ds, info, (img_h, img_w)


# ---------------------------------------------------------------------------
# pixel -> stage fit  (montage pixel coords -> stage um)
# ---------------------------------------------------------------------------
def fit_pixel_to_stage(pieces, tile_w, tile_h):
    """Fit a no-shear similarity (rotation + scale + optional flip + shift)
    mapping montage-pixel coords -> stage um from each tile's pixel centre and
    StagePosition.  Returns a dict or None (too few tiles / no stage spread)."""
    pts_px, pts_st = [], []
    for p in pieces:
        sp = p.get("StagePosition", None)
        if not isinstance(sp, (list, tuple)) or len(sp) < 2:
            continue
        c = p.get("PieceCoordinates", [0, 0, 0])
        if not isinstance(c, (list, tuple)) or len(c) < 2:
            continue
        pts_px.append((float(c[0]) + tile_w / 2.0, float(c[1]) + tile_h / 2.0))
        pts_st.append((float(sp[0]), float(sp[1])))
    if len(pts_px) < 2:
        return None
    P = np.asarray(pts_px, np.float64); S = np.asarray(pts_st, np.float64)
    if float(S.std(axis=0).max()) < 1e-6:
        return None
    n = len(P)
    px, py = P[:, 0], P[:, 1]; sx, sy = S[:, 0], S[:, 1]
    ones, zeros = np.ones(n), np.zeros(n)
    best = None
    for reflect in (False, True):
        A = np.zeros((2 * n, 4)); b = np.zeros(2 * n)
        if not reflect:
            A[0::2] = np.column_stack([px, -py, ones, zeros])
            A[1::2] = np.column_stack([py,  px, zeros, ones])
        else:
            A[0::2] = np.column_stack([px,  py, ones, zeros])
            A[1::2] = np.column_stack([-py, px, zeros, ones])
        b[0::2] = sx; b[1::2] = sy
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        rmse = float(np.sqrt(np.mean((A @ sol - b) ** 2)))
        cand = {"reflect": reflect, "a": float(sol[0]), "b": float(sol[1]),
                "tx": float(sol[2]), "ty": float(sol[3]), "rmse": rmse, "n": n}
        if best is None or rmse < best["rmse"]:
            best = cand
    return best


def make_pixel_to_stage(pieces, tile_w, tile_h, pix_um):
    """Return (fn, description) where fn(px, py) -> (stage_x_um, stage_y_um)."""
    fit = fit_pixel_to_stage(pieces, tile_w, tile_h)
    if fit is not None:
        a, b, tx, ty, refl = fit["a"], fit["b"], fit["tx"], fit["ty"], fit["reflect"]

        def fn(px, py):
            if not refl:
                return a * px - b * py + tx, b * px + a * py + ty
            return a * px + b * py + tx, b * px - a * py + ty
        kind = "rotation+flip" if refl else "rotation"
        return fn, f"similarity fit ({kind}, {fit['n']} tiles, rmse {fit['rmse']:.3f} um)"

    # fallback: per-tile inverse-distance interpolation (no rotation)
    def fn(px, py):
        w_sum = sx_sum = sy_sum = 0.0
        for p in pieces:
            c = p.get("PieceCoordinates", [0, 0, 0])
            tx0 = float(c[0]) if isinstance(c, (list, tuple)) else 0.0
            ty0 = float(c[1]) if isinstance(c, (list, tuple)) else 0.0
            cx = tx0 + tile_w / 2.0; cy = ty0 + tile_h / 2.0
            dist = max(0.01, math.hypot(px - cx, py - cy))
            w = 1.0 / dist
            sp = p.get("StagePosition", [0.0, 0.0])
            if not isinstance(sp, (list, tuple)) or len(sp) < 2:
                sp = [0.0, 0.0]
            sx = float(sp[0]) + (px - cx) * pix_um
            sy = float(sp[1]) - (py - cy) * pix_um
            w_sum += w; sx_sum += w * sx; sy_sum += w * sy
        if w_sum == 0:
            return 0.0, 0.0
        return sx_sum / w_sum, sy_sum / w_sum
    return fn, "per-tile interpolation (no rotation fit)"


# ---------------------------------------------------------------------------
# Microscope backends
# ---------------------------------------------------------------------------
class MockMicroscope:
    """Synthetic microscope so the whole UI works without SerialEM.
    `acquire` returns a fabricated image plus plausible metadata."""
    name = "MOCK (no SerialEM module found)"

    # nominal pixel size (um/px) per imaging state - purely illustrative
    _PIX_UM = {"LMMM": 0.20, "MMM": 0.05, "33kx_tomo": 0.0042,
               "81kx_tomo": 0.0017, "64kx_tomo": 0.0022, "HMMM": 0.012}

    def set_imaging_state(self, state):
        return f"[mock] imaging state -> {state}"

    def move_stage(self, x, y):
        return f"[mock] stage -> ({x:.3f}, {y:.3f}) um"

    def acquire(self, state, center):
        h = w = 1024
        px_um = self._PIX_UM.get(state, 0.05)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = (0.5 + 0.5 * np.sin(xx / 40.0) * np.cos(yy / 37.0))
        img += 0.15 * np.random.rand(h, w).astype(np.float32)
        for (fx, fy) in [(0.30, 0.30), (0.70, 0.35), (0.40, 0.72), (0.66, 0.68)]:
            cy0, cx0 = int(fy * h), int(fx * w)
            img[max(0, cy0 - 8):cy0 + 8, max(0, cx0 - 8):cx0 + 8] = 1.0
        img = np.clip(img, 0, 1)
        meta = {"center": (float(center[0]), float(center[1])),
                "pixel_size_um": px_um, "rotation_deg": 0.0, "shape": (h, w)}
        return img, meta


class SerialEMMicroscope:
    """Drives SerialEM through its Python module.  The exact command names /
    signatures can differ between SerialEM versions - lines marked VERIFY are
    the ones to check against your installation's Script Commands list."""
    name = "SerialEM"

    def __init__(self):
        import serialem  # noqa: F401  (only importable on the SerialEM PC)
        self.sem = serialem

    def set_imaging_state(self, state):
        # VERIFY: imaging states are addressed by name; older versions may use
        # SetImagingState / a numeric index instead of GoToImagingState.
        self.sem.GoToImagingState(state)
        return f"imaging state -> {state}"

    def move_stage(self, x, y):
        # VERIFY: MoveStageTo(x_um, y_um) moves X/Y in microns.
        self.sem.MoveStageTo(float(x), float(y))
        return f"stage -> ({x:.3f}, {y:.3f}) um"

    def acquire(self, state, center):
        # VERIFY: Record() acquires into buffer A.
        self.sem.Record()
        # VERIFY: bufferImage('A') returns the image as a numpy-compatible
        # array; some versions expose it as Image('A') or bufferImage(0).
        arr = np.asarray(self.sem.bufferImage("A"), dtype=np.float32)
        # VERIFY: ReportCurrentPixelSize returns nm/px for the given buffer.
        try:
            px_nm = float(self.sem.ReportCurrentPixelSize("A"))
        except Exception:
            px_nm = float("nan")
        px_um = px_nm / 1000.0 if px_nm == px_nm else 1.0   # NaN-safe
        meta = {"center": (float(center[0]), float(center[1])),
                "pixel_size_um": px_um,
                "rotation_deg": 0.0,           # VERIFY: query if you need it
                "shape": arr.shape}
        return arr, meta


def get_microscope():
    try:
        import serialem  # noqa: F401
        return SerialEMMicroscope()
    except Exception:
        return MockMicroscope()


# ---------------------------------------------------------------------------
# Similarity transform between two point sets
# ---------------------------------------------------------------------------
def similarity_S1_to_S2(S1, S2):
    """Fit a similarity transform mapping S1 -> S2.  Returns a dict with the
    3x3 matrix, scale, rotation (deg), translation and RMSE (in S2 units)."""
    S1 = np.asarray(S1, float); S2 = np.asarray(S2, float)
    if _HAVE_SKIMAGE:
        t = estimate_transform("similarity", S1, S2)
        M = np.asarray(t.params, float)
        scale = float(t.scale); rot = float(np.degrees(t.rotation))
        tx, ty = float(t.translation[0]), float(t.translation[1])
        pred = t(S1)
    else:
        M, scale, rot, tx, ty = _similarity_lstsq(S1, S2)
        pred = (M @ np.column_stack([S1, np.ones(len(S1))]).T).T[:, :2]
    rmse = float(np.sqrt(np.mean(np.sum((pred - S2) ** 2, axis=1))))
    return {"matrix": M, "scale": scale, "rotation_deg": rot,
            "tx": tx, "ty": ty, "rmse": rmse}


def _similarity_lstsq(S1, S2):
    """Fallback similarity fit (no skimage): solve sx=a*x-b*y+tx, sy=b*x+a*y+ty."""
    n = len(S1)
    x, y = S1[:, 0], S1[:, 1]
    A = np.zeros((2 * n, 4)); b = np.zeros(2 * n)
    A[0::2] = np.column_stack([x, -y, np.ones(n), np.zeros(n)])
    A[1::2] = np.column_stack([y,  x, np.zeros(n), np.ones(n)])
    b[0::2] = S2[:, 0]; b[1::2] = S2[:, 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, tx, ty = map(float, sol)
    M = np.array([[a, -bb, tx], [bb, a, ty], [0, 0, 1]], float)
    return M, math.hypot(a, bb), math.degrees(math.atan2(bb, a)), tx, ty


# ---------------------------------------------------------------------------
# Pick pane  -  an image + up to N_POINTS clickable points
# ---------------------------------------------------------------------------
class PickPane(ttk.LabelFrame):
    def __init__(self, parent, title, on_change, **kw):
        super().__init__(parent, text=title, padding=4, **kw)
        self._on_change = on_change
        self._ds = 1
        self._pixel_to_stage = None
        self._picks = []                 # list of dicts
        self._artists = []
        self.rowconfigure(0, weight=1); self.columnconfigure(0, weight=1)

        fig = Figure(figsize=(5, 5), facecolor=BG)
        self._ax = fig.add_subplot(111)
        self._ax.set_facecolor(BG); self._ax.axis("off")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        self._fig = fig
        self._canvas = FigureCanvasTkAgg(fig, master=self)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._im = None
        self._canvas.draw()
        PanZoom(self._ax, self._canvas)
        self._canvas.mpl_connect("button_press_event", self._on_click)

        bar = ttk.Frame(self); bar.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self._count = tk.StringVar(value="0 / 4 points")
        ttk.Label(bar, textvariable=self._count, style="Sm.TLabel",
                  foreground=CYA).pack(side="left")
        ttk.Button(bar, text="Remove last", style="Sm.TButton",
                   command=self._remove_last).pack(side="right", padx=2)
        ttk.Button(bar, text="Clear", style="Sm.TButton",
                   command=self.clear).pack(side="right", padx=2)

        self._coords = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._coords, style="Sm.TLabel",
                  justify="left", foreground=BG3, wraplength=420).grid(
                      row=2, column=0, sticky="ew", pady=(2, 0))

    # -- public --
    def set_image(self, disp_img, ds, pixel_to_stage):
        self._ds = ds
        self._pixel_to_stage = pixel_to_stage
        self._picks = []
        self._ax.cla(); self._ax.set_facecolor(BG); self._ax.axis("off")
        self._im = self._ax.imshow(disp_img, cmap="gray", origin="upper",
                                   aspect="equal", interpolation="nearest")
        self._canvas.draw_idle()
        self._update_labels()

    def clear(self):
        self._picks = []
        self._redraw_points()
        self._update_labels()

    def has_all(self):
        return len(self._picks) == N_POINTS

    def stage_points(self):
        return [(p["sx"], p["sy"]) for p in self._picks]

    def is_ready(self):
        return self._im is not None and self._pixel_to_stage is not None

    # -- internal --
    def _remove_last(self):
        if self._picks:
            self._picks.pop()
            self._redraw_points(); self._update_labels()

    def _on_click(self, event):
        if event.inaxes is not self._ax or event.xdata is None:
            return
        if event.button == 3:                      # right-click = undo
            self._remove_last(); return
        if event.button != 1 or _shift_held(event):
            return
        if not self.is_ready() or len(self._picks) >= N_POINTS:
            return
        px = event.xdata * self._ds
        py = event.ydata * self._ds
        sx, sy = self._pixel_to_stage(px, py)
        self._picks.append({"dx": event.xdata, "dy": event.ydata,
                            "px": px, "py": py, "sx": sx, "sy": sy})
        self._redraw_points(); self._update_labels()

    def _redraw_points(self):
        for a in self._artists:
            try: a.remove()
            except Exception: pass
        self._artists = []
        for i, p in enumerate(self._picks):
            col = PT_COLORS[i % len(PT_COLORS)]
            x, y = p["dx"], p["dy"]; arm = 12
            l1, = self._ax.plot([x - arm, x + arm], [y, y], color=col, lw=1.2, zorder=5)
            l2, = self._ax.plot([x, x], [y - arm, y + arm], color=col, lw=1.2, zorder=5)
            dot, = self._ax.plot(x, y, "o", color=col, markersize=6,
                                 markeredgecolor="white", markeredgewidth=0.8, zorder=6)
            txt = self._ax.text(x + 8, y - 8, str(i + 1), color=col,
                                fontsize=9, fontweight="bold", zorder=7)
            self._artists.extend([l1, l2, dot, txt])
        self._canvas.draw_idle()

    def _update_labels(self):
        self._count.set(f"{len(self._picks)} / {N_POINTS} points")
        lines = [f"{i+1}: ({p['sx']:.3f}, {p['sy']:.3f}) um"
                 for i, p in enumerate(self._picks)]
        self._coords.set("   ".join(lines))
        if self._on_change:
            self._on_change()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class CalibApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TEM <-> Acquired-Image Stage Calibration")
        self.configure(bg=BG)
        self.minsize(1150, 780)

        self.scope = get_microscope()

        self._mrc_path = ""
        self._pieces = []
        self._pix_um = 1.0
        self._tile_hw = (4096, 4096)
        self._tem_pix_to_stage = None
        self._last_center = None
        self._acq_meta = None
        self._n_lines = 0

        self._build_styles()
        self._build_ui()
        self._refresh_buttons()

    # -- styles --
    def _build_styles(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        s.configure("Sm.TLabel", background=BG, foreground=FG, font=("Segoe UI", 9))
        s.configure("TButton", background=ACC, foreground=BG,
                    font=("Segoe UI", 10, "bold"), padding=5)
        s.map("TButton", background=[("active", CYA), ("disabled", BG3)])
        s.configure("Sm.TButton", background=BG3, foreground=FG,
                    font=("Segoe UI", 9), padding=2)
        s.map("Sm.TButton", background=[("active", ACC), ("disabled", BG2)])
        s.configure("Accent.TButton", background=ACC2, foreground=BG,
                    font=("Segoe UI", 11, "bold"), padding=7)
        s.map("Accent.TButton", background=[("active", CYA), ("disabled", BG3)])
        s.configure("Acq.TButton", background=MAG, foreground=BG,
                    font=("Segoe UI", 11, "bold"), padding=7)
        s.map("Acq.TButton", background=[("active", CYA), ("disabled", BG3)])
        s.configure("TLabelframe", background=BG, relief="groove")
        s.configure("TLabelframe.Label", background=BG, foreground=CYA,
                    font=("Segoe UI", 10, "bold"))
        s.configure("TCombobox", fieldbackground=BG2, background=BG2)

    # -- UI --
    def _build_ui(self):
        top = ttk.Frame(self, padding=(8, 6, 8, 0)); top.pack(fill="x")
        ttk.Button(top, text="Load MRC + mdoc",
                   command=self._load).pack(side="left", padx=3)
        self._info = tk.StringVar(value="No TEM montage loaded")
        ttk.Label(top, textvariable=self._info, style="Sm.TLabel",
                  foreground=BG3).pack(side="left", padx=6)

        ttk.Label(top, text=f"Microscope: {self.scope.name}",
                  style="Sm.TLabel", foreground=YEL).pack(side="right", padx=6)

        ctl = ttk.Frame(self, padding=(8, 4)); ctl.pack(fill="x")
        ttk.Label(ctl, text="Imaging state:", style="Sm.TLabel").pack(side="left")
        self._state_var = tk.StringVar(value=IMAGING_STATES[0])
        ttk.Combobox(ctl, textvariable=self._state_var, width=12, state="readonly",
                     values=IMAGING_STATES).pack(side="left", padx=(4, 12))
        self._btn_acq = ttk.Button(ctl, text="Acquire  (move to centre + image)",
                                   style="Acq.TButton", command=self._acquire)
        self._btn_acq.pack(side="left", padx=3)
        self._btn_comp = ttk.Button(ctl, text="Compute transform -> output",
                                    style="Accent.TButton", command=self._compute)
        self._btn_comp.pack(side="left", padx=3)
        ttk.Label(ctl,
                  text="scroll=zoom   middle/shift+drag=pan   left=pick(4)   right=undo",
                  style="Sm.TLabel", foreground=BG3).pack(side="right", padx=6)

        panes = ttk.Frame(self, padding=(8, 4)); panes.pack(fill="both", expand=True)
        panes.columnconfigure(0, weight=1); panes.columnconfigure(1, weight=1)
        panes.rowconfigure(0, weight=1)
        self._pane_tem = PickPane(panes, "1.  TEM montage  -  pick 4 points",
                                  self._refresh_buttons)
        self._pane_tem.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._pane_acq = PickPane(panes, "2.  Acquired image  -  pick 4 points",
                                  self._refresh_buttons)
        self._pane_acq.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        out_lf = ttk.LabelFrame(self, text="Output  (one transform per line)",
                                padding=4)
        out_lf.pack(fill="x", padx=8, pady=(0, 4))
        out_lf.columnconfigure(0, weight=1)
        self._out = tk.Text(out_lf, height=7, bg=BG2, fg=FG, insertbackground=FG,
                            font=("Consolas", 9), wrap="none")
        self._out.grid(row=0, column=0, sticky="ew")
        osb = ttk.Scrollbar(out_lf, orient="vertical", command=self._out.yview)
        self._out.configure(yscrollcommand=osb.set)
        osb.grid(row=0, column=1, sticky="ns")
        self._out.insert("end",
            "# TEM<->acquired-image similarity transforms (maps S1 TEM stage "
            "-> S2 acquired-image stage, units um)\n")
        btnrow = ttk.Frame(out_lf); btnrow.grid(row=1, column=0, columnspan=2,
                                                sticky="ew", pady=(4, 0))
        self._btn_exp = ttk.Button(btnrow, text="Export output to TXT",
                                   style="Accent.TButton", command=self._export)
        self._btn_exp.pack(side="left")
        ttk.Button(btnrow, text="Clear output",
                   style="Sm.TButton", command=self._clear_output).pack(
                       side="left", padx=6)

        sf = ttk.Frame(self, padding=(8, 0, 8, 6)); sf.pack(fill="x")
        self._status = tk.StringVar(value="Load a TEM MRC + mdoc to begin.")
        ttk.Label(sf, textvariable=self._status, style="Sm.TLabel",
                  foreground=CYA).pack(side="left")

    # -- loading --
    def _load(self):
        path = filedialog.askopenfilename(
            title="Open MRC montage or SerialEM mdoc (either one)",
            filetypes=[("MRC / mdoc", "*.mrc *.rec *.mrcs *.map *.mdoc"),
                       ("All", "*.*")])
        if not path:
            return
        stem, ext = os.path.splitext(path); ext = ext.lower()
        mrc_exts = {".mrc", ".rec", ".mrcs", ".map"}
        if ext in mrc_exts:
            mrc_path = path
            mdoc_path = None
            for suf in (".mdoc", ".Mdoc", ".MDOC"):
                if os.path.isfile(path + suf):
                    mdoc_path = path + suf; break
                if os.path.isfile(stem + suf):
                    mdoc_path = stem + suf; break
            if mdoc_path is None:
                mdoc_path = filedialog.askopenfilename(
                    title="Locate matching .mdoc", initialdir=os.path.dirname(path),
                    filetypes=[("mdoc", "*.mdoc"), ("All", "*.*")])
                if not mdoc_path:
                    return
        elif ext == ".mdoc":
            mdoc_path = path; mrc_path = None
            for mext in (".mrc", ".rec", ".mrcs", ".map"):
                if os.path.isfile(stem + mext):
                    mrc_path = stem + mext; break
            if mrc_path is None:
                mrc_path = filedialog.askopenfilename(
                    title="Locate matching MRC", initialdir=os.path.dirname(path),
                    filetypes=[("MRC", "*.mrc *.rec *.mrcs *.map"), ("All", "*.*")])
                if not mrc_path:
                    return
        else:
            messagebox.showerror("Unknown file", f"Unrecognised extension '{ext}'.")
            return

        try:
            g, pieces, _ = parse_mdoc(mdoc_path)
        except Exception as e:
            messagebox.showerror("mdoc error", str(e)); return
        if not pieces:
            messagebox.showerror("mdoc error", "No [ZValue] tile sections found.")
            return

        ps_ang = g.get("PixelSpacing", 10.0)
        if isinstance(ps_ang, (list, tuple)):
            ps_ang = float(ps_ang[0])
        self._pix_um = float(ps_ang) / 10000.0   # Angstrom/px -> um/px

        def status_cb(msg):
            self._status.set(msg); self.update_idletasks()

        try:
            disp, (fh, fw), ds, info, (img_h, img_w) = load_tem_montage(
                mrc_path, g, pieces, status_cb=status_cb)
        except Exception as e:
            messagebox.showerror("MRC error", str(e)); return

        self._mrc_path = mrc_path
        self._pieces = pieces
        self._tile_hw = (img_h, img_w)
        fn, desc = make_pixel_to_stage(pieces, img_w, img_h, self._pix_um)
        self._tem_pix_to_stage = fn

        self._pane_tem.set_image(disp, ds, fn)
        self._pane_acq.clear()
        self._acq_meta = None
        self._info.set(f"{os.path.basename(mrc_path)}  |  {info}  |  "
                       f"{self._pix_um:.4f} um/px")
        self._status.set(f"TEM montage loaded.  pixel->stage: {desc}.  "
                         f"Pick 4 points on the left pane.")
        self._refresh_buttons()

    # -- acquire --
    def _acquire(self):
        if not self._pane_tem.has_all():
            messagebox.showwarning("Need 4 points",
                "Pick 4 points on the TEM montage first."); return
        S1 = self._pane_tem.stage_points()
        cx = float(np.mean([p[0] for p in S1]))
        cy = float(np.mean([p[1] for p in S1]))
        self._last_center = (cx, cy)
        state = self._state_var.get()
        try:
            self._status.set(self.scope.set_imaging_state(state)); self.update_idletasks()
            self._status.set(self.scope.move_stage(cx, cy)); self.update_idletasks()
            img, meta = self.scope.acquire(state, (cx, cy))
        except Exception as e:
            messagebox.showerror("Acquisition failed", str(e)); return

        self._acq_meta = meta
        disp = normalize_disp(np.asarray(img, np.float32))
        ds = max(1, int(math.ceil(max(disp.shape) / float(MAX_DISP_PX))))
        if ds > 1:
            disp = disp[::ds, ::ds]
        fn = self._make_acq_pixel_to_stage(meta)
        self._pane_acq.set_image(disp, ds, fn)
        self._status.set(
            f"Acquired in '{state}' at centre ({cx:.3f}, {cy:.3f}) um, "
            f"{meta['pixel_size_um']:.4f} um/px.  Pick the same 4 points on the right.")
        self._refresh_buttons()

    def _make_acq_pixel_to_stage(self, meta):
        """Pixel (full-res) -> stage um for the acquired image, from its centre
        stage position, pixel size and optional camera-to-stage rotation."""
        cx, cy = meta["center"]
        px_um = float(meta["pixel_size_um"])
        h, w = meta["shape"]
        theta = math.radians(float(meta.get("rotation_deg", 0.0)))
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        def fn(px, py):
            vx = (px - w / 2.0) * px_um
            vy = -(py - h / 2.0) * px_um          # image y-down -> stage y-up
            sx = cx + vx * cos_t - vy * sin_t
            sy = cy + vx * sin_t + vy * cos_t
            return sx, sy
        return fn

    # -- compute --
    def _compute(self):
        if not (self._pane_tem.has_all() and self._pane_acq.has_all()):
            messagebox.showwarning("Need both sets",
                "Pick 4 points on BOTH panes before computing."); return
        S1 = self._pane_tem.stage_points()
        S2 = self._pane_acq.stage_points()
        try:
            r = similarity_S1_to_S2(S1, S2)
        except Exception as e:
            messagebox.showerror("Transform failed", str(e)); return

        M = r["matrix"]
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cx, cy = self._last_center if self._last_center else (float("nan"),) * 2
        matrix_str = ";".join(",".join(f"{v:.8g}" for v in row) for row in M[:2])
        line = "\t".join([
            ts,
            f"state={self._state_var.get()}",
            f"centre_um=({cx:.3f},{cy:.3f})",
            f"scale={r['scale']:.6f}",
            f"rot_deg={r['rotation_deg']:.4f}",
            f"tx_um={r['tx']:.4f}",
            f"ty_um={r['ty']:.4f}",
            f"rmse_um={r['rmse']:.4f}",
            f"matrix=[{matrix_str}]",
        ])
        self._out.insert("end", line + "\n")
        self._out.see("end")
        self._n_lines += 1
        self._status.set(
            f"Transform #{self._n_lines} added:  scale {r['scale']:.4f}, "
            f"rot {r['rotation_deg']:.3f} deg, rmse {r['rmse']:.4f} um.")
        self._refresh_buttons()

    # -- export / clear --
    def _export(self):
        text = self._out.get("1.0", "end").strip()
        if not text or self._n_lines == 0:
            messagebox.showwarning("Nothing to export",
                "Compute at least one transform first."); return
        path = filedialog.asksaveasfilename(
            title="Export transforms", defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, "w") as fh:
                fh.write("# columns: timestamp  state  centre_um  scale  "
                         "rot_deg  tx_um  ty_um  rmse_um  matrix(3x3 rows 1-2)\n")
                fh.write(text + "\n")
            self._status.set(f"Exported {self._n_lines} transform(s) to "
                             f"{os.path.basename(path)}")
            messagebox.showinfo("Exported", f"Saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _clear_output(self):
        self._out.delete("1.0", "end")
        self._out.insert("end",
            "# TEM<->acquired-image similarity transforms (maps S1 TEM stage "
            "-> S2 acquired-image stage, units um)\n")
        self._n_lines = 0
        self._refresh_buttons()

    # -- enable/disable buttons by state --
    def _refresh_buttons(self):
        tem_ready = self._pane_tem.has_all()
        both_ready = tem_ready and self._pane_acq.has_all()
        self._btn_acq.configure(state="normal" if tem_ready else "disabled")
        self._btn_comp.configure(state="normal" if both_ready else "disabled")
        self._btn_exp.configure(state="normal" if self._n_lines > 0 else "disabled")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    CalibApp().mainloop()