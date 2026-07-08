"""
MRC / OME-TIFF Image Registration Tool
=======================================
Supports:
  - Single MRC file  (Load MRC)
  - Montage MRC + SerialEM mdoc  (Load MRC + mdoc)
    Assembles 3x3-or-larger SerialEM montages on demand and lets you
    scroll through all montage sections.
  - OME-TIFF stack with per-channel brightness/contrast
  - Export / Import Transform: save the fitted TIFF->MRC transform (3x3
    matrix + type + flip state) to a text file and re-apply it later, so a
    registration can be reused without re-picking landmarks.

Mouse controls (all image panels):
  Scroll wheel       - zoom in / out, centred on cursor
  Middle-click drag  - pan
  Shift + left drag  - pan (laptop / trackpad friendly)
  Left-click         - place landmark

NEW: Overlay window -> "Open Stage Picker"
  Opens a secondary window showing the composite overlay.
  Left-click on the image to record stage positions (IDW-interpolated
  from the SerialEM tile calibration data).  Points can be exported
  to a tab-separated .txt file.  A PNG screenshot is saved alongside
  the txt file automatically.
  A per-layer checkbox + brightness/contrast panel lets you show the TEM
  (MRC) reference and any z-stack channels individually or composited,
  without losing your zoom/pan or any placed points.
  Enter a FOV width (um) before exporting to also save, for every picked
  point, a registered z-stack crop (TEM + all channels x all z) as an
  ImageJ TIFF hyperstack named <txt-name>_<point#>.tif.

Display orientation:
  The TEM montage (MRC) is shown with a fixed display-only flip set by the
  module constants MONTAGE_FLIP_X / MONTAGE_FLIP_Y below.  The SAME flip is
  applied everywhere the MRC grid is drawn - the MRC panel, the overlay
  window and the stage picker - so all three are at the same orientation.
  It is purely cosmetic: clicks are un-mirrored before any coordinate use, so
  landmark coordinates, the fitted transform, stage positions and every saved
  file (overlay panels, FOV crops) stay in the true, un-flipped orientation.
  The OME-TIFF panel is NOT affected (it has its own separate flip controls).

Dependencies:
    pip install mrcfile tifffile scikit-image matplotlib numpy
"""

import sys
import os
import re
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from clem_correlation import CLEMCorrelator

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import mrcfile
except ImportError:
    sys.exit("Missing: mrcfile  ->  pip install mrcfile")
try:
    import tifffile
except ImportError:
    sys.exit("Missing: tifffile  ->  pip install tifffile")
try:
    from skimage.transform import estimate_transform, warp
except ImportError:
    sys.exit("Missing: scikit-image  ->  pip install scikit-image")


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
BG   = "#1e1e2e"
BG2  = "#313244"
BG3  = "#45475a"
FG   = "#cdd6f4"
ACC  = "#89b4fa"
ACC2 = "#a6e3a1"
RED  = "#f38ba8"
CYA  = "#89dceb"

CHANNEL_HEX  = ["#00ff00","#ff00ff","#00ffff","#ffff00","#ff8000","#8000ff"]
CHANNEL_COLORS = [
    (0.0,1.0,0.0),(1.0,0.0,1.0),(0.0,1.0,1.0),
    (1.0,1.0,0.0),(1.0,0.5,0.0),(0.5,0.0,1.0),
]
CHANNEL_COLOR_NAMES = ["green","magenta","cyan","yellow","orange","purple"]

PT_MRC  = "#FF4444"
PT_TIFF = "#44AAFF"
ZOOM_FACTOR = 1.25

# Display-only flip applied to the TEM montage (MRC) everywhere it is shown:
# the MRC panel, the overlay window and the stage picker.  Purely cosmetic -
# clicks are un-mirrored before any coordinate use, so landmark coords, the
# fitted transform, stage positions and all saved files stay in true
# orientation.  Flip Y is on by default (montage vs single image usually
# differ by a vertical/+Y flip); set either to match your acquired images.
MONTAGE_FLIP_X = False
MONTAGE_FLIP_Y = True


def _flip_for_display(arr):
    """Mirror an MRC-grid array for DISPLAY only (per MONTAGE_FLIP_X/Y).
    Works for 2-D grayscale and (H, W, 3/4) RGB(A) arrays."""
    if MONTAGE_FLIP_X:
        arr = np.fliplr(arr)
    if MONTAGE_FLIP_Y:
        arr = np.flipud(arr)
    return arr


def _unflip_x(x, w):
    """Map a clicked/display x back to the TRUE (un-flipped) x, mirroring about
    the image width w.  Its own inverse, so it also maps true -> display."""
    return (w - 1 - x) if MONTAGE_FLIP_X else x


def _unflip_y(y, h):
    """Map a clicked/display y back to the TRUE (un-flipped) y, mirroring about
    the image height h.  Its own inverse, so it also maps true -> display."""
    return (h - 1 - y) if MONTAGE_FLIP_Y else y


# ---------------------------------------------------------------------------
# Pan / Zoom
# ---------------------------------------------------------------------------

# Tk event.state bit for the Shift modifier.
_SHIFT_MASK = 0x0001

def _shift_held(mpl_event):
    """True when Shift was held during a matplotlib mouse event (TkAgg).

    Used so that a Shift + left-drag is treated as a pan gesture and does
    NOT also drop a landmark / pick point.
    """
    ge = getattr(mpl_event, "guiEvent", None)
    if ge is not None and hasattr(ge, "state"):
        try:
            return bool(int(ge.state) & _SHIFT_MASK)
        except (TypeError, ValueError):
            pass
    # Fallback: matplotlib's own modifier tracking.
    return getattr(mpl_event, "key", None) in ("shift", "Shift")


class PanZoomHandler:
    def __init__(self, ax, canvas):
        self.ax, self.canvas, self._pan = ax, canvas, None
        w = canvas.get_tk_widget()
        w.bind("<MouseWheel>",      self._on_wheel,       add="+")
        w.bind("<Button-4>",        self._on_scroll_up,   add="+")
        w.bind("<Button-5>",        self._on_scroll_down, add="+")
        # Middle-button drag to pan.
        w.bind("<Button-2>",        self._on_press,       add="+")
        w.bind("<B2-Motion>",       self._on_drag,        add="+")
        w.bind("<ButtonRelease-2>", self._on_release,     add="+")
        # Shift + left-button drag to pan (laptop / trackpad friendly).
        w.bind("<Shift-Button-1>",        self._on_press,   add="+")
        w.bind("<Shift-B1-Motion>",       self._on_drag,    add="+")
        w.bind("<Shift-ButtonRelease-1>", self._on_release, add="+")
        # Clear any in-progress pan on a plain left release too, so a gesture
        # where Shift is let go before the button does not leave a stale pan.
        w.bind("<ButtonRelease-1>",       self._on_release, add="+")

    def _tk2mpl(self, tx, ty):
        h = self.canvas.get_tk_widget().winfo_height()
        return float(tx), float(h - ty)

    def _cursor_in_ax(self, tx, ty):
        dx, dy = self._tk2mpl(tx, ty)
        bb = self.ax.get_window_extent()
        return bb.x0 <= dx <= bb.x1 and bb.y0 <= dy <= bb.y1

    def _zoom(self, tx, ty, factor):
        if not self._cursor_in_ax(tx, ty): return
        dx, dy = self._tk2mpl(tx, ty)
        cx, cy = self.ax.transData.inverted().transform((dx, dy))
        xl, xr = self.ax.get_xlim()
        yl, yr = self.ax.get_ylim()
        self.ax.set_xlim(cx+(xl-cx)*factor, cx+(xr-cx)*factor)
        self.ax.set_ylim(cy+(yl-cy)*factor, cy+(yr-cy)*factor)
        self.canvas.draw_idle()

    def _on_wheel(self, e):
        self._zoom(e.x, e.y, 1.0/ZOOM_FACTOR if e.delta>0 else ZOOM_FACTOR)
    def _on_scroll_up(self, e):   self._zoom(e.x, e.y, 1.0/ZOOM_FACTOR)
    def _on_scroll_down(self, e): self._zoom(e.x, e.y, ZOOM_FACTOR)

    def _on_press(self, e):
        if self._cursor_in_ax(e.x, e.y):
            self._pan = {"last_tx": e.x, "last_ty": e.y}

    def _on_drag(self, e):
        if self._pan is None: return
        dtx = e.x - self._pan["last_tx"]
        dty = e.y - self._pan["last_ty"]
        bb = self.ax.get_window_extent()
        w_ax, h_ax = bb.width, bb.height
        if w_ax < 1 or h_ax < 1: return
        xl, xr = self.ax.get_xlim()
        yl, yr = self.ax.get_ylim()
        self.ax.set_xlim(xl - dtx*(xr-xl)/w_ax, xr - dtx*(xr-xl)/w_ax)
        self.ax.set_ylim(yl + dty*(yl-yr)/h_ax, yr + dty*(yl-yr)/h_ax)
        self._pan["last_tx"] = e.x
        self._pan["last_ty"] = e.y
        self.canvas.draw_idle()

    def _on_release(self, e): self._pan = None


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def normalize_image(img):
    img = np.nan_to_num(img.astype(np.float32))
    lo, hi = img.min(), img.max()
    return (img-lo)/(hi-lo) if hi > lo else np.zeros_like(img)

def apply_bc(img, vmin, vmax):
    if vmax <= vmin: return np.zeros_like(img)
    return np.clip((img-vmin)/(vmax-vmin), 0.0, 1.0)

def colorize(img2d, color):
    h, w = img2d.shape
    rgba = np.zeros((h,w,4), dtype=np.float32)
    rgba[...,0]=img2d*color[0]; rgba[...,1]=img2d*color[1]
    rgba[...,2]=img2d*color[2]; rgba[...,3]=img2d
    return rgba

def composite_overlay(mrc_bc, channels_bc, alpha_mrc=0.6, alpha_ch=1.0):
    rgb = np.empty(mrc_bc.shape + (3,), dtype=np.float32)
    np.multiply(mrc_bc, alpha_mrc, out=rgb[..., 0])
    rgb[..., 1] = rgb[..., 0]
    rgb[..., 2] = rgb[..., 0]
    for idx, ch in enumerate(channels_bc):
        col = CHANNEL_COLORS[idx % len(CHANNEL_COLORS)]
        a   = ch if alpha_ch == 1.0 else (ch * alpha_ch).astype(np.float32)
        oma = 1.0 - a
        for ci, cv in enumerate(col):
            np.multiply(rgb[..., ci], oma, out=rgb[..., ci])
            if cv > 0:
                rgb[..., ci] += a * cv
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return rgb


def _fast_ds(arr, max_px=2048):
    h, w = arr.shape[:2]
    f = max(1, max(h, w) // max_px)
    if f == 1:
        return arr
    return arr[::f, ::f] if arr.ndim == 2 else arr[::f, ::f, :]


# ---------------------------------------------------------------------------
# MRC / OME-TIFF loaders
# ---------------------------------------------------------------------------

def load_mrc_single(path):
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        data  = mrc.data.copy()
        voxel = mrc.voxel_size
        info  = f"shape={data.shape}  voxel={voxel}"
    if data.ndim == 3:
        mid = data.shape[0]//2
        data = data[mid]
        info += f"  (z={mid})"
    elif data.ndim != 2:
        raise ValueError(f"Unsupported MRC ndim={data.ndim}")
    return normalize_image(data), info


def load_ometiff(path):
    with tifffile.TiffFile(path) as tf:
        data = tf.asarray()
        axes = tf.series[0].axes if tf.series else "YX"
        info = f"shape={data.shape}  axes={axes}"
    axes = axes.upper()
    for dim in list(axes):
        if dim not in "CZYX":
            idx = axes.index(dim)
            mid = data.shape[idx] // 2
            data = data.take(mid, axis=idx)
            axes = axes.replace(dim, "", 1)
    while data.ndim < 2:
        data = data[np.newaxis]; axes = "Y" + axes
    for d in ["C", "Z"]:
        if d not in axes:
            data = data[np.newaxis]; axes = d + axes
    order = [axes.index(d) for d in "CZYX"]
    data  = np.transpose(data, order)
    normed = np.zeros(data.shape, dtype=np.float32)
    for c in range(data.shape[0]):
        normed[c] = normalize_image(data[c])
    return normed, info


# ---------------------------------------------------------------------------
# mdoc parser
# ---------------------------------------------------------------------------

def _coerce(val_str):
    parts = val_str.split()
    if not parts: return val_str
    try:
        nums = [int(p) if re.fullmatch(r"-?\d+", p) else float(p) for p in parts]
        return nums[0] if len(nums)==1 else nums
    except ValueError:
        return val_str.strip()

def parse_mdoc(path):
    global_info, pieces, mont_sections = {}, [], []
    current, ctype = global_info, "global"
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            m = re.match(r"\[ZValue\s*=\s*(\d+)\]", line)
            if m:
                current = {"ZValue": int(m.group(1))}
                pieces.append(current); ctype = "zvalue"; continue
            m = re.match(r"\[MontSection\s*=\s*(\d+)\]", line)
            if m:
                current = {"MontSection": int(m.group(1))}
                mont_sections.append(current); ctype = "mont"; continue
            if line.startswith("["): ctype = "text"; continue
            if ctype == "text": continue
            if "=" in line:
                key, _, val = line.partition("=")
                current[key.strip()] = _coerce(val.strip())
    return global_info, pieces, mont_sections


# ---------------------------------------------------------------------------
# Montage assembler helpers
# ---------------------------------------------------------------------------

def cosine_weight_map(h, w, feather_px):
    feather_px = max(1, int(feather_px))
    def ramp(n):
        r = np.ones(n, dtype=np.float32)
        f = min(feather_px, n//2)
        if f > 0:
            t = np.linspace(0.0, np.pi/2, f, dtype=np.float32)
            r[:f] = np.sin(t); r[-f:] = np.sin(t)[::-1]
        return r
    return np.outer(ramp(h), ramp(w))


def assemble_one_montage(mrc_path, pieces, img_h, img_w,
                          mont_h, mont_w, feather_px, status_cb=None):
    z_indices = [p["ZValue"] for p in pieces]
    n_tiles   = len(z_indices)

    if status_cb:
        status_cb(f"Reading {n_tiles} tiles from MRC...")

    with mrcfile.open(mrc_path, mode="r", permissive=True) as mrc:
        data = mrc.data
        if data is None:
            raise ValueError("MRC contains no image data.")
        # A montage MRC is a 3-D stack with one frame per tile.  If we instead
        # get a single 2-D image, it is already an assembled/blended montage
        # (or a single-frame image): use it directly rather than trying to
        # slice tiles out of it, which raised
        # "too many indices for array: array is 1-dimensional".
        if data.ndim == 2:
            if status_cb:
                status_cb("MRC is a single 2-D image - using it directly "
                          "(no tile assembly).")
            return normalize_image(data.astype(np.float32))
        if data.ndim != 3:
            raise ValueError(
                f"Unsupported MRC shape {data.shape}: expected a 3-D tile "
                f"stack or a 2-D image.")
        n_frames = data.shape[0]
        tiles = {}
        for i, z in enumerate(z_indices):
            if z < n_frames:
                tiles[z] = data[z].astype(np.float64)
                if status_cb and i % 3 == 0:
                    status_cb(f"Loading tile {i+1}/{n_tiles}...")

    if status_cb: status_cb("Blending tiles...")

    wmap    = cosine_weight_map(img_h, img_w, feather_px).astype(np.float64)
    canvas  = np.zeros((mont_h, mont_w), dtype=np.float64)
    weights = np.zeros((mont_h, mont_w), dtype=np.float64)

    for piece in pieces:
        z_idx = piece["ZValue"]
        if z_idx not in tiles: continue

        coords = piece.get("AlignedPieceCoords",
                           piece.get("PieceCoordinates", [0,0,0]))
        cx = int(round(float(coords[0])))
        cy = int(round(float(coords[1])))

        tile   = tiles[z_idx]
        sy0 = max(0,-cy);    sx0 = max(0,-cx)
        sy1 = min(img_h, mont_h-cy); sx1 = min(img_w, mont_w-cx)
        if sy1 <= sy0 or sx1 <= sx0: continue

        dy0=cy+sy0; dx0=cx+sx0; dy1=dy0+(sy1-sy0); dx1=dx0+(sx1-sx0)
        tc = tile[sy0:sy1,sx0:sx1]
        wc = wmap[sy0:sy1,sx0:sx1]
        canvas [dy0:dy1,dx0:dx1] += tc*wc
        weights[dy0:dy1,dx0:dx1] += wc

    valid = weights > 0
    canvas[valid] /= weights[valid]
    return normalize_image(canvas.astype(np.float32))


# ---------------------------------------------------------------------------
# B&C controls
# ---------------------------------------------------------------------------

class BCControls(ttk.Frame):
    def __init__(self, parent, callback, **kw):
        super().__init__(parent, **kw)
        self._cb = callback
        self.vmin_var = tk.DoubleVar(value=0.0)
        self.vmax_var = tk.DoubleVar(value=1.0)
        self._lbl_min = self._lbl_max = None
        self._build()

    def _build(self):
        for i, (lbl, var, cb) in enumerate([
            ("Min", self.vmin_var, "_chg_min"),
            ("Max", self.vmax_var, "_chg_max"),
        ]):
            row = ttk.Frame(self); row.pack(fill="x", pady=1)
            ttk.Label(row, text=lbl, width=4, anchor="e").pack(side="left")
            ttk.Scale(row, from_=0.0, to=1.0, orient="horizontal",
                      variable=var, command=getattr(self,cb)).pack(
                          side="left", fill="x", expand=True)
            l = ttk.Label(row, width=5, text=f"{var.get():.2f}"); l.pack(side="left")
            setattr(self, "_lbl_min" if i==0 else "_lbl_max", l)

    def _chg_min(self, _=None):
        v = self.vmin_var.get()
        if v >= self.vmax_var.get(): v=max(0.0,self.vmax_var.get()-0.01); self.vmin_var.set(v)
        self._lbl_min.config(text=f"{v:.2f}"); self._cb()
    def _chg_max(self, _=None):
        v = self.vmax_var.get()
        if v <= self.vmin_var.get(): v=min(1.0,self.vmin_var.get()+0.01); self.vmax_var.set(v)
        self._lbl_max.config(text=f"{v:.2f}"); self._cb()

    @property
    def vmin(self): return self.vmin_var.get()
    @property
    def vmax(self): return self.vmax_var.get()
    def reset(self):
        self.vmin_var.set(0.0); self._lbl_min.config(text="0.00")
        self.vmax_var.set(1.0); self._lbl_max.config(text="1.00")


class ChannelBCPanel(ttk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._rows = []

    def build(self, n_channels, callback):
        for w in self.winfo_children(): w.destroy()
        self._rows = []
        for c in range(n_channels):
            name = CHANNEL_COLOR_NAMES[c % len(CHANNEL_COLOR_NAMES)]
            hex_ = CHANNEL_HEX[c % len(CHANNEL_HEX)]
            hdr  = ttk.Frame(self); hdr.pack(fill="x", pady=(4,0))
            tk.Label(hdr, text="  ", bg=hex_, width=2).pack(side="left", padx=(2,4))
            ttk.Label(hdr, text=f"Ch {c}  ({name})", style="Sm.TLabel").pack(side="left")
            bc = BCControls(self, callback=lambda c=c: callback(c))
            bc.pack(fill="x", padx=4)
            self._rows.append(bc)

    def bc(self, idx):
        return self._rows[idx] if 0 <= idx < len(self._rows) else None

    def reset_all(self):
        for bc in self._rows: bc.reset()

    @property
    def n_channels(self): return len(self._rows)


# ---------------------------------------------------------------------------
# Stage Position Picker Window
# ---------------------------------------------------------------------------

class StagePickerWindow(tk.Toplevel):
    """
    Secondary window that shows a composite overlay and lets the user
    left-click to record stage positions.

    The window is handed the individual grayscale LAYERS that make up the
    overlay - the TEM (MRC) reference plus every channel of the OME-TIFF
    z-stack (warped onto the MRC grid).  A checkbox per layer chooses which
    layers are shown: tick one for an individual view, tick several to
    composite them together.  Each layer additionally has its own
    brightness/contrast (min/max) control, and the picker recombines the
    visible layers live.  All layers share the same montage geometry, so the
    downsample factor, the pick coordinates and the pixel->stage fit are
    identical regardless of which layers are on or how B&C is set - toggling
    or adjusting them never moves a placed point or the current zoom/pan.

    The composite is shown with the same fixed display-only flip as the rest of
    the tool (MONTAGE_FLIP_X / MONTAGE_FLIP_Y).  The flip mirrors only the
    drawn pixels; each click is un-mirrored back to the true montage pixel
    before the pixel->stage conversion, so recorded stage positions and the
    FOV crops are identical with or without the flip.

    Stage position is computed by inverse-distance-weighted interpolation
    across the SerialEM tile centres:
        stage_x = sum(w_i * (tile_stage_x_i + (px - tile_cx_i) * pix_um))
        stage_y = sum(w_i * (tile_stage_y_i + (py - tile_cy_i) * pix_um))
    where w_i = 1 / distance(click, tile_centre_i).

    Coordinate convention: top = -Y, left = -X (matches SerialEM stage).
    """

    def __init__(self, parent, mrc_gray, channels_gray, channel_names,
                 pieces, pixel_spacing_um, tile_hw,
                 warp_slice=None, n_z=1, image_shift_um=(0.0, 0.0),
                 title="Stage Position Picker"):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=BG)

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w    = min(1150, int(screen_w * 0.92))
        win_h    = min(780,  int(screen_h * 0.88))
        self.geometry(f"{win_w}x{win_h}")
        self.minsize(700, 400)
        self.maxsize(screen_w, screen_h - 100)

        self._pieces   = pieces
        self._pix_um   = pixel_spacing_um
        self._tile_h, self._tile_w = tile_hw

        # A single image-shift correction (um) added to EVERY tile's
        # StagePosition.  Read it off the scope with ReportImageShift()[4],[5]
        # and enter it here (the picker is offline and cannot query SerialEM).
        self._img_shift = np.asarray(image_shift_um[:2], dtype=float)

        self._picks    = []
        self._pt_artists = []

        # Full-resolution grayscale (0..1) source layers, all on the montage grid:
        #   _mrc_full      -> the TEM reference (MRC)
        #   _chan_full[i]  -> channel i of the OME-TIFF z-stack, warped to grid
        # The picker renders at FULL RESOLUTION via viewport rendering: instead
        # of ever building a composite of the whole montage (gigabytes of RGB),
        # it composites only the CURRENTLY VISIBLE window, sampled to about
        # screen resolution.  Zoom in and the visible window is small, so it is
        # sampled at stride 1 - true full resolution under the cursor - while the
        # array handed to matplotlib stays ~screen-sized, so memory stays bounded
        # no matter how large the montage is.  Picks, the pixel->stage fit and
        # FOV crops are all in true full-res montage pixels.
        self._mrc_full   = mrc_gray
        self._chan_full  = list(channels_gray)
        self._chan_names = list(channel_names)

        # For per-point FOV crops: callback (c, z) -> warped 2-D slice on the
        # MRC grid, and the number of z-planes in the fluorescence stack.
        self._warp_slice = warp_slice
        self._n_z        = max(1, int(n_z))

        self._H, self._W  = self._mrc_full.shape[:2]
        # Max samples across the longer side of the view; the rendered window
        # never exceeds ~this in either dimension (bounds memory & redraw cost).
        self._view_target   = 2000
        self._render_pending = False

        # Fit a no-shear similarity map (pixel -> stage) from the tiles.
        self._fit      = self._fit_pixel_to_stage()
        if self._fit is not None:
            kind = "rotation+flip" if self._fit["reflect"] else "rotation"
            self._fit_desc = (f"map: similarity fit ({kind}), "
                              f"{self._fit['n']} tiles, rmse "
                              f"{self._fit['rmse']:.3f} um")
        else:
            self._fit_desc = "map: per-tile interpolation (no rotation fit)"
        print(f"[StagePicker] {self._fit_desc}")

        self._build_styles()
        self._build_ui()

    def _build_styles(self):
        s = ttk.Style(self)
        try:
            s.configure("Sm.TLabel", background=BG, foreground=FG,
                        font=("Segoe UI", 9))
        except Exception:
            pass

    def _build_ui(self):
        main = ttk.Frame(self, padding=6)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)

        fig = Figure(figsize=(6, 5), facecolor=BG)
        self._ax = fig.add_subplot(111)
        self._ax.set_facecolor(BG)
        for sp in self._ax.spines.values():
            sp.set_edgecolor(BG3)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02)

        self._canvas = FigureCanvasTkAgg(fig, master=main)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._im = self._ax.imshow(np.zeros((1, 1, 3), dtype=np.float32),
                                   origin="upper", aspect="equal",
                                   interpolation="nearest")
        # Data coords = full-res montage pixels (display frame).  Set the view to
        # the whole montage; the actual pixels shown are sampled per-view.
        self._ax.set_xlim(-0.5, self._W - 0.5)
        self._ax.set_ylim(self._H - 0.5, -0.5)   # origin upper
        self._ax.set_autoscale_on(False)
        self._ax.axis("off")
        PanZoomHandler(self._ax, self._canvas)
        self._canvas.mpl_connect("button_press_event", self._on_click)
        # Re-render the visible window whenever the view changes (zoom/pan).
        self._ax.callbacks.connect("xlim_changed",
                                   lambda _ax: self._schedule_render())
        self._ax.callbacks.connect("ylim_changed",
                                   lambda _ax: self._schedule_render())

        # Store figure reference for screenshot export
        self._fig = fig

        side = ttk.Frame(main, padding=(6, 4, 4, 4), width=270)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        side.rowconfigure(0, weight=0)   # header
        side.rowconfigure(1, weight=0)   # layers
        side.rowconfigure(2, weight=0)   # buttons
        side.rowconfigure(3, weight=1)   # table
        side.rowconfigure(4, weight=0)   # status

        hdr = ttk.Frame(side)
        hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(hdr, text="STAGE POSITIONS",
                  foreground=CYA,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(hdr,
                  text=f"spacing: {self._pix_um:.4f} um/px   "
                       f"(top=-Y, left=-X)",
                  style="Sm.TLabel", foreground=BG3).pack(anchor="w")

        # ---- Image-shift correction (um), added to every tile's StagePosition.
        # Read it from the scope with ReportImageShift()[4],[5] and type it here.
        is_row = ttk.Frame(hdr); is_row.pack(fill="x", pady=(3, 0))
        ttk.Label(is_row, text="Image shift (um)  X", style="Sm.TLabel").pack(
            side="left")
        self._isx_var = tk.StringVar(value=f"{self._img_shift[0]:.4f}")
        ttk.Entry(is_row, textvariable=self._isx_var, width=8).pack(
            side="left", padx=(2, 4))
        ttk.Label(is_row, text="Y", style="Sm.TLabel").pack(side="left")
        self._isy_var = tk.StringVar(value=f"{self._img_shift[1]:.4f}")
        ttk.Entry(is_row, textvariable=self._isy_var, width=8).pack(
            side="left", padx=(2, 4))
        ttk.Button(is_row, text="Apply", style="Sm.TButton",
                   command=self._apply_image_shift).pack(side="left")

        ttk.Separator(hdr, orient="horizontal").pack(fill="x", pady=(4, 2))

        # ---- Layers: checkboxes + per-layer brightness/contrast --------------
        layers_lf = ttk.LabelFrame(side, text="Layers  (check = show)",
                                   padding=(4, 2))
        layers_lf.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        self._build_layers_panel(layers_lf)

        btn = ttk.Frame(side)
        btn.grid(row=2, column=0, sticky="ew", pady=(2, 4))
        btn.columnconfigure(0, weight=1)
        btn.columnconfigure(1, weight=1)

        # FOV crop width (um).  When set, exporting also writes one registered
        # z-stack crop per picked point (TEM + all channels x all z).
        fov_row = ttk.Frame(btn)
        fov_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 1))
        ttk.Label(fov_row, text="FOV width (um):",
                  style="Sm.TLabel").pack(side="left")
        self._fov_var = tk.StringVar(value="")
        ttk.Entry(fov_row, textvariable=self._fov_var, width=8).pack(
            side="left", padx=(4, 0))
        hint = ("blank = positions only; set a width to also save one "
                "registered z-stack crop per point")
        if self._warp_slice is None:
            hint += "  (channel data unavailable - crops disabled)"
        ttk.Label(btn, text=hint, style="Sm.TLabel", foreground=BG3,
                  wraplength=255, justify="left").grid(
                      row=1, column=0, columnspan=2, sticky="ew", pady=(0, 3))

        ttk.Button(btn, text="Export to TXT  (+ FOV crops)",
                   style="Accent.TButton",
                   command=self._export).grid(
                       row=2, column=0, columnspan=2,
                       sticky="ew", pady=(0, 4))
        ttk.Button(btn, text="Remove last",
                   style="Danger.TButton",
                   command=self._remove_last).grid(
                       row=3, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(btn, text="Clear all",
                   style="Danger.TButton",
                   command=self._clear).grid(
                       row=3, column=1, sticky="ew", padx=(2, 0))

        tf = ttk.Frame(side)
        tf.grid(row=3, column=0, sticky="nsew", pady=(2, 0))
        tf.rowconfigure(0, weight=1)
        tf.columnconfigure(0, weight=1)

        cols   = ("#", "Stage X (um)", "Stage Y (um)", "Pix X", "Pix Y")
        widths = [26, 94, 94, 50, 50]
        self._tree = ttk.Treeview(tf, columns=cols, show="headings")
        for c, w in zip(cols, widths):
            self._tree.heading(c, text=c)
            self._tree.column(c, width=w, anchor="center")
        tv_vsb = ttk.Scrollbar(tf, orient="vertical",
                               command=self._tree.yview)
        self._tree.configure(yscrollcommand=tv_vsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        tv_vsb.grid(row=0, column=1, sticky="ns")

        self._status = tk.StringVar(
            value="Left-click image to pick.\n"
                  "Scroll=zoom   Middle / Shift+drag=pan")
        ttk.Label(side, textvariable=self._status,
                  wraplength=255, justify="left",
                  style="Sm.TLabel").grid(
                      row=4, column=0, sticky="ew", pady=(4, 0))

        # Initial render now that the layer controls exist.
        self._recompose()

    def _build_layers_panel(self, container):
        """Scrollable list of layers: TEM (MRC) + each z-stack channel, each
        with a show/hide checkbox and its own brightness/contrast control."""
        sc  = tk.Canvas(container, bg=BG, highlightthickness=0, height=190)
        vsb = ttk.Scrollbar(container, orient="vertical", command=sc.yview)
        sc.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        sc.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(sc)
        sc.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: sc.configure(scrollregion=sc.bbox("all")))

        # TEM (MRC) layer - shown as grayscale, so use a neutral swatch.
        self._tem_on = tk.BooleanVar(value=True)
        trow = ttk.Frame(inner); trow.pack(fill="x", pady=(2, 0))
        tk.Label(trow, text="  ", bg="#cccccc", width=2).pack(side="left",
                                                              padx=(2, 4))
        ttk.Checkbutton(trow, text="TEM (MRC)", variable=self._tem_on,
                        command=self._recompose).pack(side="left")
        self._tem_bc = BCControls(inner, callback=self._recompose)
        self._tem_bc.pack(fill="x", padx=4)

        # Fluorescence channels (all channels in the z-stack).
        self._chan_on = []
        self._chan_bc = []
        for i in range(len(self._chan_full)):
            name = (self._chan_names[i] if i < len(self._chan_names)
                    else CHANNEL_COLOR_NAMES[i % len(CHANNEL_COLOR_NAMES)])
            hex_ = CHANNEL_HEX[i % len(CHANNEL_HEX)]
            on   = tk.BooleanVar(value=True)
            row  = ttk.Frame(inner); row.pack(fill="x", pady=(4, 0))
            tk.Label(row, text="  ", bg=hex_, width=2).pack(side="left",
                                                            padx=(2, 4))
            ttk.Checkbutton(row, text=f"Ch {i}  ({name})", variable=on,
                            command=self._recompose).pack(side="left")
            bc = BCControls(inner, callback=self._recompose)
            bc.pack(fill="x", padx=4)
            self._chan_on.append(on)
            self._chan_bc.append(bc)

    def _view_window(self):
        """Visible window in display-frame full-res pixels, plus a sampling
        stride chosen so the rendered window is ~_view_target across.  Stride 1
        (full resolution) is reached automatically once you zoom in enough that
        the visible window is <= _view_target pixels."""
        xl = self._ax.get_xlim(); yl = self._ax.get_ylim()
        x0, x1 = sorted((float(xl[0]), float(xl[1])))
        y0, y1 = sorted((float(yl[0]), float(yl[1])))
        dc0 = int(np.clip(np.floor(x0), 0, self._W))
        dc1 = int(np.clip(np.ceil(x1) + 1, 0, self._W))
        dr0 = int(np.clip(np.floor(y0), 0, self._H))
        dr1 = int(np.clip(np.ceil(y1) + 1, 0, self._H))
        if dc1 <= dc0: dc0, dc1 = 0, self._W
        if dr1 <= dr0: dr0, dr1 = 0, self._H
        s = max(1, int(np.ceil(max(dc1 - dc0, dr1 - dr0) / self._view_target)))
        return dc0, dc1, dr0, dr1, s

    def _view_indices(self, dc0, dc1, dr0, dr1, s):
        """Display-frame sample positions (cols, rows) and the matching TRUE
        source indices (mirrored for the cosmetic flip).  Indexing a source
        layer with np.ix_(trows, tcols) yields the window already in display
        orientation at the chosen stride - no separate flip needed, and the
        sampling is exactly aligned to the display window origin."""
        cols = np.arange(dc0, dc1, s)
        rows = np.arange(dr0, dr1, s)
        tcols = (self._W - 1 - cols) if MONTAGE_FLIP_X else cols
        trows = (self._H - 1 - rows) if MONTAGE_FLIP_Y else rows
        return cols, rows, trows, tcols

    def _render_view(self):
        """Composite only the visible window (at the view's stride) from the
        checked layers and push it to the image, with an extent that places it
        at the correct full-res coordinates.  Markers are separate artists in
        data coordinates, so they stay put across renders."""
        self._render_pending = False
        dc0, dc1, dr0, dr1, s = self._view_window()
        cols, rows, trows, tcols = self._view_indices(dc0, dc1, dr0, dr1, s)
        if len(cols) == 0 or len(rows) == 0:
            return
        rr = np.ix_(trows, tcols)
        shape = (len(rows), len(cols))

        if self._tem_on.get():
            base = apply_bc(self._mrc_full[rr].astype(np.float32),
                            self._tem_bc.vmin, self._tem_bc.vmax) * 0.6
        else:
            base = np.zeros(shape, dtype=np.float32)
        rgb = np.empty(shape + (3,), dtype=np.float32)
        rgb[..., 0] = base; rgb[..., 1] = base; rgb[..., 2] = base

        for i, L in enumerate(self._chan_full):
            if not self._chan_on[i].get():
                continue
            a   = apply_bc(L[rr].astype(np.float32),
                           self._chan_bc[i].vmin, self._chan_bc[i].vmax)
            col = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            oma = 1.0 - a
            for k, cv in enumerate(col):
                rgb[..., k] *= oma
                if cv > 0:
                    rgb[..., k] += a * cv
        np.clip(rgb, 0.0, 1.0, out=rgb)

        # Extent in display-frame coords: sampled cols span [dc0, dc0+ncols*s).
        x_left  = dc0 - 0.5
        x_right = dc0 + len(cols) * s - 0.5
        y_top   = dr0 - 0.5
        y_bot   = dr0 + len(rows) * s - 0.5
        self._im.set_data(rgb)
        self._im.set_extent([x_left, x_right, y_bot, y_top])  # origin upper
        self._canvas.draw_idle()

    def _schedule_render(self):
        """Coalesce multiple view/B&C/layer changes into a single redraw."""
        if not self._render_pending:
            self._render_pending = True
            self.after_idle(self._render_view)

    def _recompose(self, *_):
        """Layer toggles and B&C changes request a re-render of the current
        view (full resolution within the visible window).  Zoom/pan also
        trigger this via the axes lim-changed callbacks."""
        self._schedule_render()

    def _active_layers_desc(self):
        parts = []
        if self._tem_on.get():
            parts.append("TEM")
        for i, on in enumerate(self._chan_on):
            if on.get():
                parts.append(f"Ch{i}")
        return "+".join(parts) if parts else "none"

    def _tile_origin(self, piece):
        c = piece.get("AlignedPieceCoords",
                      piece.get("PieceCoordinates", [0, 0, 0]))
        return float(c[0]), float(c[1])

    def _stage_anchor(self, piece):
        """This tile's stage position (um) with the single image-shift
        correction added.  Offline: the correction comes from self._img_shift
        (entered in the UI), NOT from a live ReportImageShift() call.  Returns
        None when the tile has no usable StagePosition."""
        sp = piece.get("StagePosition", None)
        if not isinstance(sp, (list, tuple)) or len(sp) < 2:
            return None
        return np.asarray(sp[:2], dtype=float) + self._img_shift

    def _apply_image_shift(self):
        """Read the image-shift correction (um) from the entries, add it to all
        tile stage positions (re-fit the pixel->stage map), and re-derive any
        already-placed picks.  Pixel coordinates are unchanged - only the stage
        values shift."""
        try:
            x = float(self._isx_var.get()); y = float(self._isy_var.get())
        except ValueError:
            self._status.set("Image shift: enter two numbers (um).")
            return
        self._img_shift = np.asarray([x, y], dtype=float)
        # Re-fit the pixel->stage map with the corrected anchors.
        self._fit = self._fit_pixel_to_stage()
        if self._fit is not None:
            kind = "rotation+flip" if self._fit["reflect"] else "rotation"
            self._fit_desc = (f"map: similarity fit ({kind}), "
                              f"{self._fit['n']} tiles, rmse "
                              f"{self._fit['rmse']:.3f} um")
        else:
            self._fit_desc = "map: per-tile interpolation (no rotation fit)"
        # Re-derive existing picks from their (unchanged) pixel coordinates.
        for pick in self._picks:
            pick["sx"], pick["sy"] = self._stage_from_pixel(pick["px"], pick["py"])
        self._refresh_tree()
        self._redraw_points()
        self._status.set(f"Image shift applied: ({x:+.4f}, {y:+.4f}) um added "
                         f"to all tiles.\n{self._fit_desc}")

    def _fit_pixel_to_stage(self):
        """Fit a no-shear similarity (rotation + uniform scale + optional flip
        + translation) mapping montage-pixel coords -> stage microns, using
        each tile's pixel centre and StagePosition (+ image-shift correction)
        as a correspondence.

        Returns a dict describing the transform, or None when it cannot/should
        not be fit (fewer than 2 tiles, or stage positions too clustered, e.g.
        a pure image-shift montage) - in which case the caller falls back to
        the per-tile interpolation.
        """
        pts_px, pts_st = [], []
        for p in self._pieces:
            sp = self._stage_anchor(p)
            if sp is None:
                continue
            tx, ty = self._tile_origin(p)
            pts_px.append((tx + self._tile_w / 2.0, ty + self._tile_h / 2.0))
            pts_st.append((float(sp[0]), float(sp[1])))

        if len(pts_px) < 2:
            return None

        P = np.asarray(pts_px, dtype=np.float64)
        S = np.asarray(pts_st, dtype=np.float64)
        # Degenerate if stage barely varies (image-shift montage): fall back.
        if float(S.std(axis=0).max()) < 1e-6:
            return None

        n   = len(P)
        px, py = P[:, 0], P[:, 1]
        sx, sy = S[:, 0], S[:, 1]
        ones, zeros = np.ones(n), np.zeros(n)

        best = None
        for reflect in (False, True):
            A = np.zeros((2 * n, 4))
            b = np.zeros(2 * n)
            if not reflect:
                # rotation:  sx = a*px - b*py + tx ;  sy = b*px + a*py + ty
                A[0::2] = np.column_stack([px, -py, ones, zeros])
                A[1::2] = np.column_stack([py,  px, zeros, ones])
            else:
                # flip:      sx = a*px + b*py + tx ;  sy = b*px - a*py + ty
                A[0::2] = np.column_stack([px,  py, ones, zeros])
                A[1::2] = np.column_stack([-py, px, zeros, ones])
            b[0::2] = sx
            b[1::2] = sy
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            rmse = float(np.sqrt(np.mean((A @ sol - b) ** 2)))
            cand = {"reflect": reflect, "a": float(sol[0]), "b": float(sol[1]),
                    "tx": float(sol[2]), "ty": float(sol[3]),
                    "rmse": rmse, "n": n}
            if best is None or rmse < best["rmse"]:
                best = cand
        return best

    def _apply_fit(self, px, py):
        f = self._fit
        a, b, tx, ty = f["a"], f["b"], f["tx"], f["ty"]
        if not f["reflect"]:
            return a * px - b * py + tx, b * px + a * py + ty
        return a * px + b * py + tx, b * px - a * py + ty

    def _stage_from_pixel(self, px, py):
        # Preferred: the fitted similarity map (handles rotation + flip + scale).
        if self._fit is not None:
            return self._apply_fit(px, py)

        # Fallback: per-tile inverse-distance interpolation (no rotation),
        # used only when the similarity fit was not possible.
        containing = []
        for p in self._pieces:
            tx, ty = self._tile_origin(p)
            if (tx <= px < tx + self._tile_w and
                    ty <= py < ty + self._tile_h):
                containing.append(p)
        candidates = containing if containing else self._pieces

        w_sum = sx_sum = sy_sum = 0.0
        for p in candidates:
            tx, ty = self._tile_origin(p)
            cx = tx + self._tile_w / 2.0
            cy = ty + self._tile_h / 2.0
            dist = max(0.01, ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5)
            w    = 1.0 / dist
            sp   = self._stage_anchor(p)
            if sp is None:
                sp = self._img_shift            # no StagePosition -> shift only
            sx = float(sp[0]) + (px - cx) * self._pix_um
            sy = float(sp[1]) + (py - cy) * self._pix_um
            w_sum  += w
            sx_sum += w * sx
            sy_sum += w * sy

        return sx_sum / w_sum, sy_sum / w_sum

    def _on_click(self, event):
        if event.button != 1 or event.inaxes is not self._ax:
            return
        if _shift_held(event):          # Shift+drag pans; don't record a pick
            return
        if event.xdata is None:
            return

        # Data coords are full-res montage pixels in the display (flipped) frame;
        # un-mirror to the TRUE montage pixel for the stage/crop math.  Markers
        # are drawn at the clicked display coords (dx/dy), correct on the flipped
        # view because the flip is fixed.
        dx, dy = event.xdata, event.ydata
        px = _unflip_x(dx, self._W)
        py = _unflip_y(dy, self._H)
        sx, sy = self._stage_from_pixel(px, py)

        self._picks.append({
            "px": px,  "py": py,
            "dx": dx,  "dy": dy,
            "sx": sx,  "sy": sy,
        })
        self._refresh_tree()
        self._redraw_points()
        self._status.set(
            f"Point #{len(self._picks)}\n"
            f"Stage X:  {sx:.3f} um\n"
            f"Stage Y:  {sy:.3f} um\n\n"
            f"Pixel:  ({px:.0f}, {py:.0f})")

    def _redraw_points(self):
        for a in self._pt_artists:
            try: a.remove()
            except Exception: pass
        self._pt_artists = []
        for i, pick in enumerate(self._picks):
            x, y = pick["dx"], pick["dy"]
            arm  = 14
            l1, = self._ax.plot([x-arm, x+arm], [y, y],
                                color=CYA, lw=1.2, zorder=5)
            l2, = self._ax.plot([x, x], [y-arm, y+arm],
                                color=CYA, lw=1.2, zorder=5)
            dot, = self._ax.plot(x, y, "o", color=CYA, markersize=6,
                                 markeredgecolor="white", markeredgewidth=0.8, zorder=6)
            txt  = self._ax.text(x+9, y-9, str(i+1), color=CYA,
                                 fontsize=8, fontweight="bold", zorder=7)
            self._pt_artists.extend([l1, l2, dot, txt])
        self._canvas.draw_idle()

    def _refresh_tree(self):
        self._tree.delete(*self._tree.get_children())
        for i, pick in enumerate(self._picks):
            self._tree.insert("", "end", values=(
                i + 1,
                f"{pick['sx']:.3f}",
                f"{pick['sy']:.3f}",
                f"{pick['px']:.0f}",
                f"{pick['py']:.0f}",
            ))
        if self._picks:
            self._tree.see(self._tree.get_children()[-1])

    def _remove_last(self):
        if self._picks:
            self._picks.pop()
            self._refresh_tree()
            self._redraw_points()
            self._status.set(
                f"Removed last point.\n{len(self._picks)} point(s) remaining.")

    def _clear(self):
        self._picks.clear()
        self._refresh_tree()
        self._redraw_points()
        self._status.set("All points cleared.")

    def _fov_width_px(self):
        """Parse the FOV width entry (um) into an even crop side in full-res
        pixels.  Returns an int, None when blank, or 'bad' when invalid."""
        s = self._fov_var.get().strip()
        if not s:
            return None
        try:
            fov_um = float(s)
        except ValueError:
            return "bad"
        if fov_um <= 0 or self._pix_um <= 0:
            return "bad"
        cw = int(round(fov_um / self._pix_um))
        cw = max(2, cw)
        if cw % 2:                      # keep it even for a symmetric crop
            cw += 1
        return cw

    @staticmethod
    def _crop_centered(full, px, py, cw):
        """Extract a cw x cw window centred on (px, py) in full-res pixels.
        Regions outside the image are zero-padded so every crop is exactly
        cw x cw."""
        H, W = full.shape
        half = cw // 2
        x0 = int(round(px)) - half
        y0 = int(round(py)) - half
        out = np.zeros((cw, cw), dtype=np.float32)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W, x0 + cw), min(H, y0 + cw)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = full[sy0:sy1, sx0:sx1]
        return out

    def _write_crops(self, root, cw):
        """Write one registered z-stack crop per picked point.

        Each point -> an ImageJ hyperstack (Z, C, cw, cw) float32, where
        C = TEM + every fluorescence channel and Z = number of z-planes (the
        single TEM plane is replicated across z).  All channels are included
        regardless of which layer checkboxes are ticked - those only affect the
        on-screen view.  Files are named <root>_<point#>.tif.  Returns the list
        of basenames written."""
        n_pts = len(self._picks)
        n_ch  = len(self._chan_full)
        C_out = 1 + n_ch
        Z     = self._n_z

        # One output buffer per point.  Buffers are small (crop-sized), so we
        # can warp each (channel, z) slice just once and crop every point from
        # it rather than re-warping per point.
        stacks = [np.zeros((Z, C_out, cw, cw), dtype=np.float32)
                  for _ in range(n_pts)]

        # TEM (channel 0), replicated across z.
        for pi, pick in enumerate(self._picks):
            tem = self._crop_centered(self._mrc_full, pick["px"], pick["py"], cw)
            for z in range(Z):
                stacks[pi][z, 0] = tem

        # Fluorescence channels, full z-extent.
        for c in range(n_ch):
            for z in range(Z):
                self._status.set(f"Cropping channel {c}  z {z + 1}/{Z} ...")
                self.update_idletasks()
                full = self._warp_slice(c, z)
                for pi, pick in enumerate(self._picks):
                    stacks[pi][z, c + 1] = self._crop_centered(
                        full, pick["px"], pick["py"], cw)

        res    = (1.0 / self._pix_um) if self._pix_um > 0 else 1.0
        labels = (["TEM"] + [f"Ch{c}" for c in range(n_ch)]) * Z
        written = []
        for pi in range(n_pts):
            out_path = f"{root}_{pi + 1}.tif"
            tifffile.imwrite(
                out_path, stacks[pi], imagej=True,
                resolution=(res, res),
                metadata={"axes": "ZCYX", "unit": "um", "Labels": labels})
            written.append(os.path.basename(out_path))
        return written

    def _export(self):
        if not self._picks:
            messagebox.showwarning("Nothing to export",
                                   "Pick at least one point first.",
                                   parent=self)
            return
        # Validate the FOV width before asking for a path, so a typo fails fast.
        fov = self._fov_width_px()
        if fov == "bad":
            messagebox.showwarning(
                "FOV width",
                "Enter a positive number for the FOV width (um), or leave it "
                "blank to export positions only.", parent=self)
            return

        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export stage positions",
            defaultextension=".txt",
            filetypes=[
                ("Tab-separated text", "*.txt"),
                ("CSV", "*.csv"),
                ("All files", "*"),
            ])
        if not path:
            return
        try:
            # -- write the stage positions txt --------------------------------
            with open(path, "w") as fh:
                # fh.write("# Stage positions exported from MRC Registration Tool\n")
                # fh.write(f"# pixel_spacing_um = {self._pix_um:.6f}\n")
                # fh.write(f"# layers_shown = {self._active_layers_desc()}\n")
                # fh.write("# point\tstage_x_um\tstage_y_um\tpixel_x\tpixel_y\n")
                for i, pick in enumerate(self._picks):
                    fh.write(
                        f"{i+1}\t"
                        f"{pick['sx']:.4f}\t"
                        f"{pick['sy']:.4f}\t"
                        f"{pick['px']:.1f}\t"
                        f"{pick['py']:.1f}\n"
                    )

            # -- save a screenshot of the picker canvas -----------------------
            # Derive the PNG path by replacing the txt/csv extension.
            root, _ = os.path.splitext(path)
            screenshot_path = root + "_screenshot.png"
            self._fig.savefig(
                screenshot_path,
                dpi=150,
                facecolor=self._fig.get_facecolor(),
                bbox_inches="tight",
            )

            # -- optional: per-point FOV crop z-stacks ------------------------
            crop_msg = ""
            if fov is not None:
                if self._warp_slice is None:
                    crop_msg = ("\n\nFOV crops skipped: no channel data is "
                                "available for this picker.")
                else:
                    written = self._write_crops(root, fov)
                    self._status.set(
                        f"Wrote {len(written)} FOV crop stack(s).")
                    shown = "\n".join(written[:8])
                    if len(written) > 8:
                        shown += f"\n... (+{len(written) - 8} more)"
                    crop_msg = (f"\n\n{len(written)} FOV crop z-stack(s) "
                                f"({fov}x{fov} px) saved:\n{shown}")

            messagebox.showinfo(
                "Exported",
                f"Saved {len(self._picks)} point(s) to:\n{path}\n\n"
                f"Screenshot saved to:\n{os.path.basename(screenshot_path)}"
                f"{crop_msg}",
                parent=self)

        except Exception as e:
            messagebox.showerror("Export error", str(e), parent=self)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class RegistrationApp(tk.Tk):

    def __init__(self, mrc_reader, site_id):
        super().__init__()
        self.mrc_reader = mrc_reader
        self.correlator = CLEMCorrelator(mrc_reader=self.mrc_reader)
        self.site_id = site_id
        self.title("MRC / OME-TIFF Registration Tool")
        self.configure(bg=BG)
        self.minsize(1150, 820)

        self.mrc_image       = None
        self.tiff_stack      = None
        self.warped_channels = []
        self.point_pairs     = []

        self.flip_x = tk.BooleanVar(value=False)
        self.flip_y = tk.BooleanVar(value=False)

        self.mrc_is_montage    = False
        self.mrc_file_path     = None
        self.mrc_section_map   = {}
        self.mrc_mont_canvas   = {}
        self.mrc_mont_info     = {}
        self.mrc_img_hw        = (4096, 4096)
        self.mrc_feather_px    = 410
        self.mrc_montage_cache = {}
        self.mrc_n_sections    = 0

        self._mrc_pt_artists  = []
        self._tiff_pt_artists = []

        self.mrc_current_pieces    = []
        self.mrc_pixel_spacing_um  = 1.0

        self._build_styles()
        self._build_ui()

        if self.site_id is not None:
            self._load_site_data()

    def _display_loaded_site_data(self, data):
        self._display_loaded_mrc_data(data)
        self._display_loaded_tiff_data(data)

    def _display_loaded_mrc_data(self, data):
        self.mrc_file_path = os.fspath(data["mrc_path"])
        self.mrc_section_map = data["section_pieces"]
        self.mrc_montage_cache = dict(data["montages"])
        self.mrc_img_hw = data["img_hw"]
        self.mrc_feather_px = data["feather_px"]
        self.mrc_pixel_spacing_um = data["pixel_spacing_um"]
        self.mrc_is_montage = True

        sections = sorted(self.mrc_section_map.keys())
        self.mrc_n_sections = len(sections)

        self.mrc_mont_canvas = {
            sec: (img.shape[1], img.shape[0])
            for sec, img in self.mrc_montage_cache.items()
        }

        self.mrc_mont_info = {
            sec: f"{len(self.mrc_section_map.get(sec, []))} tiles"
            for sec in sections
        }

        self.montage_spin.config(to=max(0, self.mrc_n_sections - 1))
        self.montage_var.set(0)
        self.montage_count_var.set(f"/ {self.mrc_n_sections} sections")

        self.bc_mrc.reset()
        self.mrc_info_var.set(os.path.basename(self.mrc_file_path))

        self._show_montage_nav()
        self._on_montage_changed()
    
    def _display_loaded_tiff_data(self, data):
        self.tiff_stack = data["tiff_stack"]
        self._tiff_img_dirty = True

        C, Z = self.tiff_stack.shape[:2]

        self.channel_spin.config(to=max(0, C - 1))
        self.z_spin.config(to=max(0, Z - 1))
        self.channel_var.set(0)
        self.z_var.set(0)

        self.flip_x.set(False)
        self.flip_y.set(False)

        self.bc_tiff_panel.build(C, self._on_bc_tiff)
        self.tiff_info_var.set(
            os.path.basename(os.fspath(data["ome_path"])) + "  " + data["tiff_info"]
        )

        self._draw_tiff()

    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame",            background=BG)
        s.configure("TLabel",            background=BG, foreground=FG,
                    font=("Segoe UI", 10))
        s.configure("Sm.TLabel",         background=BG, foreground=FG,
                    font=("Segoe UI", 9))
        s.configure("TButton",           babuckground=ACC, foreground=BG,
                    font=("Segoe UI", 10, "bold"), padding=5)
        s.map("TButton", background=[("active",CYA),("disabled",BG3)])
        s.configure("Accent.TButton",    background=ACC2, foreground=BG,
                    font=("Segoe UI", 11, "bold"), padding=7)
        s.configure("Danger.TButton",    background=RED, foreground=BG,
                    font=("Segoe UI", 10, "bold"), padding=5)
        s.configure("Mont.TButton",      background="#cba6f7", foreground=BG,
                    font=("Segoe UI", 10, "bold"), padding=5)
        s.configure("TLabelframe",       background=BG, relief="groove")
        s.configure("TLabelframe.Label", background=BG, foreground=CYA,
                    font=("Segoe UI", 10, "bold"))
        s.configure("TRadiobutton",      background=BG, foreground=FG,
                    font=("Segoe UI", 9))
        s.configure("TScale",            background=BG, troughcolor=BG2,
                    sliderlength=14)
        s.configure("Treeview",          background=BG2, foreground=FG,
                    fieldbackground=BG2, rowheight=22)
        s.configure("Treeview.Heading",  background=BG3, foreground=FG)
        s.map("Treeview", background=[("selected",ACC)])

    def _build_ui(self):
        PAD = 8

        top = ttk.Frame(self, padding=(PAD,PAD,PAD,0))
        top.pack(fill="x")

        ttk.Button(top, text="Load MRC",
                   command=self._load_mrc).pack(side="left", padx=3)
        ttk.Button(top, text="Load MRC + mdoc",
           style="Mont.TButton",
           command=self._load_mrc_mdoc).pack(side="left", padx=3)
        self.mrc_info_var = tk.StringVar(value="No MRC loaded")
        ttk.Label(top, textvariable=self.mrc_info_var,
                  width=40, style="Sm.TLabel").pack(side="left", padx=3)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Button(top, text="Load OME-TIFF",
                   command=self._load_tiff).pack(side="left", padx=3)
        self.tiff_info_var = tk.StringVar(value="No OME-TIFF loaded")
        ttk.Label(top, textvariable=self.tiff_info_var,
                  width=38, style="Sm.TLabel").pack(side="left", padx=3)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Label(top, text="Ch:", style="Sm.TLabel").pack(side="left")
        self.channel_var  = tk.IntVar(value=0)
        self.channel_spin = ttk.Spinbox(top, from_=0, to=0,
                                        textvariable=self.channel_var,
                                        width=4, command=self._refresh_tiff)
        self.channel_spin.pack(side="left", padx=2)

        ttk.Label(top, text="Z:", style="Sm.TLabel").pack(side="left")
        self.z_var  = tk.IntVar(value=0)
        self.z_spin = ttk.Spinbox(top, from_=0, to=0,
                                  textvariable=self.z_var,
                                  width=4, command=self._refresh_tiff)
        self.z_spin.pack(side="left", padx=2)

        ttk.Label(top,
                  text="  scroll=zoom   middle-drag / shift+drag=pan   left-click=landmark",
                  style="Sm.TLabel", foreground=BG3).pack(side="right", padx=6)

        panels = ttk.Frame(self, padding=PAD)
        panels.pack(fill="both", expand=True)
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)
        panels.rowconfigure(0, weight=1)

        (self.canvas_mrc,  self.ax_mrc,
         self.bc_mrc,
         self.mrc_nav_frame) = self._make_mrc_panel(panels)

        (self.canvas_tiff, self.ax_tiff,
         self.bc_tiff_panel) = self._make_tiff_panel(panels)

        bot = ttk.Frame(self, padding=(PAD,0,PAD,PAD))
        bot.pack(fill="x")

        pf = ttk.LabelFrame(bot, text="Landmark pairs", padding=4)
        pf.pack(side="left", fill="both", expand=True)
        cols = ("#","MRC (x, y)","TIFF (x, y)")
        self.tree = ttk.Treeview(pf, columns=cols, show="headings", height=2)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=130, anchor="center")
        sb = ttk.Scrollbar(pf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")

        bf = ttk.Frame(bot, padding=(10,0,0,0))
        bf.pack(side="left", fill="y")

        self.status_var = tk.StringVar(value="Load images, then click landmarks.")
        ttk.Label(bf, textvariable=self.status_var, wraplength=215,
                  justify="left", style="Sm.TLabel").pack(pady=(0,4), anchor="w")

        tf_row = ttk.Frame(bf)
        tf_row.pack(fill="x", pady=(0, 4))
        ttk.Label(tf_row, text="Transform:", style="Sm.TLabel").pack(side="left")
        self.transform_var = tk.StringVar(value="similarity")
        ttk.Combobox(tf_row, textvariable=self.transform_var, width=11,
                     state="readonly",
                     values=["euclidean","similarity","affine","projective"]
                     ).pack(side="left", padx=(4,0))

        bg = ttk.Frame(bf)
        bg.pack(fill="x")
        bg.columnconfigure(0, weight=1)
        bg.columnconfigure(1, weight=1)
        ttk.Button(bg, text="Remove last", style="Danger.TButton",
                   command=self._remove_last).grid(
                       row=0, column=0, sticky="ew", padx=(0,2), pady=2)
        ttk.Button(bg, text="Clear all", style="Danger.TButton",
                   command=self._clear_points).grid(
                       row=0, column=1, sticky="ew", padx=(2,0), pady=2)
        ttk.Button(bg, text="Apply Transform", style="Accent.TButton",
                   command=self._apply_transform).grid(
                       row=1, column=0, columnspan=2, sticky="ew", pady=(4,2))
        ttk.Button(bg, text="Show Overlay",
                   command=self._show_overlay).grid(
                       row=2, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(bg, text="Export Transform",
                   command=self._export_transform).grid(
                       row=3, column=0, sticky="ew", padx=(0,2), pady=2)
        ttk.Button(bg, text="Import Transform",
                   command=self._import_transform).grid(
                       row=3, column=1, sticky="ew", padx=(2,0), pady=2)

    def _make_mrc_panel(self, parent):
        lf = ttk.LabelFrame(parent, text="MRC  --  left-click to place landmark",
                             padding=4)
        lf.grid(row=0, column=0, sticky="nsew", padx=(0,4))
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)

        fig = Figure(figsize=(5,3.2), facecolor=BG)
        ax  = fig.add_subplot(111)
        self._style_ax(ax)
        fig.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.03)

        canvas = FigureCanvasTkAgg(fig, master=lf)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        nav = ttk.Frame(lf, padding=(2,2))
        ttk.Label(nav, text="Montage:", style="Sm.TLabel").pack(side="left")
        self.montage_var  = tk.IntVar(value=0)
        self.montage_spin = ttk.Spinbox(
            nav, from_=0, to=0, textvariable=self.montage_var,
            width=4, command=self._on_montage_changed)
        self.montage_spin.pack(side="left", padx=(2,0))
        self.montage_count_var = tk.StringVar(value="/ 0")
        ttk.Label(nav, textvariable=self.montage_count_var,
                  style="Sm.TLabel").pack(side="left", padx=(2,8))

        ttk.Button(nav, text="<", width=2,
                   command=self._montage_prev).pack(side="left", padx=1)
        ttk.Button(nav, text=">", width=2,
                   command=self._montage_next).pack(side="left", padx=1)
        ttk.Button(nav, text="Refine alignment",
                   command=self._refine_tile_alignment).pack(side="left", padx=(10,1))

        self.montage_info_var = tk.StringVar(value="")
        ttk.Label(nav, textvariable=self.montage_info_var,
                  style="Sm.TLabel", foreground=CYA).pack(side="left", padx=(8,0))

        bc_frame = ttk.LabelFrame(lf, text="Brightness / Contrast  (MRC)",
                                   padding=(4,2))
        bc_frame.grid(row=2, column=0, sticky="ew", pady=(4,0))
        bc = BCControls(bc_frame, callback=self._on_bc_mrc)
        bc.pack(fill="x")

        PanZoomHandler(ax, canvas)
        canvas.mpl_connect("button_press_event", self._on_click_mrc)
        return canvas, ax, bc, nav

    def _make_tiff_panel(self, parent):
        lf = ttk.LabelFrame(parent, text="OME-TIFF  --  left-click to place landmark",
                             padding=4)
        lf.grid(row=0, column=1, sticky="nsew", padx=(4,0))
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)

        fig = Figure(figsize=(5,3.2), facecolor=BG)
        ax  = fig.add_subplot(111)
        self._style_ax(ax)
        fig.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.03)

        canvas = FigureCanvasTkAgg(fig, master=lf)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        flip_frame = ttk.Frame(lf, padding=(4, 2))
        flip_frame.grid(row=1, column=0, sticky="ew")
        ttk.Label(flip_frame, text="Flip:", style="Sm.TLabel").pack(side="left", padx=(0,4))
        ttk.Checkbutton(flip_frame, text="X  (left <-> right)",
                        variable=self.flip_x,
                        command=self._refresh_tiff).pack(side="left", padx=4)
        ttk.Checkbutton(flip_frame, text="Y  (top <-> bottom)",
                        variable=self.flip_y,
                        command=self._refresh_tiff).pack(side="left", padx=4)

        outer = ttk.LabelFrame(lf, text="Brightness / Contrast  (per channel)",
                                padding=(4,2))
        outer.grid(row=2, column=0, sticky="ew", pady=(4,0))

        sc = tk.Canvas(outer, bg=BG, highlightthickness=0, height=80)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=sc.yview)
        sc.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y"); sc.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(sc); sc.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: sc.configure(scrollregion=sc.bbox("all")))

        bc_panel = ChannelBCPanel(inner)
        bc_panel.pack(fill="x")

        PanZoomHandler(ax, canvas)
        canvas.mpl_connect("button_press_event", self._on_click_tiff)
        return canvas, ax, bc_panel

    @staticmethod
    def _style_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=BG3)
        for sp in ax.spines.values(): sp.set_edgecolor(BG3)

    def _show_montage_nav(self):
        self.mrc_nav_frame.grid(row=1, column=0, sticky="ew", pady=(2,2))

    def _hide_montage_nav(self):
        self.mrc_nav_frame.grid_remove()

    def _montage_prev(self):
        idx = self.montage_var.get()
        if idx > 0:
            self.montage_var.set(idx-1)
            self._on_montage_changed()

    def _montage_next(self):
        idx = self.montage_var.get()
        if idx < self.mrc_n_sections - 1:
            self.montage_var.set(idx+1)
            self._on_montage_changed()

    def _on_bc_mrc(self):
        imgs = self.ax_mrc.get_images()
        if imgs: imgs[0].set_clim(self.bc_mrc.vmin, self.bc_mrc.vmax); self.canvas_mrc.draw_idle()

    def _on_bc_tiff(self, channel_idx):
        cur_c = min(self.channel_var.get(),
                    self.tiff_stack.shape[0]-1 if self.tiff_stack is not None else 0)
        if channel_idx != cur_c: return
        bc = self.bc_tiff_panel.bc(cur_c)
        if bc is None: return
        imgs = self.ax_tiff.get_images()
        if imgs: imgs[0].set_clim(bc.vmin, bc.vmax); self.canvas_tiff.draw_idle()

    def _load_mrc(self):
        path = filedialog.askopenfilename(
            title="Open MRC",
            filetypes=[("MRC", ("*.mrc", "*.rec", "*.mrcs", "*.map")),
                       ("All files", "*")])
        if not path: return
        try:
            img, info = load_mrc_single(path)
            self.mrc_image     = img
            self._mrc_img_dirty = True
            self.mrc_is_montage = False
            self.mrc_montage_cache.clear()
            self.mrc_current_pieces   = []
            self.mrc_pixel_spacing_um = 1.0
            self.bc_mrc.reset()
            self.mrc_info_var.set(os.path.basename(path)+"  "+info)
            self._hide_montage_nav()
            self._draw_mrc()
            self.status_var.set("MRC loaded.")
        except Exception as e:
            messagebox.showerror("MRC load error", str(e))

    def _load_site_data(self):
        data = self.mrc_reader.load_latest_from_site(self.site_id)
        self._display_loaded_site_data(data)

    def _resolve_mrc_path(self, mdoc_path, global_info):
        """Locate the montage MRC for an mdoc, robust to stale Windows paths.
        Tries ImageFile as-written, then its basename next to the mdoc, then
        the mdoc name minus '.mdoc', then asks the user."""
        mdoc_dir = os.path.dirname(mdoc_path)
        img_file = global_info.get("ImageFile", "")
        if isinstance(img_file, (list, tuple)):
            img_file = " ".join(str(x) for x in img_file)
        img_file = str(img_file).strip().strip('"')
        base = os.path.basename(img_file.replace("\\", "/")) if img_file else ""

        candidates = []
        if img_file:
            candidates.append(img_file)                  # path as written in the mdoc
            if base:
                candidates.append(os.path.join(mdoc_dir, base))  # basename, next to mdoc
        stem, ext = os.path.splitext(mdoc_path)
        if ext.lower() == ".mdoc":
            candidates.append(stem)                      # foo.mrc.mdoc -> foo.mrc
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        return filedialog.askopenfilename(
            title=f"Locate MRC for '{os.path.basename(mdoc_path)}'"
                  + (f"  (mdoc names: {base})" if base else ""),
            initialdir=mdoc_dir,
            filetypes=[("MRC", ("*.mrc", "*.rec", "*.mrcs", "*.map")),
                       ("All files", "*")])

    def _load_mrc_mdoc(self):
        path = filedialog.askopenfilename(
            title="Open MRC montage",
            filetypes=[("MRC", ("*.mrc", "*.rec", "*.mrcs", "*.map")), ("All files", "*"),],)
        if not path:
            return

        try:
            data = self.mrc_reader.load_mrc_montage_data(path)
            self._display_loaded_mrc_data(data)
            self.status_var.set("MRC montage loaded.")
        except Exception as e:
            messagebox.showerror("MRC montage load error", str(e))

    def _load_tiff(self):
        path = filedialog.askopenfilename(
            title="Open OME-TIFF",
            filetypes=[("TIFF", ("*.tif", "*.tiff")), ("All files", "*")])
        if not path:
            return

        try:
            data = self.mrc_reader.load_ome_tiff_data(path)
            self._display_loaded_tiff_data(data)
            self.status_var.set("OME-TIFF loaded.")
        except Exception as e:
            messagebox.showerror("TIFF load error", str(e))

    def _on_montage_changed(self):
        if not self.mrc_is_montage: return
        idx  = self.montage_var.get()
        secs = sorted(self.mrc_section_map.keys())
        if idx >= len(secs): idx = len(secs)-1; self.montage_var.set(idx)
        sec  = secs[idx]

        self.mrc_current_pieces = self.mrc_section_map.get(sec, [])

        self.montage_info_var.set(self.mrc_mont_info.get(sec,""))
        self.status_var.set(f"Assembling montage {idx+1}/{self.mrc_n_sections}...")
        self.update_idletasks()

        montage = self._get_montage(sec)
        self.mrc_image = montage
        self._mrc_img_dirty = True
        self._draw_mrc()
        self.status_var.set(
            f"Montage {idx+1}/{self.mrc_n_sections}  "
            f"{self.mrc_mont_info.get(sec,'')}")

    def _get_montage(self, sec_idx):
        if sec_idx not in self.mrc_montage_cache:
            pieces  = self.mrc_section_map[sec_idx]
            fms     = self.mrc_mont_canvas.get(sec_idx, None)
            if fms is None:
                img_h, img_w = self.mrc_img_hw
                max_x = max_y = 0
                for p in pieces:
                    c = p.get("AlignedPieceCoords", p.get("PieceCoordinates",[0,0,0]))
                    if isinstance(c,(list,tuple)):
                        max_x = max(max_x, int(c[0])+img_w)
                        max_y = max(max_y, int(c[1])+img_h)
                fms = (max_x, max_y)
            mont_w, mont_h = fms
            img_h, img_w   = self.mrc_img_hw

            def status_cb(msg):
                self.status_var.set(msg); self.update_idletasks()

            mont = assemble_one_montage(
                self.mrc_file_path, pieces,
                img_h, img_w, mont_h, mont_w,
                self.mrc_feather_px, status_cb=status_cb)
            self.mrc_montage_cache[sec_idx] = mont
        return self.mrc_montage_cache[sec_idx]

    def _refine_tile_alignment(self):
        try:
            from skimage.registration import phase_cross_correlation as _pcc
        except ImportError:
            try:
                from skimage.feature import register_translation as _pcc
            except ImportError:
                messagebox.showerror(
                    "scikit-image too old",
                    "phase_cross_correlation not found.\n"
                    "Update:  pip install -U scikit-image"); return

        if not self.mrc_is_montage:
            messagebox.showwarning("Montage mode required",
                                   "Load MRC + mdoc first."); return

        secs   = sorted(self.mrc_section_map.keys())
        idx    = min(self.montage_var.get(), len(secs) - 1)
        sec    = secs[idx]
        pieces = self.mrc_section_map[sec]
        n      = len(pieces)
        if n < 2:
            messagebox.showinfo("Nothing to refine",
                                "Only one tile in this section."); return

        img_h, img_w = self.mrc_img_hw
        overlap      = self.mrc_feather_px
        ps_x         = img_w - overlap
        ps_y         = img_h - overlap
        max_shift    = overlap

        def cur_pos(p):
            c = p.get("AlignedPieceCoords", p.get("PieceCoordinates", [0,0,0]))
            return float(c[0]), float(c[1])

        grid = {}
        for p in pieces:
            pc  = p.get("PieceCoordinates", [0,0,0])
            gx  = int(round(float(pc[0]) / ps_x)) if ps_x > 0 else 0
            gy  = int(round(float(pc[1]) / ps_y)) if ps_y > 0 else 0
            grid[(gx, gy)] = p

        self.status_var.set("Loading tiles for alignment refinement...")
        self.update_idletasks()
        with mrcfile.open(self.mrc_file_path, mode="r", permissive=True) as mrc:
            if mrc.data is None or mrc.data.ndim != 3:
                messagebox.showinfo(
                    "Nothing to refine",
                    "This MRC is a single 2-D image, not a stack of tiles, so "
                    "tile alignment cannot be refined."); return
            n_frames = mrc.data.shape[0]
            tiles = {p["ZValue"]: mrc.data[p["ZValue"]].astype(np.float32)
                     for p in pieces if p["ZValue"] < n_frames}

        z_list = [p["ZValue"] for p in pieces]
        z2idx  = {z: i for i, z in enumerate(z_list)}

        A_rows, b_rows = [], []
        n_pairs = 0

        for (gx, gy), p_ref in sorted(grid.items()):
            zr = p_ref["ZValue"]
            if zr not in tiles:
                continue

            for direction, (dgx, dgy) in [("R", (1, 0)), ("D", (0, 1))]:
                nbr = (gx + dgx, gy + dgy)
                if nbr not in grid:
                    continue
                p_mov = grid[nbr]
                zm    = p_mov["ZValue"]
                if zm not in tiles:
                    continue

                n_pairs += 1
                self.status_var.set(
                    f"Cross-correlating pair {n_pairs}  "
                    f"(z={zr} vs z={zm}, {direction})...")
                self.update_idletasks()

                if direction == "R":
                    ref_strip = tiles[zr][:, -overlap:]
                    mov_strip = tiles[zm][:,  :overlap]
                else:
                    ref_strip = tiles[zr][-overlap:, :]
                    mov_strip = tiles[zm][ :overlap, :]

                try:
                    def _norm(a):
                        a = a.astype(np.float32)
                        s = a.std()
                        return (a - a.mean()) / s if s > 0 else a - a.mean()
                    raw   = _pcc(_norm(ref_strip), _norm(mov_strip),
                                 upsample_factor=10,
                                 normalization='phase')
                    shift = raw[0] if isinstance(raw, tuple) else raw
                    dy, dx = float(shift[0]), float(shift[1])
                except Exception:
                    dy, dx = 0.0, 0.0

                if abs(dy) > max_shift: dy = 0.0
                if abs(dx) > max_shift: dx = 0.0

                i_r = z2idx[zr]
                i_m = z2idx[zm]

                def add(var_j, var_i, rhs):
                    row         = np.zeros(2 * n)
                    row[var_j]  =  1.0
                    row[var_i]  = -1.0
                    A_rows.append(row)
                    b_rows.append(rhs)

                if direction == "R":
                    add(2*i_m,   2*i_r,   ps_x + dx)
                    add(2*i_m+1, 2*i_r+1, dy)
                else:
                    add(2*i_m,   2*i_r,   dx)
                    add(2*i_m+1, 2*i_r+1, ps_y + dy)

        if not A_rows:
            messagebox.showwarning("No pairs",
                                   "No adjacent tile pairs could be measured."); return

        x0, y0    = cur_pos(pieces[0])
        anchor_w  = 1e4
        row = np.zeros(2 * n); row[0] = anchor_w
        A_rows.append(row); b_rows.append(x0 * anchor_w)
        row = np.zeros(2 * n); row[1] = anchor_w
        A_rows.append(row); b_rows.append(y0 * anchor_w)

        self.status_var.set("Solving global alignment..."); self.update_idletasks()
        A = np.array(A_rows, dtype=np.float64)
        b = np.array(b_rows, dtype=np.float64)
        result, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)

        for i, p in enumerate(pieces):
            old   = p.get("AlignedPieceCoords",
                          p.get("PieceCoordinates", [0, 0, 0]))
            z_val = old[2] if isinstance(old, (list,tuple)) and len(old) >= 3 else 0
            p["AlignedPieceCoords"] = [float(result[2*i]),
                                        float(result[2*i + 1]),
                                        z_val]

        self.mrc_montage_cache.pop(sec, None)
        self.mrc_current_pieces = self.mrc_section_map.get(sec, [])
        self.status_var.set("Re-assembling with refined positions...")
        self.update_idletasks()
        self._on_montage_changed()

        messagebox.showinfo(
            "Refinement complete",
            f"Refined {n} tile(s) from {n_pairs} cross-correlation pair(s).\n"
            f"System rank: {rank} / {2*n}\n\n"
            "Montage re-assembled with refined tile positions.\n"
            "You can run refinement again to iterate.")

    def _draw_mrc(self, keep_view=False):
        if self.mrc_image is None: return
        if self._mrc_img_dirty:
            xl, yl = self._save_view(self.ax_mrc) if keep_view else (None, None)
            self.ax_mrc.clear(); self._style_ax(self.ax_mrc)
            img  = self.mrc_image
            h, w = img.shape
            ds   = max(1, max(h, w) // 1024)
            disp = img[::ds, ::ds] if ds > 1 else img
            # DISPLAY-ONLY flip of the TEM montage (extent stays full-res, so
            # event.xdata/ydata remain true MRC pixels and are un-mirrored in
            # _on_click_mrc before being stored).
            disp = _flip_for_display(disp)
            self.ax_mrc.imshow(disp, cmap="gray", origin="upper", aspect="equal",
                               extent=[-0.5, w-0.5, h-0.5, -0.5],
                               vmin=self.bc_mrc.vmin, vmax=self.bc_mrc.vmax)
            if keep_view: self._restore_view(self.ax_mrc, xl, yl)
            self._mrc_img_dirty  = False
            self._mrc_pt_artists = []
        else:
            for a in self._mrc_pt_artists:
                try: a.remove()
                except Exception: pass
            self._mrc_pt_artists = []
        h, w = self.mrc_image.shape
        for i, pair in enumerate(self.point_pairs):
            if "mrc" in pair:
                x, y = pair["mrc"]
                # Landmarks are stored as TRUE coords; draw them at the mirrored
                # (displayed) position so they sit on the flipped image.
                dx, dy = _unflip_x(x, w), _unflip_y(y, h)
                ln, = self.ax_mrc.plot(dx, dy, "o", color=PT_MRC, markersize=8,
                                       markeredgecolor="white", markeredgewidth=0.8, zorder=5)
                tx  = self.ax_mrc.text(dx+6, dy-6, str(i+1), color=PT_MRC,
                                       fontsize=9, fontweight="bold", zorder=6)
                self._mrc_pt_artists.extend([ln, tx])
        self.canvas_mrc.draw_idle()

    def _draw_tiff(self, keep_view=False):
        if self.tiff_stack is None: return
        c    = min(self.channel_var.get(), self.tiff_stack.shape[0]-1)
        z    = min(self.z_var.get(),       self.tiff_stack.shape[1]-1)
        bc   = self.bc_tiff_panel.bc(c)
        vmin, vmax = (bc.vmin, bc.vmax) if bc else (0.0, 1.0)
        if self._tiff_img_dirty:
            xl, yl = self._save_view(self.ax_tiff) if keep_view else (None, None)
            self.ax_tiff.clear(); self._style_ax(self.ax_tiff)
            img  = self._get_tiff_slice(c, z)
            h, w = img.shape
            ds   = max(1, max(h, w) // 1024)
            disp = img[::ds, ::ds] if ds > 1 else img
            self.ax_tiff.imshow(disp, cmap="gray", origin="upper", aspect="equal",
                                extent=[-0.5, w-0.5, h-0.5, -0.5],
                                vmin=vmin, vmax=vmax)
            if keep_view: self._restore_view(self.ax_tiff, xl, yl)
            self._tiff_img_dirty  = False
            self._tiff_pt_artists = []
        else:
            for a in self._tiff_pt_artists:
                try: a.remove()
                except Exception: pass
            self._tiff_pt_artists = []
        for i, pair in enumerate(self.point_pairs):
            if "tiff" in pair:
                x, y = pair["tiff"]
                ln, = self.ax_tiff.plot(x, y, "o", color=PT_TIFF, markersize=8,
                                        markeredgecolor="white", markeredgewidth=0.8, zorder=5)
                tx  = self.ax_tiff.text(x+6, y-6, str(i+1), color=PT_TIFF,
                                        fontsize=9, fontweight="bold", zorder=6)
                self._tiff_pt_artists.extend([ln, tx])
        self.canvas_tiff.draw_idle()

    def _refresh_tiff(self): self._tiff_img_dirty = True; self._draw_tiff(keep_view=True)

    def _get_tiff_slice(self, c, z):
        img = self.tiff_stack[c, z]
        if self.flip_x.get(): img = np.fliplr(img)
        if self.flip_y.get(): img = np.flipud(img)
        return img

    @staticmethod
    def _save_view(ax):    return ax.get_xlim(), ax.get_ylim()
    @staticmethod
    def _restore_view(ax, xl, yl):
        if xl is not None: ax.set_xlim(xl); ax.set_ylim(yl)

    def _draw_points(self, ax, key, color):
        for i, pair in enumerate(self.point_pairs):
            if key in pair:
                x, y = pair[key]
                ax.plot(x,y,"o",color=color,markersize=8,
                        markeredgecolor="white",markeredgewidth=0.8,zorder=5)
                ax.text(x+6,y-6,str(i+1),color=color,
                        fontsize=9,fontweight="bold",zorder=6)

    def _on_click_mrc(self, event):
        if event.button!=1 or event.inaxes is not self.ax_mrc: return
        if _shift_held(event): return   # Shift+drag pans; don't place a point
        if self.mrc_image is None or event.xdata is None: return
        # The MRC display is mirrored (display-only); un-mirror the click back
        # to the true montage pixel before storing, so the landmark - and thus
        # the fitted transform - is in the true, un-flipped frame.
        h, w = self.mrc_image.shape
        x = _unflip_x(event.xdata, w)
        y = _unflip_y(event.ydata, h)
        for pair in self.point_pairs:
            if "mrc" not in pair:
                pair["mrc"] = (x,y); self._update_tree()
                self._draw_mrc(keep_view=True)
                self.status_var.set(
                    f"MRC pt -> pair #{self.point_pairs.index(pair)+1}.\n"
                    "Now click matching TIFF point."); return
        self.point_pairs.append({"mrc":(x,y)}); self._update_tree()
        self._draw_mrc(keep_view=True)
        self.status_var.set(
            f"MRC point #{len(self.point_pairs)} placed.\nNow click matching TIFF point.")

    def _on_click_tiff(self, event):
        if event.button!=1 or event.inaxes is not self.ax_tiff: return
        if _shift_held(event): return   # Shift+drag pans; don't place a point
        if self.tiff_stack is None or event.xdata is None: return
        x, y = event.xdata, event.ydata
        for pair in reversed(self.point_pairs):
            if "mrc" in pair and "tiff" not in pair:
                pair["tiff"] = (x,y); self._update_tree()
                self._draw_tiff(keep_view=True)
                n = sum(1 for p in self.point_pairs if "mrc" in p and "tiff" in p)
                self.status_var.set(
                    f"Pair complete!  {n} pair(s).\nAdd more or Apply Transformation."); return
        self.point_pairs.append({"tiff":(x,y)}); self._update_tree()
        self._draw_tiff(keep_view=True)
        self.status_var.set(
            f"TIFF point #{len(self.point_pairs)} placed.\nNow click matching MRC point.")

    def _update_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, pair in enumerate(self.point_pairs):
            m = "({:.1f}, {:.1f})".format(*pair["mrc"])  if "mrc"  in pair else "-"
            t = "({:.1f}, {:.1f})".format(*pair["tiff"]) if "tiff" in pair else "-"
            self.tree.insert("","end",values=(i+1,m,t))

    def _remove_last(self):
        if self.point_pairs:
            self.point_pairs.pop(); self._update_tree()
            self._draw_mrc(keep_view=True); self._draw_tiff(keep_view=True)

    def _clear_points(self):
        self.point_pairs.clear(); self.warped_channels.clear()
        self._update_tree()
        self._draw_mrc(keep_view=True); self._draw_tiff(keep_view=True)
        self.status_var.set("Points cleared.")

    def _apply_transform(self):
        if self.mrc_image is None or self.tiff_stack is None:
            messagebox.showwarning("Missing data","Load both images first."); return
        
        def status_cb(msg): self.status_var.set(msg); self.update_idletasks()

        ttype    = self.transform_var.get()

        try:
            result = self.correlator.apply_transform(point_pairs=self.point_pairs, transform_type=ttype,
            tiff_stack=self.tiff_stack, mrc_shape=self.mrc_image.shape, flip_x=bool(self.flip_x.get()), flip_y=bool(self.flip_y.get()),
            status_cb=status_cb,)
        except ValueError as e:
            messagebox.showwarning("Not enough points", str(e))
            return
        except Exception as e:
            messagebox.showerror("Transform failed", str(e))
            return

        self._last_tform = result["transform"]
        self.warped_channels = result["warped_channels"]

        C = len(self.warped_channels)
        mrc_h, mrc_w = self.mrc_image.shape
        fit_txt = result["fit_info"]["text"]
        n_pairs = result["n_pairs"]


        self.status_var.set("{ttype.capitalize()} applied -- {C} ch warped.\n"
                        f"{fit_txt}\nClick Show Overlay.")
        

        msg = (
                    f"{ttype.capitalize()} from {n_pairs} pairs.\n"
                    f"{C} ch warped to grid ({mrc_w} x {mrc_h}).\n\n"
                    f"{fit_txt}\n\n"
                    f"(scale ~ TIFF_um_per_px / MRC_um_per_px; MRC = {self.mrc_pixel_spacing_um:.4f} um/px. "
                    f"If scale looks wrong or RMSE is large, add more corner-spread pairs.)\n\n"
                    "Click Show Overlay."
                )
        
        if "scale_check" in result:
            scale_check = result["scale_check"]

            if scale_check["ok"]:
                msg += ("\n\nScale check passed:\n"
                    f"{scale_check['message']}")
            else:
                msg += ("\n\nWARNING: scale check failed:\n"
                    f"{scale_check['message']}\n"
                    "This may indicate mismatched landmarks or an incorrect pixel size.")
            
        messagebox.showinfo("Done", msg)

    def _warp_channels_with_tform(self, tform):
        """Warp every channel (max-projected over z) onto the MRC grid using
        the given TIFF->MRC transform, and remember the transform.  Shared by
        Apply Transform (fit from point pairs) and Import Transform (loaded from
        file).  Returns (n_channels, mrc_w, mrc_h).

        The z-max is accumulated one plane at a time (np.maximum) rather than
        building a list of all Z warped planes and calling np.max on it - on a
        large montage that list was Z x montage-size and could exhaust memory.
        The result is kept full-resolution (the picker and overlay save use it
        at full res); only the on-screen previews downsample it."""
        mrc_h, mrc_w = self.mrc_image.shape
        C, Z = self.tiff_stack.shape[:2]
        # Keep the transform so the stage picker can re-warp individual
        # (channel, z) slices on demand for the per-point FOV crops, without us
        # having to hold the whole per-z warped stack in memory.
        self._last_tform = tform
        self.warped_channels = []
        for c in range(C):
            acc = None
            for z in range(Z):
                self.status_var.set(f"Warping channel {c+1}/{C}, z {z+1}/{Z}...")
                self.update_idletasks()
                w_ = warp(self._get_tiff_slice(c, z), tform.inverse,
                          output_shape=(mrc_h, mrc_w), order=1,
                          preserve_range=True, mode="constant",
                          cval=0).astype(np.float32)
                if acc is None:
                    acc = w_
                else:
                    np.maximum(acc, w_, out=acc)
                    del w_
            self.warped_channels.append(acc)
        return C, mrc_w, mrc_h

    def _export_transform(self):
        """Save the current TIFF->MRC transform (3x3 homogeneous matrix) plus
        the transform type and flip state to a text file."""
        if getattr(self, "_last_tform", None) is None:
            messagebox.showwarning(
                "No transform",
                "Apply a transform (or import one) before exporting."); return
        path = filedialog.asksaveasfilename(
            title="Export transform",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All files", "*")])
        if not path: return
        try:
            M = np.asarray(self._last_tform.params, dtype=float)
            npairs = sum(1 for p in self.point_pairs
                         if "mrc" in p and "tiff" in p)
            with open(path, "w") as fh:
                fh.write("# MRC Registration Tool - transform export\n")
                fh.write("# maps TIFF (source) pixel coords -> "
                         "MRC (destination) pixel coords\n")
                fh.write("# apply as: warp(img, "
                         "ProjectiveTransform(matrix=M).inverse)\n")
                fh.write(f"# transform_type = {self.transform_var.get()}\n")
                fh.write(f"# flip_x = {bool(self.flip_x.get())}\n")
                fh.write(f"# flip_y = {bool(self.flip_y.get())}\n")
                if self.mrc_image is not None:
                    h, w = self.mrc_image.shape
                    fh.write(f"# mrc_shape_hw = {h},{w}\n")
                if self.tiff_stack is not None:
                    fh.write("# tiff_shape_czyx = "
                             f"{','.join(str(s) for s in self.tiff_stack.shape)}\n")
                fh.write(f"# pixel_spacing_um = {self.mrc_pixel_spacing_um:.6f}\n")
                fh.write(f"# n_pairs = {npairs}\n")
                fh.write("# matrix 3x3 row-major (homogeneous):\n")
                for row in M:
                    fh.write("\t".join(f"{v:.10g}" for v in row) + "\n")
            self.status_var.set(f"Transform exported: {os.path.basename(path)}")
            messagebox.showinfo("Exported", f"Transform saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _import_transform(self):
        """Load a TIFF->MRC transform from a text file (as written by Export
        Transform) and apply it to all channels.  Restores the flip state that
        was active when the transform was created, since landmarks are placed on
        the flipped display."""
        if self.mrc_image is None or self.tiff_stack is None:
            messagebox.showwarning(
                "Missing data",
                "Load both the MRC and the OME-TIFF before importing a "
                "transform, so the channels can be warped."); return
        path = filedialog.askopenfilename(
            title="Import transform",
            filetypes=[("Text", "*.txt"), ("All files", "*")])
        if not path: return
        try:
            ttype = None; fx = fy = None; rows = []
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    s = line.strip()
                    if not s:
                        continue
                    if s.startswith("#"):
                        body = s.lstrip("#").strip().lower()
                        if "=" not in body:
                            continue
                        key, _, val = body.partition("=")
                        key = key.strip(); val = val.strip()
                        if key == "transform_type":
                            ttype = val
                        elif key == "flip_x":
                            fx = val in ("1", "true", "yes")
                        elif key == "flip_y":
                            fy = val in ("1", "true", "yes")
                        continue
                    parts = s.replace(",", " ").split()
                    try:
                        nums = [float(p) for p in parts]
                    except ValueError:
                        continue
                    if len(nums) >= 3:
                        rows.append(nums[:3])
            if len(rows) < 3:
                messagebox.showerror(
                    "Import error",
                    "Could not find a 3x3 transform matrix in the file."); return
            M = np.array(rows[:3], dtype=float)
        except Exception as e:
            messagebox.showerror("Import error", str(e)); return

        from skimage.transform import ProjectiveTransform
        try:
            tform = ProjectiveTransform(matrix=M)
        except Exception as e:
            messagebox.showerror("Import error", f"Invalid matrix:\n{e}"); return

        # Restore the transform type label and the flip state, then re-warp.
        if ttype in ("euclidean", "similarity", "affine", "projective"):
            self.transform_var.set(ttype)
        if fx is not None:
            self.flip_x.set(fx)
        if fy is not None:
            self.flip_y.set(fy)
        self._refresh_tiff()

        C, mrc_w, mrc_h = self._warp_channels_with_tform(tform)
        flip_note = f"flip X={bool(self.flip_x.get())}, Y={bool(self.flip_y.get())}"
        self.status_var.set(
            f"Transform imported and applied -- {C} ch warped ({flip_note}).\n"
            "Click Show Overlay.")
        messagebox.showinfo(
            "Imported",
            f"Transform imported from:\n{os.path.basename(path)}\n\n"
            f"Applied to {C} channel(s), warped to grid ({mrc_w} x {mrc_h}).\n"
            f"Restored {flip_note}.\n\nClick Show Overlay.")

    def _warp_slice_to_grid(self, c, z):
        """Warp one (channel, z) TIFF slice onto the full-resolution MRC grid,
        using the same transform/flip as the displayed overlay.  Returns a 2-D
        float32 array the size of the MRC montage.  Used by the stage picker to
        build per-point FOV crops as registered z-stacks."""
        mrc_h, mrc_w = self.mrc_image.shape
        return warp(self._get_tiff_slice(c, z), self._last_tform.inverse,
                    output_shape=(mrc_h, mrc_w), order=1,
                    preserve_range=True, mode="constant", cval=0).astype(np.float32)

    def _show_overlay(self):
        if self.mrc_image is None:
            messagebox.showwarning("No MRC","Load MRC first."); return
        if not self.warped_channels:
            messagebox.showwarning("No data","Apply Transformation first."); return

        C = len(self.warped_channels)

        # DISPLAY PATH (cheap): downsample FIRST, then brightness/contrast and
        # composite on the small arrays.  Building full-res RGB composites here
        # for a large montage (e.g. 10340x10340 -> ~1.3 GB each) is what pinned
        # memory at 100%; the full-res versions are now built lazily, only when
        # the user clicks "Save all panels" (see save_overlay).  _fast_ds is
        # plain striding, so downsample-then-apply_bc == apply_bc-then-downsample.
        mrc_disp = apply_bc(_fast_ds(self.mrc_image),
                            self.bc_mrc.vmin, self.bc_mrc.vmax)
        chs_disp = []
        for idx, ch in enumerate(self.warped_channels):
            bc = self.bc_tiff_panel.bc(idx)
            vmin, vmax = (bc.vmin, bc.vmax) if bc else (0.0, 1.0)
            chs_disp.append(apply_bc(_fast_ds(ch), vmin, vmax))
        per_ch_imgs = [composite_overlay(mrc_disp, [ch]) for ch in chs_disp]
        full_img    = composite_overlay(mrc_disp, chs_disp)

        ncols = C + 2
        win   = tk.Toplevel(self)
        win.title("Overlay result"); win.configure(bg=BG); win.minsize(600,350)

        fig  = Figure(figsize=(3.6*ncols,4.4), facecolor=BG)
        axes = [fig.add_subplot(1,ncols,i+1) for i in range(ncols)]
        fig.subplots_adjust(left=0.01,right=0.99,top=0.90,bottom=0.02,wspace=0.04)

        def sa(ax, t):
            ax.set_facecolor(BG); ax.set_title(t,color=CYA,fontsize=8,pad=4); ax.axis("off")

        # All panels are MRC-grid space -> mirror each for display (same flip).
        axes[0].imshow(_flip_for_display(mrc_disp),cmap="gray",origin="upper",vmin=0,vmax=1)
        sa(axes[0],"MRC (reference)")
        for idx,img in enumerate(per_ch_imgs):
            name = CHANNEL_COLOR_NAMES[idx%len(CHANNEL_COLOR_NAMES)]
            axes[idx+1].imshow(_flip_for_display(img),origin="upper")
            sa(axes[idx+1],f"MRC + Ch {idx}  ({name})")
        axes[-1].imshow(_flip_for_display(full_img),origin="upper"); sa(axes[-1],"Full composite")

        canvas = FigureCanvasTkAgg(fig,master=win)
        canvas.get_tk_widget().pack(fill="both",expand=True)
        canvas.draw()
        for ax in axes: PanZoomHandler(ax,canvas)

        btn_frame = ttk.Frame(win, padding=(6, 0, 6, 6))
        btn_frame.pack(fill="x")

        has_stage = bool(self.mrc_current_pieces)

        def open_stage_picker():
            if not has_stage:
                messagebox.showwarning(
                    "No mdoc data",
                    "Load MRC + mdoc to enable stage position picking.\n"
                    "Stage positions require tile calibration data from the mdoc file.",
                    parent=win)
                return

            # Hand the picker the individual grayscale layers (TEM + every
            # warped channel) so it can show/hide them via checkboxes and apply
            # its own per-layer brightness/contrast.  All share the MRC grid, so
            # the picker downsamples once and keeps coordinates in full-res px.
            # Also hand it a callback to re-warp any (channel, z) slice onto the
            # grid plus the z-count, so it can write per-point FOV crops as
            # registered z-stacks (TEM + all channels x all z) to disk.
            channel_names = [CHANNEL_COLOR_NAMES[i % len(CHANNEL_COLOR_NAMES)]
                             for i in range(C)]
            n_z = self.tiff_stack.shape[1] if self.tiff_stack is not None else 1
            StagePickerWindow(
                win,
                mrc_gray         = self.mrc_image,
                channels_gray    = self.warped_channels,
                channel_names    = channel_names,
                pieces           = self.mrc_current_pieces,
                pixel_spacing_um = self.mrc_pixel_spacing_um,
                tile_hw          = self.mrc_img_hw,
                warp_slice       = self._warp_slice_to_grid,
                n_z              = n_z,
                image_shift_um   = (0.0, 0.0),   # enter ReportImageShift()[4],[5] in picker
                title            = "Stage Position Picker",
            )

        ttk.Button(
            btn_frame,
            text   = "Open Stage Picker",
            style  = "Mont.TButton",
            command= open_stage_picker,
        ).pack(side="left", padx=(0, 6), pady=4)

        if not has_stage:
            ttk.Label(
                btn_frame,
                text   = "(load MRC + mdoc to enable)",
                style  = "Sm.TLabel",
                foreground = BG3,
            ).pack(side="left", pady=4)

        def save_overlay():
            base = filedialog.asksaveasfilename(
                title="Choose base filename",
                defaultextension=".tif",
                filetypes=[("TIFF","*.tif"),("PNG","*.png"),("All files","*")])
            if not base: return
            root, ext = os.path.splitext(base)
            ext = ext.lower() if ext else ".tif"
            use_png = ext==".png"
            def to_u8(a):
                if a.ndim==2: a=np.stack([a]*3,axis=-1)
                return (np.clip(a,0,1)*255).astype(np.uint8)
            def to_u16(a):
                return (np.clip(a,0,1)*65535).astype(np.uint16)
            def write_panel(sfx, arr):
                out = root + sfx + ext
                if use_png:
                    from PIL import Image
                    Image.fromarray(to_u8(arr)).save(out)
                elif arr.ndim == 2:
                    tifffile.imwrite(out, to_u16(arr), photometric="minisblack")
                else:
                    tifffile.imwrite(out, to_u8(arr), photometric="rgb")
                return os.path.basename(out)

            # Full-resolution panels are built HERE, on demand, one at a time,
            # and freed immediately - so saving a big montage never holds more
            # than one full-res composite in memory at once, and opening the
            # overlay (display only) never builds them at all.  Saved data is in
            # TRUE orientation (no _flip_for_display).
            try:
                self.status_var.set("Saving overlay panels..."); self.update_idletasks()
                saved = []
                # 1) MRC reference (grayscale, full-res, brightness-corrected).
                mrc_bc = apply_bc(self.mrc_image, self.bc_mrc.vmin, self.bc_mrc.vmax)
                saved.append(write_panel("_01_mrc", mrc_bc))
                # 2) one MRC+channel composite per channel.
                for i, ch in enumerate(self.warped_channels):
                    bc = self.bc_tiff_panel.bc(i)
                    vmin, vmax = (bc.vmin, bc.vmax) if bc else (0.0, 1.0)
                    ch_bc = apply_bc(ch, vmin, vmax)
                    comp  = composite_overlay(mrc_bc, [ch_bc])
                    nm = CHANNEL_COLOR_NAMES[i % len(CHANNEL_COLOR_NAMES)]
                    saved.append(write_panel(f"_0{i+2}_mrc_ch{i}_{nm}", comp))
                    del ch_bc, comp
                # 3) full composite, built incrementally: start from the MRC
                # base and blend one brightness-corrected channel at a time, so
                # we never hold all channels at once (only the RGB accumulator
                # plus one channel).  This matches composite_overlay's result.
                full_composite = composite_overlay(mrc_bc, [])   # gray base -> RGB
                for i, ch in enumerate(self.warped_channels):
                    bc = self.bc_tiff_panel.bc(i)
                    vmin, vmax = (bc.vmin, bc.vmax) if bc else (0.0, 1.0)
                    a   = apply_bc(ch, vmin, vmax)
                    col = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
                    oma = 1.0 - a
                    for k, cv in enumerate(col):
                        full_composite[..., k] *= oma
                        if cv > 0:
                            full_composite[..., k] += a * cv
                    del a, oma
                np.clip(full_composite, 0.0, 1.0, out=full_composite)
                saved.append(write_panel(f"_{C+2:02d}_full_composite", full_composite))
                del full_composite, mrc_bc
                self.status_var.set(f"Saved {len(saved)} overlay panel(s).")
                messagebox.showinfo("Saved",
                    f"{len(saved)} file(s) in:\n{os.path.dirname(os.path.abspath(root))}\n\n"
                    +"\n".join(saved))
            except MemoryError:
                messagebox.showerror(
                    "Out of memory",
                    "Not enough memory to build the full-resolution panels for "
                    "this montage.  Try saving as PNG, or save fewer channels.")
            except Exception as e:
                messagebox.showerror("Save error",str(e))

        ttk.Button(
            btn_frame,
            text   = "Save all panels as individual files",
            command= save_overlay,
            style  = "Accent.TButton",
        ).pack(side="right", pady=4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RegistrationApp().mainloop()
