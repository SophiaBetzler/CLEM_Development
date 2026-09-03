"""
MRC / OME-TIFF Registration Tool  (simplified flip model)
=========================================================

Idea
----
The TIFF's orientation relative to the MRC is arbitrary, so rather than model a
mirror inside the transform we simply flip the TIFF *for display* (X and/or Y)
until it matches the always-Y-flipped MRC, then pick landmarks on that matched
view.  Because both images are then in the same handedness, the transform is a
plain proper fit -- no reflection matrix, no "case" to choose.  The chosen TIFF
orientation (flip_x, flip_y) is stored so a transform can be re-applied to a
different TIFF later.

Coordinate rule (main window + overlay + picker)
------------------------------------------------
* Everything is handled in the DISPLAY frame.  The MRC is shown Y-flipped
  (MONTAGE_FLIP_Y); the TIFF is shown flipped per the X and Y checkboxes.
* Landmark picks are stored exactly as clicked (display pixels) -- no
  un-flipping and no re-flipping when drawing.
* The fit maps display-TIFF -> display-MRC; warping flips the raw TIFF by the
  checkbox state and writes into the display-MRC frame.
* The overlay shows the display-MRC arrays directly.  The stage picker needs
  true montage pixels for its stage calibration, so it is handed true-frame
  arrays (a single un-flip at that boundary) and uses its own display<->true
  handling (CLEMPicker.add_pick_from_display + the montage flip flags).

Because picks are stored in the display frame, changing a TIFF flip after
placing points invalidates them -- so toggling X or Y clears the pairs.  Set
the orientation first, then pick.

Dependencies:  pip install mrcfile tifffile scikit-image matplotlib numpy
"""

import sys
import os
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from clem_correlation import CLEMCorrelator
from clem_dataclasses import SiteDataSummary, MRCSummary
from clem_mrc_mdoc_reader import MRCReader
from clem_tem_communication import TEMComm

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patheffects as pe

try:
    import mrcfile
except ImportError:
    sys.exit("Missing: mrcfile  ->  pip install mrcfile")
try:
    import tifffile
except ImportError:
    sys.exit("Missing: tifffile  ->  pip install tifffile")
try:
    from skimage.transform import warp, ProjectiveTransform
except ImportError:
    sys.exit("Missing: scikit-image  ->  pip install scikit-image")


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
BG   = "#1e1e2e"; BG2 = "#313244"; BG3 = "#45475a"; FG = "#cdd6f4"
ACC  = "#89b4fa"; ACC2 = "#a6e3a1"; RED = "#f38ba8"; CYA = "#89dceb"

CHANNEL_HEX    = ["#00ff00", "#ff00ff", "#00ffff", "#ffff00", "#ff8000", "#8000ff"]
CHANNEL_COLORS = [(0., 1., 0.), (1., 0., 1.), (0., 1., 1.),
                  (1., 1., 0.), (1., .5, 0.), (.5, 0., 1.)]
CHANNEL_COLOR_NAMES = ["green", "magenta", "cyan", "yellow", "orange", "purple"]

# ---------------------------------------------------------------------------
# Channel roles
# ---------------------------------------------------------------------------
# A stack's channels are assigned roles rather than colours-by-position, because
# not every dataset carries every channel. The default is positional
# (ch0 reflection, then red, green, blue) and can be overridden per channel from
# the dropdown in the Brightness/Contrast panel.
#
# "reflection" is a reference channel, not a fluorophore: it is deliberately
# EXCLUDED from the fluorescence composite and instead gets its own
# TEM + reflection overlay, drawn in green.
CHANNEL_ROLES = ("reflection", "red", "green", "blue", "off")

ROLE_RGB = {"reflection": (0., 1., 0.),      # green, in its own overlay only
            "red":        (1., 0., 0.),
            "green":      (0., 1., 0.),
            "blue":       (0., 0., 1.),
            "off":        None}

ROLE_HEX = {"reflection": "#00ff00", "red": "#ff0000", "green": "#00ff00",
            "blue": "#0000ff", "off": "#555555"}

DEFAULT_ROLE_ORDER = ("reflection", "red", "green", "blue")

# Roles that take part in the fluorescence composite.
COMPOSITE_ROLES = ("red", "green", "blue")


def default_role(idx):
    """Positional default for channel `idx`; anything beyond blue starts off."""
    return DEFAULT_ROLE_ORDER[idx] if idx < len(DEFAULT_ROLE_ORDER) else "off"


def role_rgb(role):
    return ROLE_RGB.get(role)


def is_composite_role(role):
    return role in COMPOSITE_ROLES


# Compression for saved overlay panels. "zlib" (deflate) is lossless and
# readable by ImageJ/Fiji, tifffile and bioformats without extra packages. Set
# to None to write uncompressed; "lzw" is an alternative, and "zstd" is faster
# and smaller but needs the imagecodecs package.
TIFF_COMPRESSION = "zlib"


def tiff_write(path, data, **kwargs):
    """tifffile.imwrite with compression, tolerating tifffile API differences.

    Newer tifffile takes compression=; older versions took compress=; and a
    codec may be unavailable, as is compression itself for ImageJ hyperstacks.
    Any of those falls back to uncompressed rather than losing the save.
    """
    if TIFF_COMPRESSION:
        for kw in ("compression", "compress"):
            try:
                return tifffile.imwrite(path, data, **{kw: TIFF_COMPRESSION}, **kwargs)
            except TypeError:
                continue            # this tifffile doesn't take that keyword
            except ValueError:
                break               # keyword is fine, codec/format is not
    return tifffile.imwrite(path, data, **kwargs)


def overlay_panel_plan(roles, n_channels):
    """Which overlay panels to build for a given set of channel roles.

    Returns a list of (title, colors, is_gray). `colors` is the per-channel
    colour list to hand to composite_overlay -- None entries drop that channel
    -- or None for the plain MRC panel.

    Copes with stacks that carry only some of the channels: a role that is not
    present simply produces no panel, so a 2-channel file yields a shorter
    figure rather than an error.
    """
    roles = list(roles)[:n_channels]
    roles += ["off"] * (n_channels - len(roles))

    plan = [("MRC (reference)", None, True)]

    # Reflection is a reference channel, not a fluorophore: its own overlay,
    # in green, and deliberately absent from the composite below.
    refl = [i for i, r in enumerate(roles) if r == "reflection"]
    if refl:
        plan.append(("TEM + reflection (green)",
                     [role_rgb("reflection") if i in refl else None
                      for i in range(n_channels)],
                     False))

    comp = [role_rgb(r) if is_composite_role(r) else None for r in roles]
    for i, col in enumerate(comp):
        if col is None:
            continue
        plan.append((f"MRC + Ch {i} ({roles[i]})",
                     [col if k == i else None for k in range(n_channels)],
                     False))

    # With a single fluorescence channel the composite would just repeat the
    # per-channel panel, so only add it when there is something to combine.
    if sum(c is not None for c in comp) > 1:
        plan.append(("Full composite", comp, False))
    return plan


def _panel_slug(title):
    """Filename-safe suffix for a panel title."""
    out = title.lower()
    for a, b in ((" + ", "_"), (" ", "_"), ("(", ""), (")", "")):
        out = out.replace(a, b)
    return out

PT_MRC = "#FF4444"; PT_TIFF = "#44AAFF"
ZOOM_FACTOR = 1.25

# The montage display-flip convention (and the flip function itself) live on
# MRCReader; the UI sources them from there rather than re-declaring them.

# TIFF display flip defaults (the checkboxes).  Y on, X off, by request.
TIFF_FLIP_X_DEFAULT = False
TIFF_FLIP_Y_DEFAULT = True


def _mrc_to_display(arr):
    """True montage array -> displayed array (2-D or (H,W,3/4)), using the
    reader's flip convention."""
    return MRCReader._flip_for_display(arr, MRCReader.MONTAGE_FLIP_X, MRCReader.MONTAGE_FLIP_Y)

def _mrc_display_to_true_2d(arr):
    """Displayed montage array -> true-frame array (self-inverse)."""
    return MRCReader._flip_for_display(arr, MRCReader.MONTAGE_FLIP_X, MRCReader.MONTAGE_FLIP_Y)



# ---------------------------------------------------------------------------
# Pan / Zoom
# ---------------------------------------------------------------------------

_SHIFT_MASK = 0x0001

def _shift_held(mpl_event):
    ge = getattr(mpl_event, "guiEvent", None)
    if ge is not None and hasattr(ge, "state"):
        try:
            return bool(int(ge.state) & _SHIFT_MASK)
        except (TypeError, ValueError):
            pass
    return getattr(mpl_event, "key", None) in ("shift", "Shift")


class PanZoomHandler:
    def __init__(self, ax, canvas):
        self.ax, self.canvas, self._pan = ax, canvas, None
        w = canvas.get_tk_widget()
        w.bind("<MouseWheel>",      self._wheel,   add="+")
        w.bind("<Button-4>",        self._sc_up,   add="+")
        w.bind("<Button-5>",        self._sc_down, add="+")
        w.bind("<Button-2>",        self._press,   add="+")
        w.bind("<B2-Motion>",       self._drag,    add="+")
        w.bind("<ButtonRelease-2>", self._release, add="+")
        w.bind("<Shift-Button-1>",        self._press,   add="+")
        w.bind("<Shift-B1-Motion>",       self._drag,    add="+")
        w.bind("<Shift-ButtonRelease-1>", self._release, add="+")

    def _tk2mpl(self, tx, ty):
        h = self.canvas.get_tk_widget().winfo_height()
        return float(tx), float(h - ty)

    def _in_ax(self, tx, ty):
        dx, dy = self._tk2mpl(tx, ty)
        bb = self.ax.get_window_extent()
        return bb.x0 <= dx <= bb.x1 and bb.y0 <= dy <= bb.y1

    def _zoom(self, tx, ty, f):
        if not self._in_ax(tx, ty):
            return
        dx, dy = self._tk2mpl(tx, ty)
        cx, cy = self.ax.transData.inverted().transform((dx, dy))
        xl, xr = self.ax.get_xlim(); yl, yr = self.ax.get_ylim()
        self.ax.set_xlim(cx + (xl - cx) * f, cx + (xr - cx) * f)
        self.ax.set_ylim(cy + (yl - cy) * f, cy + (yr - cy) * f)
        self.canvas.draw_idle()

    def _wheel(self, e):   self._zoom(e.x, e.y, 1 / ZOOM_FACTOR if e.delta > 0 else ZOOM_FACTOR)
    def _sc_up(self, e):   self._zoom(e.x, e.y, 1 / ZOOM_FACTOR)
    def _sc_down(self, e): self._zoom(e.x, e.y, ZOOM_FACTOR)

    def _press(self, e):
        if self._in_ax(e.x, e.y):
            self._pan = (e.x, e.y)

    def _drag(self, e):
        if self._pan is None:
            return
        dtx, dty = e.x - self._pan[0], e.y - self._pan[1]
        bb = self.ax.get_window_extent()
        if bb.width < 1 or bb.height < 1:
            return
        xl, xr = self.ax.get_xlim(); yl, yr = self.ax.get_ylim()
        self.ax.set_xlim(xl - dtx * (xr - xl) / bb.width, xr - dtx * (xr - xl) / bb.width)
        self.ax.set_ylim(yl + dty * (yl - yr) / bb.height, yr + dty * (yl - yr) / bb.height)
        self._pan = (e.x, e.y)
        self.canvas.draw_idle()

    def _release(self, e):
        self._pan = None


# ---------------------------------------------------------------------------
# Display-only rendering helpers (brightness/contrast, overlay compositing,
# and view downsampling).  Kept in the UI because they exist purely to paint
# pixels on screen and depend on UI state (BC controls, CHANNEL_COLORS).
# Image normalization and MRC reading live on MRCReader.
# ---------------------------------------------------------------------------

def apply_bc(img, vmin, vmax):
    if vmax <= vmin:
        return np.zeros_like(img)
    return np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)

def composite_overlay(mrc_bc, channels_bc, alpha_mrc=0.6, colors=None):
    """Composite channels over the MRC.

    `colors` is an optional per-channel list of RGB tuples; a None entry means
    "leave this channel out" (role 'off', or reflection when building the
    fluorescence composite). Without it, the legacy positional palette is used.
    """
    rgb = np.empty(mrc_bc.shape + (3,), dtype=np.float32)
    np.multiply(mrc_bc, alpha_mrc, out=rgb[..., 0])
    rgb[..., 1] = rgb[..., 0]; rgb[..., 2] = rgb[..., 0]
    for idx, ch in enumerate(channels_bc):
        if colors is not None:
            col = colors[idx] if idx < len(colors) else None
        else:
            col = CHANNEL_COLORS[idx % len(CHANNEL_COLORS)]
        if col is None:
            continue
        a = ch; oma = 1.0 - a
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
# Brightness / contrast controls
# ---------------------------------------------------------------------------

class BCControls(ttk.Frame):
    def __init__(self, parent, callback, **kw):
        super().__init__(parent, **kw)
        self._cb = callback
        self.vmin_var = tk.DoubleVar(value=0.0)
        self.vmax_var = tk.DoubleVar(value=1.0)
        self._build()

    def _build(self):
        for i, (lbl, var, cb) in enumerate([("Min", self.vmin_var, "_chg_min"),
                                            ("Max", self.vmax_var, "_chg_max")]):
            row = ttk.Frame(self); row.pack(fill="x", pady=1)
            ttk.Label(row, text=lbl, width=4, anchor="e").pack(side="left")
            ttk.Scale(row, from_=0.0, to=1.0, orient="horizontal",
                      variable=var, command=getattr(self, cb)).pack(side="left", fill="x", expand=True)
            l = ttk.Label(row, width=5, text=f"{var.get():.2f}"); l.pack(side="left")
            setattr(self, "_lbl_min" if i == 0 else "_lbl_max", l)

    def _chg_min(self, _=None):
        v = self.vmin_var.get()
        if v >= self.vmax_var.get():
            v = max(0.0, self.vmax_var.get() - 0.01); self.vmin_var.set(v)
        self._lbl_min.config(text=f"{v:.2f}"); self._cb()

    def _chg_max(self, _=None):
        v = self.vmax_var.get()
        if v <= self.vmin_var.get():
            v = min(1.0, self.vmin_var.get() + 0.01); self.vmax_var.set(v)
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
        self._role_vars = []
        self._swatches = []

    def build(self, n_channels, callback, roles=None, on_role_change=None):
        for w in self.winfo_children():
            w.destroy()
        self._rows = []
        self._role_vars = []
        self._swatches = []
        for c in range(n_channels):
            role = roles[c] if (roles and c < len(roles)) else default_role(c)
            var = tk.StringVar(value=role)
            hdr = ttk.Frame(self); hdr.pack(fill="x", pady=(4, 0))
            sw = tk.Label(hdr, text="  ", bg=ROLE_HEX.get(role, "#555555"), width=2)
            sw.pack(side="left", padx=(2, 4))
            ttk.Label(hdr, text=f"Ch {c}", style="Sm.TLabel").pack(side="left")
            combo = ttk.Combobox(hdr, textvariable=var, values=list(CHANNEL_ROLES),
                                 width=10, state="readonly")
            combo.pack(side="left", padx=(6, 0))

            def _role_changed(_evt=None, c=c):
                self._swatches[c].configure(
                    bg=ROLE_HEX.get(self._role_vars[c].get(), "#555555"))
                if on_role_change is not None:
                    on_role_change(c)

            combo.bind("<<ComboboxSelected>>", _role_changed)
            bc = BCControls(self, callback=lambda c=c: callback(c))
            bc.pack(fill="x", padx=4)
            self._rows.append(bc)
            self._role_vars.append(var)
            self._swatches.append(sw)

    def bc(self, idx):
        return self._rows[idx] if 0 <= idx < len(self._rows) else None

    def role(self, idx):
        return self._role_vars[idx].get() if 0 <= idx < len(self._role_vars) else "off"

    def roles(self):
        return [v.get() for v in self._role_vars]

    def colors(self):
        """Per-channel RGB, None where the role contributes nothing."""
        return [role_rgb(v.get()) for v in self._role_vars]

    def composite_colors(self):
        """Per-channel RGB for the fluorescence composite; reflection excluded."""
        return [role_rgb(v.get()) if is_composite_role(v.get()) else None
                for v in self._role_vars]

    def indices_with_role(self, role):
        return [i for i, v in enumerate(self._role_vars) if v.get() == role]

    @property
    def n_channels(self): return len(self._rows)


# ---------------------------------------------------------------------------
# Stage Position Picker  (fed TRUE-frame arrays; flips for display itself)
# ---------------------------------------------------------------------------

class StagePickerWindow(tk.Toplevel):
    """Composite overlay + click-to-record stage positions.

    Internally the picker works in TRUE montage pixels (its stage calibration
    requires them).  It is handed true-frame arrays and displays them flipped
    (MONTAGE_FLIP_*); each click is un-flipped by CLEMPicker.add_pick_from_display
    using the montage flip flags, so stage positions and FOV crops are correct.
    """

    def __init__(self, parent, clem_picker, mrc_reader, mrc_gray_true, channels_gray_true,
                 channel_names, warp_slice_true=None, n_z=1, site_data=None,
                 title="Stage Position Picker", channel_roles=None,
                 warp_crop_true=None):
        super().__init__(parent)
        self.title(title); self.configure(bg=BG)
        self.clem_picker = clem_picker
        self.mrc_reader = mrc_reader
        self.site_data = site_data
        self.site_id = site_data.site_id if site_data is not None else None
        self._pix_um = clem_picker.pixel_spacing_um

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1150, int(sw*0.92))}x{min(780, int(sh*0.88))}")
        self.minsize(700, 400)

        self._picks, self._pt_artists = [], []
        self._mrc_true = mrc_gray_true              # TRUE montage
        self._chan_true = list(channels_gray_true)  # TRUE-frame warped channels
        self._chan_names = list(channel_names)
        # Roles drive the colours here too, so the picker matches the overlay.
        # Falls back to positional defaults when the caller doesn't pass any.
        self._chan_roles = (list(channel_roles) if channel_roles is not None
                            else [default_role(i) for i in range(len(self._chan_true))])
        self._warp_slice_true = warp_slice_true     # (c,z) -> TRUE-frame full-res warp
        # (c,z,x0,y0,cw) -> TRUE-frame crop; avoids warping the whole montage
        # per channel per z when only small windows are needed.
        self._warp_crop_true = warp_crop_true
        self._n_z = max(1, int(n_z))
        self._H, self._W = self._mrc_true.shape[:2]
        self._view_target = 2000
        self._render_pending = False

        s = ttk.Style(self)
        try:
            s.configure("Sm.TLabel", background=BG, foreground=FG, font=("Segoe UI", 9))
        except Exception:
            pass
        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self, padding=6); main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1); main.rowconfigure(0, weight=1)

        fig = Figure(figsize=(6, 5), facecolor=BG)
        self._ax = fig.add_subplot(111); self._ax.set_facecolor(BG)
        for sp in self._ax.spines.values():
            sp.set_edgecolor(BG3)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02)
        self._fig = fig
        self._canvas = FigureCanvasTkAgg(fig, master=main)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._im = self._ax.imshow(np.zeros((1, 1, 3), np.float32),
                                   origin="upper", aspect="equal", interpolation="nearest")
        self._ax.set_xlim(-0.5, self._W - 0.5)
        self._ax.set_ylim(self._H - 0.5, -0.5)
        self._ax.set_autoscale_on(False); self._ax.axis("off")
        PanZoomHandler(self._ax, self._canvas)
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._ax.callbacks.connect("xlim_changed", lambda _a: self._schedule_render())
        self._ax.callbacks.connect("ylim_changed", lambda _a: self._schedule_render())

        side = ttk.Frame(main, padding=(6, 4, 4, 4), width=270)
        side.grid(row=0, column=1, sticky="nsew")
        for r in range(5):
            side.rowconfigure(r, weight=1 if r == 3 else 0)
        side.columnconfigure(0, weight=1)

        hdr = ttk.Frame(side); hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(hdr, text="STAGE POSITIONS", foreground=CYA,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(hdr, text=f"spacing: {self._pix_um:.4f} um/px",
                  style="Sm.TLabel", foreground=BG3).pack(anchor="w")

        layers = ttk.LabelFrame(side, text="Layers (check = show)", padding=(4, 2))
        layers.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        self._build_layers(layers)

        btn = ttk.Frame(side); btn.grid(row=2, column=0, sticky="ew", pady=(2, 4))
        btn.columnconfigure(0, weight=1); btn.columnconfigure(1, weight=1)

        # One crop FOV for everything: the crop export below and the paceTomo
        # reference crops further down both read this box.
        fov = ttk.Frame(btn); fov.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(fov, text="Crop FOV (um):", style="Sm.TLabel").pack(side="left")
        self._crop_fov_var = tk.StringVar(value="2.0")
        ttk.Entry(fov, textvariable=self._crop_fov_var, width=6).pack(side="left", padx=(4, 0))

        ttk.Button(btn, text="Export", style="Accent.TButton",
                   command=self._export).grid(row=1, column=0, columnspan=2,
                                              sticky="ew", pady=(4, 4))

        ttk.Button(btn, text="Remove last", style="Danger.TButton",
                   command=self._remove_last).grid(row=2, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(btn, text="Clear all", style="Danger.TButton",
                   command=self._clear).grid(row=2, column=1, sticky="ew", padx=(2, 0))

        self._export_status = tk.StringVar(value="")
        ttk.Label(btn, textvariable=self._export_status, style="Sm.TLabel",
                  foreground=BG3, wraplength=255,
                  justify="left").grid(row=3, column=0, columnspan=2,
                                       sticky="ew", pady=(6, 0))

        tf = ttk.Frame(side); tf.grid(row=3, column=0, sticky="nsew", pady=(2, 0))
        tf.rowconfigure(0, weight=1); tf.columnconfigure(0, weight=1)
        cols = ("#", "Img X (px)", "Img Y (px)")
        self._tree = ttk.Treeview(tf, columns=cols, show="headings")
        for c, w in zip(cols, [26, 94, 94, 50, 50]):
            self._tree.heading(c, text=c); self._tree.column(c, width=w, anchor="center")
        vsb = ttk.Scrollbar(tf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew"); vsb.grid(row=0, column=1, sticky="ns")

        self._status = tk.StringVar(value="Left-click to pick.\nScroll=zoom  Shift+drag=pan")
        ttk.Label(side, textvariable=self._status, wraplength=255, justify="left",
                  style="Sm.TLabel").grid(row=4, column=0, sticky="ew", pady=(4, 0))
        self._recompose()

    def _build_layers(self, container):
        sc = tk.Canvas(container, bg=BG, highlightthickness=0, height=190)
        vsb = ttk.Scrollbar(container, orient="vertical", command=sc.yview)
        sc.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y"); sc.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(sc); sc.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: sc.configure(scrollregion=sc.bbox("all")))

        self._tem_on = tk.BooleanVar(value=True)
        row = ttk.Frame(inner); row.pack(fill="x", pady=(2, 0))
        tk.Label(row, text="  ", bg="#cccccc", width=2).pack(side="left", padx=(2, 4))
        ttk.Checkbutton(row, text="TEM (MRC)", variable=self._tem_on,
                        command=self._recompose).pack(side="left")
        self._tem_bc = BCControls(inner, callback=self._recompose); self._tem_bc.pack(fill="x", padx=4)

        self._chan_on, self._chan_bc = [], []
        for i in range(len(self._chan_true)):
            role = self._chan_roles[i] if i < len(self._chan_roles) else default_role(i)
            name = self._chan_names[i] if i < len(self._chan_names) else f"ch{i}"
            hex_ = ROLE_HEX.get(role, CHANNEL_HEX[i % len(CHANNEL_HEX)])
            # Reflection is a reference channel: available, but off by default
            # so the picker opens on the fluorescence view.
            on = tk.BooleanVar(value=(role not in ("reflection", "off")))
            r = ttk.Frame(inner); r.pack(fill="x", pady=(4, 0))
            tk.Label(r, text="  ", bg=hex_, width=2).pack(side="left", padx=(2, 4))
            ttk.Checkbutton(r, text=f"Ch {i} ({name})", variable=on,
                            command=self._recompose).pack(side="left")
            bc = BCControls(inner, callback=self._recompose); bc.pack(fill="x", padx=4)
            self._chan_on.append(on); self._chan_bc.append(bc)

    def _view_window(self):
        xl = self._ax.get_xlim(); yl = self._ax.get_ylim()
        x0, x1 = sorted((float(xl[0]), float(xl[1])))
        y0, y1 = sorted((float(yl[0]), float(yl[1])))
        dc0 = int(np.clip(np.floor(x0), 0, self._W)); dc1 = int(np.clip(np.ceil(x1) + 1, 0, self._W))
        dr0 = int(np.clip(np.floor(y0), 0, self._H)); dr1 = int(np.clip(np.ceil(y1) + 1, 0, self._H))
        if dc1 <= dc0: dc0, dc1 = 0, self._W
        if dr1 <= dr0: dr0, dr1 = 0, self._H
        s = max(1, int(np.ceil(max(dc1 - dc0, dr1 - dr0) / self._view_target)))
        return dc0, dc1, dr0, dr1, s

    def _render_view(self):
        self._render_pending = False
        dc0, dc1, dr0, dr1, s = self._view_window()
        cols = np.arange(dc0, dc1, s); rows = np.arange(dr0, dr1, s)
        if len(cols) == 0 or len(rows) == 0:
            return
        # display column/row -> true montage index (montage flip)
        tcols = (self._W - 1 - cols) if MRCReader.MONTAGE_FLIP_X else cols
        trows = (self._H - 1 - rows) if MRCReader.MONTAGE_FLIP_Y else rows
        rr = np.ix_(trows, tcols); shape = (len(rows), len(cols))
        if self._tem_on.get():
            base = apply_bc(self._mrc_true[rr].astype(np.float32),
                            self._tem_bc.vmin, self._tem_bc.vmax) * 0.6
        else:
            base = np.zeros(shape, np.float32)
        rgb = np.empty(shape + (3,), np.float32)
        rgb[..., 0] = base; rgb[..., 1] = base; rgb[..., 2] = base
        for i, L in enumerate(self._chan_true):
            if not self._chan_on[i].get():
                continue
            a = apply_bc(L[rr].astype(np.float32), self._chan_bc[i].vmin, self._chan_bc[i].vmax)
            role = self._chan_roles[i] if i < len(self._chan_roles) else default_role(i)
            col = role_rgb(role) or CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            oma = 1.0 - a
            for k, cv in enumerate(col):
                rgb[..., k] *= oma
                if cv > 0:
                    rgb[..., k] += a * cv
        np.clip(rgb, 0.0, 1.0, out=rgb)
        self._im.set_data(rgb)
        self._im.set_extent([dc0 - 0.5, dc0 + len(cols) * s - 0.5,
                             dr0 + len(rows) * s - 0.5, dr0 - 0.5])
        self._canvas.draw_idle()

    def _schedule_render(self):
        if not self._render_pending:
            self._render_pending = True
            self.after_idle(self._render_view)

    def _recompose(self, *_):
        self._schedule_render()

    def _on_click(self, event):
        if event.button != 1 or event.inaxes is not self._ax or _shift_held(event):
            return
        if event.xdata is None or event.ydata is None:
            return
        # display pixel -> CLEMPicker un-flips to true montage via montage flags
        pick = self.clem_picker.add_pick_from_display(event.xdata, event.ydata)
        self._picks.append(pick)
        self._refresh_tree(); self._redraw_points()
        self._status.set(f"Point #{len(self._picks)}\n"
                 f"Img X: {pick.image_coord_x:.0f} px\n"
                 f"Img Y: {pick.image_coord_y:.0f} px")

    def _redraw_points(self):
        for a in self._pt_artists:
            try: a.remove()
            except Exception: pass
        self._pt_artists = []
        for i, pick in enumerate(self._picks):
            x, y = pick.image_coord_x, pick.image_coord_y      # already display coords
            dot, = self._ax.plot(x, y, "o", color=CYA, markersize=6,
                                markeredgecolor="white", markeredgewidth=0.8, zorder=6)
            txt = self._ax.text(x + 9, y - 9, str(i + 1), color=CYA,
                                fontsize=8, fontweight="bold", zorder=7)
            self._pt_artists.extend([dot, txt])
        self._canvas.draw_idle()

    def _refresh_tree(self):
        self._tree.delete(*self._tree.get_children())
        for i, p in enumerate(self._picks):
            self._tree.insert("", "end", values=(
                i + 1, f"{p.image_coord_x:.0f}", f"{p.image_coord_y:.0f}"))
        if self._picks:
            self._tree.see(self._tree.get_children()[-1])

    def _remove_last(self):
        if self._picks:
            self._picks.pop(); self.clem_picker.remove_last_pick()
            self._refresh_tree(); self._redraw_points()

    def _clear(self):
        self._picks.clear(); self.clem_picker.clear_picks()
        self._refresh_tree(); self._redraw_points()

    def _picks_dir(self):
        """<site folder>/picks -- where the paceTomo export writes too."""
        root = getattr(self.clem_picker, "site_output_root", None)
        if root is None and self.site_data is not None:
            root = self.site_data.path
        if root is None:
            root = os.path.dirname(self.clem_picker.mrc_dataclass.mrc_path)
        return os.path.join(str(root), "picks")

    def _export(self):
        """Export the picks. Four steps, no dialogs:

          1. return the coordinates to SerialEM as navigator points, noted
             "<site>_pick_<n>";
          2. write one MRC crop per pick;
          3. write one multichannel TIFF overlay per pick;
          4. save a screenshot of the picker view.

        Crops use the crop FOV entered above. Everything lands in <site>/picks.
        """
        if not self._picks:
            messagebox.showwarning("Nothing to export", "Pick at least one point.",
                                   parent=self)
            return
        if self.clem_picker is None:
            messagebox.showerror("No CLEMPicker", "CLEMPicker was not provided.",
                                 parent=self)
            return
        try:
            fov = float(self._crop_fov_var.get())
            if fov <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Crop FOV",
                                   "Enter a positive crop FOV in micrometres.",
                                   parent=self)
            return

        picks_dir = self._picks_dir()
        site_label = getattr(self.clem_picker, "site_id", None)
        done = []
        try:
            os.makedirs(picks_dir, exist_ok=True)

            # 1. coordinates back to SerialEM
            self._status.set(f"Registering {len(self._picks)} pick(s) in the navigator...")
            self.update_idletasks()
            nav_idx = self.clem_picker.add_picks_to_navigator(
                self.clem_picker.mrc_dataclass, buffer=self.clem_picker.nav_map_buffer,
                site_label=site_label)
            n_nav = len(nav_idx or [])
            done.append(f"{n_nav} navigator point(s)"
                        + (f' noted "{site_label}_pick_N"' if site_label else ""))

            # 2 + 3. MRC crops and multichannel TIFF overlays
            self._status.set(f"Writing crops for {len(self._picks)} pick(s)...")
            self.update_idletasks()
            res = self.mrc_reader.write_fov_crops(
                mrc_dataclass=self.clem_picker.mrc_dataclass,
                warp_slice=self._warp_slice_true,
                warp_crop=self._warp_crop_true,
                n_channels=len(self._chan_true), n_z=self._n_z,
                fov_um=fov, picks_dir=picks_dir)
            done.append(f"{len(res['mrc'])} MRC crop(s)")
            done.append(f"{len(res['tif'])} TIFF overlay(s) at {fov} um")

            # 4. screenshot of the picks
            shot = os.path.join(picks_dir, "picker_screenshot.png")
            self._fig.savefig(shot, dpi=150, facecolor=self._fig.get_facecolor(),
                              bbox_inches="tight")
            done.append(f"screenshot {os.path.basename(shot)}")

            summary = "\n".join(f"  - {d}" for d in done)
            self._export_status.set(f"Exported to {picks_dir}")
            self._status.set("Export complete.")
            messagebox.showinfo("Export complete",
                                f"Written to:\n{picks_dir}\n\n{summary}", parent=self)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._status.set("Export failed.")
            self._export_status.set(
                "Export failed" + (f" after: {', '.join(done)}" if done else "."))
            messagebox.showerror("Export error", str(e), parent=self)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class RegistrationApp(tk.Tk):

    def __init__(self, mrc_reader, site_data, tem_communication,
                 reuse_transform=False):
        super().__init__()
        self.mrc_reader = mrc_reader
        self.tem = tem_communication
        self.correlator = CLEMCorrelator(mrc_reader=self.mrc_reader)
        self.site_data = site_data
        self.site_id = self.site_data.site_id
        # When True, startup re-applies the newest transform stored under
        # <site>/transforms and locks the flip checkboxes to that orientation.
        # Off by default: a normal run should start from a clean slate, and
        # only a deliberate restart of the alignment stage should reuse a fit.
        self.reuse_transform = bool(reuse_transform)
        self.title("MRC / OME-TIFF Registration Tool")
        self.configure(bg=BG)
        self.minsize(1150, 820)

        self.mrc_image = None          # TRUE montage array
        self.tiff_stack = None         # raw (C,Z,Y,X)
        self.warped_channels = []      # DISPLAY-MRC frame
        self.point_pairs = []          # DISPLAY-frame picks

        self.flip_x = tk.BooleanVar(value=TIFF_FLIP_X_DEFAULT)   # TIFF display X flip
        self.flip_y = tk.BooleanVar(value=TIFF_FLIP_Y_DEFAULT)   # TIFF display Y flip

        self._last_tform = None
        self._loaded_record = None
        # Rotation preview for the right panel (from "Import Rot/Flip Only"):
        # display-TIFF -> display-MRC rotation in degrees, plus the forward /
        # inverse 3x3 maps between unrotated and rotated display coordinates.
        self.tiff_disp_rot_deg = 0.0
        self._disp_rot_fwd = None
        self._disp_rot_inv = None
        self._rot_canvas_hw = None
        self._mrc_img_dirty = False
        self._tiff_img_dirty = False
        self._mrc_pt_artists = []
        self._tiff_pt_artists = []

        self.mrc_is_montage = False
        self.mrc_file_path = None
        self.mrc_section_map = {}
        self.mrc_mont_canvas = {}
        self.mrc_mont_info = {}
        self.mrc_img_hw = (4096, 4096)
        self.mrc_feather_px = 410
        self.mrc_montage_cache = {}
        self.mrc_n_sections = 0
        self.mrc_current_pieces = []
        self.mrc_pixel_spacing_um = 1.0

        self._build_styles()
        self._build_ui()

        if self.site_id is not None:
            try:
                self._load_site_data()
            except Exception as e:
                import traceback; traceback.print_exc()
                messagebox.showerror("Could not load site data",
                                     f"{type(e).__name__}: {e}\n\nThe window will open empty.")

    # ---------------- helpers ----------------

    def _tiff_scale(self):
        t = self._resolve_latest_tiff()
        if t is None:
            return None
        tps = getattr(t, "pixel_spacing_um", None) or getattr(t, "czi_pixel_spacing_um", None)
        return float(tps) if tps else None

    # ---------------- resolve entries from the site dataclass ----------------

    def _resolve_latest_mrc(self):
        """Latest MRCSummary held in the site's mrc dictionary.  Prefers the
        reader's finder, then the newest entry in site_data.mrcs, then any
        legacy single-mrc attribute."""
        finder = getattr(self.mrc_reader, "_find_latest_mrc_dataclass", None)
        if callable(finder):
            try:
                m = finder(self.site_data)
                if m is not None:
                    return m
            except Exception:
                pass
        mrcs = getattr(self.site_data, "mrcs", None) or {}
        candidates = [m for m in mrcs.values() if m is not None]
        if candidates:
            return max(candidates, key=lambda m: getattr(m, "timestamp", None) or "")
        return getattr(self.site_data, "mrc", None)   # legacy fallback

    @staticmethod
    def _tiff_has_data(t):
        return t is not None and (getattr(t, "stack_czyx", None) is not None
                                  or getattr(t, "czi_overview", None) is not None)

    def _resolve_latest_tiff(self):
        """Latest TiffSummary that actually carries image data (tif stack or
        czi overview) from the site's tiff dictionary; None if tif AND czi are
        both empty."""
        tiffs = getattr(self.site_data, "tiffs", None) or {}
        candidates = [t for t in tiffs.values() if self._tiff_has_data(t)]
        if candidates:
            return candidates[-1]
        legacy = getattr(self.site_data, "tiff", None)     # legacy fallback
        return legacy if self._tiff_has_data(legacy) else None

    MRC_EXTS = (".mrc", ".rec", ".mrcs", ".map", ".st")

    @staticmethod
    def _is_real_image_file(p):
        """False for macOS AppleDouble sidecars ('._name.tif', header
        0x00051607) and other hidden files, which pathlib.glob('*.tif')
        happily matches and which crash tifffile/mrcfile."""
        return not Path(p).name.startswith((".", "._"))

    def _candidate_mrcs_in_folder(self, folder):
        """All MRC-like files in the site folder (then one level of subfolders),
        newest last.  Files that have an mdoc beside them come first, because
        build_montage_summary() needs one."""
        folder = Path(folder)
        found = []
        for pattern in ("*", "*/*"):
            for p in folder.glob(pattern):
                if (p.is_file() and p.suffix.lower() in self.MRC_EXTS
                        and self._is_real_image_file(p)):
                    found.append(p)
            if found:
                break
        finder = getattr(self.mrc_reader, "_find_mdoc_path", None)

        def _has_mdoc(p):
            if not callable(finder):
                return True
            try:
                return finder(mrc_filepath=str(p)) is not None
            except Exception:
                return False

        return sorted(found, key=lambda p: (_has_mdoc(p), p.stat().st_mtime))

    def _load_latest_mrc_from_folder(self):
        """Pull the newest MRC montage sitting in the site folder into the
        dataclass and return its MRCSummary.  Tries candidates newest-first so
        a single bad/mdoc-less file does not abort the auto-load."""
        folder = getattr(self.site_data, "path", None)
        if not folder:
            return None
        candidates = self._candidate_mrcs_in_folder(folder)
        if not candidates:
            self.status_var.set(f"No MRC file found in {folder}")
            return None
        last_err = None
        for p in reversed(candidates):
            try:
                mrc_dc = self.mrc_reader.load_mrc_into_data_class(
                    site_data=self.site_data, mrc_path=str(p))
                if mrc_dc is not None:
                    return mrc_dc
            except Exception as e:
                last_err = e
                print(f"[WARN] Skipping {p.name}: {type(e).__name__}: {e}")
        if last_err is not None:
            self.status_var.set(f"Could not load MRC from folder: {last_err}")
        return None

    def _load_latest_tiff_from_folder(self):
        """tif AND czi both empty in the dataclass: pull the newest LM file
        from the site folder into the dataclass (via the reader) and return
        the resulting TiffSummary."""
        folder = getattr(self.site_data, "path", None)
        if not folder:
            return None
        folder = Path(folder)
        # NB: '*.tif' already covers '*.ome.tif'; use a set so nothing is
        # counted twice, and drop AppleDouble '._' sidecars (glob matches
        # them, tifffile then dies with "not a Tiff file: header=0x00051607").
        ome = sorted({p for p in [*folder.glob("*.tif"), *folder.glob("*.tiff")]
                      if self._is_real_image_file(p)},
                     key=lambda p: p.stat().st_mtime)
        czi = sorted((p for p in folder.glob("*.czi")
                      if self._is_real_image_file(p)),
                     key=lambda p: p.stat().st_mtime)
        try:
            if ome:
                self.mrc_reader.load_tiff_into_data_class(site_data=self.site_data,
                                                          ome_path=str(ome[-1]))
            elif czi:
                self.mrc_reader.load_czi_into_data_class(site_data=self.site_data,
                                                         czi_path=str(czi[-1]))
            else:
                return None
        except Exception as e:
            import traceback; traceback.print_exc()
            self.status_var.set(f"Could not load LM image from folder: {e}")
            return None
        return self._resolve_latest_tiff()

    def _sync_flip_to_mrc(self):
        """Persist the current flip toggles onto the displayed (latest) MRC
        dataclass, read directly from the site data."""
        mrc_dc = self._resolve_latest_mrc()
        if mrc_dc is None:
            mrc_dc = self._load_latest_mrc_from_folder()
        if mrc_dc is not None:
            self._display_loaded_mrc_data(mrc_dc)

    def _get_tiff_slice(self, c, z):
        """DISPLAY slice: flipped per the X and Y checkboxes, then rotated by
        the imported rotation preview (if one is active)."""
        img = MRCReader._flip_for_display(self.tiff_stack[c, z], self.flip_x.get(), self.flip_y.get())
        if self._disp_rot_inv is not None:
            img = warp(img, ProjectiveTransform(matrix=self._disp_rot_inv),
                       output_shape=self._rot_canvas_hw, order=1,
                       preserve_range=True, mode="constant", cval=0.0).astype(np.float32)
        return img

    # ---------------- rotation preview (Import Rot/Flip Only) ----------------

    def _set_display_rotation(self, rot_deg):
        """Enable/disable the rotation preview on the right panel.

        rot_deg is the display-TIFF -> display-MRC rotation (deg) taken from
        the transform record; the rotated canvas is enlarged so the whole
        frame stays visible.  Content is rotated exactly as the warp onto the
        MRC rotates it, so the right panel previews the TIFF as it will land
        on the montage."""
        self.tiff_disp_rot_deg = float(rot_deg or 0.0)
        self._disp_rot_fwd = self._disp_rot_inv = self._rot_canvas_hw = None
        if self.tiff_stack is None or abs(self.tiff_disp_rot_deg) < 1e-6:
            return
        h, w = self.tiff_stack.shape[-2:]
        a = np.deg2rad(self.tiff_disp_rot_deg)
        R = np.array([[np.cos(a), -np.sin(a)],
                      [np.sin(a),  np.cos(a)]], dtype=float)
        c_in = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
        rc = (R @ (corners - c_in).T).T
        w2 = int(np.ceil(rc[:, 0].max() - rc[:, 0].min() + 1))
        h2 = int(np.ceil(rc[:, 1].max() - rc[:, 1].min() + 1))
        F = np.eye(3)
        F[:2, :2] = R
        F[:2, 2] = np.array([(w2 - 1) / 2.0, (h2 - 1) / 2.0]) - R @ c_in
        self._disp_rot_fwd = F                       # unrotated -> rotated coords
        self._disp_rot_inv = np.linalg.inv(F)        # rotated -> unrotated coords
        self._rot_canvas_hw = (h2, w2)

    def _pairs_for_fit(self):
        """Landmark pairs with TIFF picks mapped back from the rotated display
        canvas to the unrotated display frame the correlator fits/warps in."""
        if self._disp_rot_inv is None:
            return self.point_pairs
        out = []
        for p in self.point_pairs:
            q = dict(p)
            if "tiff" in q:
                v = self._disp_rot_inv @ np.array([q["tiff"][0], q["tiff"][1], 1.0])
                q["tiff"] = (float(v[0]), float(v[1]))
            out.append(q)
        return out

    # ---------------- display of stored site data ----------------

    def _display_loaded_site_data(self):
        # MRC: newest entry from the site's mrc dictionary; if the dictionary is
        # empty, pull the newest MRC montage from the site folder into it.
        mrc_dc = self._resolve_latest_mrc()
        if mrc_dc is None:
            mrc_dc = self._load_latest_mrc_from_folder()
        if mrc_dc is not None:
            self._display_loaded_mrc_data(mrc_dc)

        # TIFF / CZI: from the dataclass; if tif AND czi are both empty, pull
        # the newest LM file from the site folder into the dataclass.
        tiff_dc = self._resolve_latest_tiff()
        if tiff_dc is None:
            tiff_dc = self._load_latest_tiff_from_folder()
        if tiff_dc is not None:
            self._display_loaded_tiff_data(tiff_dc)

    def _display_loaded_mrc_data(self, data):
        self.mrc_file_path = os.fspath(data.mrc_path)
        self.mrc_montage_cache = {data.section: data.image}
        self.mrc_img_hw = (data.image_height, data.image_width)
        self.mrc_feather_px = data.feather_pixels
        self.mrc_pixel_spacing_um = data.pixel_spacing_um
        self.mrc_is_montage = True
        self.mrc_section_map = {data.section: data.tiles}
        self.mrc_current_pieces = data.tiles      # enables the stage picker
        self.mrc_n_sections = 1
        self.mrc_mont_canvas = {data.section: (data.image.shape[1], data.image.shape[0])}
        self.mrc_mont_info = {data.section: f"{len(data.tiles)} tiles"}
        self.montage_spin.config(to=0); self.montage_var.set(0)
        self.montage_count_var.set("/ 1 sections")
        self.bc_mrc.reset()
        self.mrc_info_var.set(os.path.basename(self.mrc_file_path))
        self.mrc_nav_frame.grid(row=1, column=0, sticky="ew", pady=(2, 2))
        self.mrc_image = data.image
        self._mrc_img_dirty = True
        self._draw_mrc()
        self._update_scale_info()

    def _display_loaded_tiff_data(self, data):
        stack = data.stack_czyx if getattr(data, "stack_czyx", None) is not None else data.czi_overview
        if stack is None:
            return
        self.tiff_stack = stack
        self._set_display_rotation(0.0)     # new image: drop any rotation preview
        self._tiff_img_dirty = True
        C, Z = self.tiff_stack.shape[:2]
        self.channel_spin.config(to=max(0, C - 1)); self.channel_var.set(0)
        self.z_spin.config(to=max(0, Z - 1)); self.z_var.set(0)
        self.flip_x.set(TIFF_FLIP_X_DEFAULT); self.flip_y.set(TIFF_FLIP_Y_DEFAULT)
        self._sync_flip_to_mrc()
        # Keep the operator's role assignments when reloading a stack with the
        # same channel count; otherwise fall back to the positional defaults.
        prev_roles = self.bc_tiff_panel.roles()
        keep = prev_roles if len(prev_roles) == C else None
        self.bc_tiff_panel.build(C, self._on_bc_tiff, roles=keep,
                                 on_role_change=self._on_channel_role_change)
        src_path = getattr(data, "ome_path", None) or getattr(data, "czi_path", None)
        label = os.path.basename(os.fspath(src_path)) if src_path else "image"
        self.tiff_info_var.set(label + "  " + (getattr(data, "info", "") or ""))
        self._update_scale_info()
        self._draw_tiff()

    # ---------------- styles / UI ----------------

    def _build_styles(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        s.configure("Sm.TLabel", background=BG, foreground=FG, font=("Segoe UI", 9))
        s.configure("TButton", background=ACC, foreground=BG, font=("Segoe UI", 10, "bold"), padding=5)
        s.map("TButton", background=[("active", CYA), ("disabled", BG3)])
        s.configure("Accent.TButton", background=ACC2, foreground=BG, font=("Segoe UI", 11, "bold"), padding=7)
        s.configure("Danger.TButton", background=RED, foreground=BG, font=("Segoe UI", 10, "bold"), padding=5)
        s.configure("Mont.TButton", background="#cba6f7", foreground=BG, font=("Segoe UI", 10, "bold"), padding=5)
        s.configure("TLabelframe", background=BG, relief="groove")
        s.configure("TLabelframe.Label", background=BG, foreground=CYA, font=("Segoe UI", 10, "bold"))
        s.configure("TRadiobutton", background=BG, foreground=FG, font=("Segoe UI", 9))
        s.configure("TScale", background=BG, troughcolor=BG2, sliderlength=14)
        s.configure("Treeview", background=BG2, foreground=FG, fieldbackground=BG2, rowheight=22)
        s.configure("Treeview.Heading", background=BG3, foreground=FG)
        s.map("Treeview", background=[("selected", ACC)])

    def _build_ui(self):
        PAD = 8
        top = ttk.Frame(self, padding=(PAD, PAD, PAD, 0)); top.pack(fill="x")
        ttk.Button(top, text="Load MRC + mdoc", style="Mont.TButton",
                   command=self._load_mrc_mdoc).pack(side="left", padx=3)
        self.mrc_info_var = tk.StringVar(value="No MRC loaded")
        ttk.Label(top, textvariable=self.mrc_info_var, width=38, style="Sm.TLabel").pack(side="left", padx=3)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(top, text="Load OME-TIFF / CZI", command=self._load_tiff).pack(side="left", padx=3)
        self.tiff_info_var = tk.StringVar(value="No image loaded")
        ttk.Label(top, textvariable=self.tiff_info_var, width=36, style="Sm.TLabel").pack(side="left", padx=3)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(top, text="Ch:", style="Sm.TLabel").pack(side="left")
        self.channel_var = tk.IntVar(value=0)
        self.channel_spin = ttk.Spinbox(top, from_=0, to=0, textvariable=self.channel_var,
                                        width=4, command=self._refresh_tiff)
        self.channel_spin.pack(side="left", padx=2)
        ttk.Label(top, text="Z:", style="Sm.TLabel").pack(side="left")
        self.z_var = tk.IntVar(value=0)
        self.z_spin = ttk.Spinbox(top, from_=0, to=0, textvariable=self.z_var,
                                  width=4, command=self._refresh_tiff)
        self.z_spin.pack(side="left", padx=2)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        self.scale_info_var = tk.StringVar(value="TIF px ?   MRC px ?   scale ?")
        ttk.Label(top, textvariable=self.scale_info_var, style="Sm.TLabel",
                  foreground=CYA).pack(side="left", padx=3)
        ttk.Label(top, text="  scroll=zoom  middle/shift-drag=pan  left-click=landmark",
                  style="Sm.TLabel", foreground=BG3).pack(side="right", padx=6)

        panels = ttk.Frame(self, padding=PAD); panels.pack(fill="both", expand=True)
        panels.columnconfigure(0, weight=1); panels.columnconfigure(1, weight=1)
        panels.rowconfigure(0, weight=1)
        (self.canvas_mrc, self.ax_mrc, self.bc_mrc, self.mrc_nav_frame) = self._make_mrc_panel(panels)
        (self.canvas_tiff, self.ax_tiff, self.bc_tiff_panel) = self._make_tiff_panel(panels)

        bot = ttk.Frame(self, padding=(PAD, 0, PAD, PAD)); bot.pack(fill="x")
        pf = ttk.LabelFrame(bot, text="Landmark pairs (display-frame pixels)", padding=4)
        pf.pack(side="left", fill="both", expand=True)
        cols = ("#", "MRC (x, y)", "TIFF (x, y)")
        self.tree = ttk.Treeview(pf, columns=cols, show="headings", height=2)
        for c in cols:
            self.tree.heading(c, text=c); self.tree.column(c, width=130, anchor="center")
        sb = ttk.Scrollbar(pf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True); sb.pack(side="left", fill="y")

        bf = ttk.Frame(bot, padding=(10, 0, 0, 0)); bf.pack(side="left", fill="y")
        self.status_var = tk.StringVar(value="Flip the TIFF to match the MRC, then click landmarks.")
        ttk.Label(bf, textvariable=self.status_var, wraplength=215, justify="left",
                  style="Sm.TLabel").pack(pady=(0, 4), anchor="w")
        tf_row = ttk.Frame(bf); tf_row.pack(fill="x", pady=(0, 4))
        ttk.Label(tf_row, text="Transform:", style="Sm.TLabel").pack(side="left")
        self.transform_var = tk.StringVar(value="similarity")
        ttk.Combobox(tf_row, textvariable=self.transform_var, width=11, state="readonly",
                     values=["euclidean", "similarity", "affine", "projective"]).pack(side="left", padx=(4, 0))
        bg = ttk.Frame(bf); bg.pack(fill="x")
        bg.columnconfigure(0, weight=1); bg.columnconfigure(1, weight=1)
        ttk.Button(bg, text="Remove last", style="Danger.TButton",
                   command=self._remove_last).grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=2)
        ttk.Button(bg, text="Clear all", style="Danger.TButton",
                   command=self._clear_points).grid(row=0, column=1, sticky="ew", padx=(2, 0), pady=2)
        ttk.Button(bg, text="Apply Transform", style="Accent.TButton",
                   command=self._apply_transform).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 2))
        ttk.Button(bg, text="Show Overlay",
                   command=self._show_overlay).grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(bg, text="Export Transform",
                   command=self._export_transform).grid(row=3, column=0, sticky="ew", padx=(0, 2), pady=2)
        ttk.Button(bg, text="Import Transform",
                   command=self._import_transform).grid(row=3, column=1, sticky="ew", padx=(2, 0), pady=2)
        ttk.Button(bg, text="Import Rot/Flip Only", style="Mont.TButton",
                   command=self._import_transform_rot_only).grid(row=4, column=0, columnspan=2,
                                                                 sticky="ew", pady=2)

    def _make_mrc_panel(self, parent):
        lf = ttk.LabelFrame(parent, text="MRC  --  left-click to place landmark", padding=4)
        lf.grid(row=0, column=0, sticky="nsew", padx=(0, 4)); lf.rowconfigure(0, weight=1); lf.columnconfigure(0, weight=1)
        fig = Figure(figsize=(5, 3.2), facecolor=BG); ax = fig.add_subplot(111); self._style_ax(ax)
        fig.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.03)
        canvas = FigureCanvasTkAgg(fig, master=lf); canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        nav = ttk.Frame(lf, padding=(2, 2))
        ttk.Label(nav, text="Montage:", style="Sm.TLabel").pack(side="left")
        self.montage_var = tk.IntVar(value=0)
        self.montage_spin = ttk.Spinbox(nav, from_=0, to=0, textvariable=self.montage_var,
                                        width=4, command=self._on_montage_changed)
        self.montage_spin.pack(side="left", padx=(2, 0))
        self.montage_count_var = tk.StringVar(value="/ 0")
        ttk.Label(nav, textvariable=self.montage_count_var, style="Sm.TLabel").pack(side="left", padx=(2, 8))
        ttk.Button(nav, text="<", width=2, command=self._montage_prev).pack(side="left", padx=1)
        ttk.Button(nav, text=">", width=2, command=self._montage_next).pack(side="left", padx=1)
        self.montage_info_var = tk.StringVar(value="")
        ttk.Label(nav, textvariable=self.montage_info_var, style="Sm.TLabel", foreground=CYA).pack(side="left", padx=(8, 0))
        bc_frame = ttk.LabelFrame(lf, text="Brightness / Contrast (MRC)", padding=(4, 2))
        bc_frame.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        bc = BCControls(bc_frame, callback=self._on_bc_mrc); bc.pack(fill="x")
        PanZoomHandler(ax, canvas)
        canvas.mpl_connect("button_press_event", self._on_click_mrc)
        return canvas, ax, bc, nav

    def _make_tiff_panel(self, parent):
        lf = ttk.LabelFrame(parent, text="OME-TIFF  --  flip to match MRC, then left-click", padding=4)
        lf.grid(row=0, column=1, sticky="nsew", padx=(4, 0)); lf.rowconfigure(0, weight=1); lf.columnconfigure(0, weight=1)
        fig = Figure(figsize=(5, 3.2), facecolor=BG); ax = fig.add_subplot(111); self._style_ax(ax)
        fig.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.03)
        canvas = FigureCanvasTkAgg(fig, master=lf); canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        flip = ttk.Frame(lf, padding=(4, 2)); flip.grid(row=1, column=0, sticky="ew")
        ttk.Label(flip, text="Flip:", style="Sm.TLabel").pack(side="left", padx=(0, 4))
        self.flip_x_cb = ttk.Checkbutton(flip, text="X  (left <-> right)", variable=self.flip_x,
                                         command=self._on_flip_changed)
        self.flip_x_cb.pack(side="left", padx=4)
        self.flip_y_cb = ttk.Checkbutton(flip, text="Y  (up <-> down)", variable=self.flip_y,
                                         command=self._on_flip_changed)
        self.flip_y_cb.pack(side="left", padx=4)
        outer = ttk.LabelFrame(lf, text="Brightness / Contrast (per channel)", padding=(4, 2))
        outer.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        sc = tk.Canvas(outer, bg=BG, highlightthickness=0, height=80)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=sc.yview)
        sc.configure(yscrollcommand=vsb.set); vsb.pack(side="right", fill="y"); sc.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(sc); sc.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: sc.configure(scrollregion=sc.bbox("all")))
        bc_panel = ChannelBCPanel(inner); bc_panel.pack(fill="x")
        PanZoomHandler(ax, canvas)
        canvas.mpl_connect("button_press_event", self._on_click_tiff)
        return canvas, ax, bc_panel

    @staticmethod
    def _style_ax(ax):
        ax.set_facecolor(BG); ax.tick_params(colors=BG3)
        for sp in ax.spines.values():
            sp.set_edgecolor(BG3)

    def _montage_prev(self):
        if self.montage_var.get() > 0:
            self.montage_var.set(self.montage_var.get() - 1); self._on_montage_changed()

    def _montage_next(self):
        if self.montage_var.get() < self.mrc_n_sections - 1:
            self.montage_var.set(self.montage_var.get() + 1); self._on_montage_changed()

    def _on_bc_mrc(self):
        imgs = self.ax_mrc.get_images()
        if imgs:
            imgs[0].set_clim(self.bc_mrc.vmin, self.bc_mrc.vmax); self.canvas_mrc.draw_idle()

    def _on_bc_tiff(self, channel_idx):
        cur = min(self.channel_var.get(), self.tiff_stack.shape[0] - 1) if self.tiff_stack is not None else 0
        if channel_idx != cur:
            return
        bc = self.bc_tiff_panel.bc(cur)
        imgs = self.ax_tiff.get_images()
        if bc and imgs:
            imgs[0].set_clim(bc.vmin, bc.vmax); self.canvas_tiff.draw_idle()

    def _on_channel_role_change(self, channel_idx):
        """A channel was reassigned to a different role.

        The single-channel TIFF preview is greyscale, so nothing on the main
        window needs repainting; the roles are read when an overlay is built.
        """
        role = self.bc_tiff_panel.role(channel_idx)
        self.status_var.set(f"Ch {channel_idx} set to '{role}'. "
                            f"Rebuild the overlay to see it.")

    # ---- channel roles ---------------------------------------------------- #

    def channel_roles(self):
        return self.bc_tiff_panel.roles()

    def _composite_colors(self):
        """Colours for the fluorescence composite (reflection/off excluded)."""
        colors = self.bc_tiff_panel.composite_colors()
        n = len(self.warped_channels)
        return (colors + [None] * n)[:n]

    def _reflection_indices(self):
        return [i for i in self.bc_tiff_panel.indices_with_role("reflection")
                if i < len(self.warped_channels)]

    def _role_label(self, idx):
        role = self.bc_tiff_panel.role(idx) if idx < self.bc_tiff_panel.n_channels else "off"
        return role

    # ---------------- loading ----------------

    def _resolve_picked_mrc_path(self, path):
        """Accept either an MRC-like file or an .mdoc; for an .mdoc, return the
        MRC it belongs to."""
        p = Path(path)
        if p.suffix.lower() != ".mdoc":
            return p
        stem = p.with_suffix("")                       # foo.mrc.mdoc -> foo.mrc
        if stem.suffix.lower() in self.MRC_EXTS and stem.is_file():
            return stem
        for ext in self.MRC_EXTS:                      # foo.mdoc -> foo.mrc
            cand = p.with_suffix(ext)
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"No MRC file found next to {p.name}.")

    def _load_mrc_mdoc(self):
        path = filedialog.askopenfilename(title="Open MRC montage",
            initialdir=getattr(self.site_data, "path", None) or None,
            filetypes=[("MRC / mdoc", ("*.mrc", "*.rec", "*.mrcs", "*.map", "*.st", "*.mdoc")),
                       ("MRC", ("*.mrc", "*.rec", "*.mrcs", "*.map", "*.st")),
                       ("All files", "*")])
        if not path:
            return
        try:
            mrc_path = self._resolve_picked_mrc_path(path)
            mrc_dc = self.mrc_reader.load_mrc_into_data_class(
                site_data=self.site_data, mrc_path=str(mrc_path))
            if mrc_dc is None:
                # Older readers return None and only write into the dataclass.
                mrc_dc = self._resolve_latest_mrc()
            if mrc_dc is None:
                raise RuntimeError(
                    f"{Path(mrc_path).name} was not stored in site_data.mrcs "
                    f"(load_mrc_into_data_class returned nothing).")
            self._display_loaded_mrc_data(mrc_dc)
            if getattr(mrc_dc, "mrc_path", None):
                self.tem.load_mrc_in_nav(mrc_dataclass=mrc_dc, buffer='S')
            self.status_var.set("MRC montage loaded.")
            self._draw_mrc(keep_view=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("MRC montage load error", str(e))

    def _load_tiff(self):
        path = filedialog.askopenfilename(title="Open OME-TIFF / CZI",
            filetypes=[("Light microscopy", ("*.tif", "*.tiff", "*.czi")), ("All files", "*")])
        if not path:
            return
        try:
            if not self._is_real_image_file(path):
                real = Path(path).with_name(Path(path).name.lstrip("._"))
                raise ValueError(
                    f"{Path(path).name} is a macOS metadata sidecar (AppleDouble), "
                    f"not an image. Load {real.name} instead.")
            if os.path.splitext(path)[1].lower() == ".czi":
                self.mrc_reader.load_czi_into_data_class(site_data=self.site_data, czi_path=path)
            else:
                self.mrc_reader.load_tiff_into_data_class(site_data=self.site_data, ome_path=path)
            self.warped_channels = []
            self._loaded_record = None
            self._last_tform = None
            for cb in (getattr(self, "flip_x_cb", None), getattr(self, "flip_y_cb", None)):
                if cb is not None:
                    cb.config(state="normal")
            tiff_dc = self._resolve_latest_tiff()
            if tiff_dc is not None:
                self._display_loaded_tiff_data(tiff_dc)
            self.status_var.set("Image loaded. Flip to match the MRC, then pick landmarks.")
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("TIFF load error", str(e))

    def _load_site_data(self):
        # Everything is read from the site dataclass that was handed to the app.
        self._display_loaded_site_data()
        mrc_dc = self._resolve_latest_mrc()
        if mrc_dc is not None and getattr(mrc_dc, "mrc_path", None):
            self.tem.load_mrc_in_nav(mrc_dataclass=mrc_dc, buffer='S')
        # Startup auto re-apply if a stored transform exists for this site.
        # Opt-in via RegistrationApp(..., reuse_transform=True) -- used when
        # restarting the alignment stage on a site that was already fitted.
        record = None
        if self.reuse_transform:
            finder = getattr(self.mrc_reader, "_find_latest_transform", None)
            if callable(finder):
                try:
                    # The finder returns a path; the correlator parses it into
                    # a TransformRecord (yaml / csv / legacy txt).
                    path = finder(self.site_data)
                    if path is not None:
                        record = self.correlator.load_transform(os.fspath(path))
                        print(f"[INFO] Reusing stored transform: {path}")
                except Exception:
                    import traceback; traceback.print_exc()
                    record = None
        if record is not None and self.tiff_stack is not None:
            self._loaded_record = record
            self.flip_x.set(bool(record.flip_x))
            self.flip_y.set(bool(record.flip_y))
            self._sync_flip_to_mrc()
            for cb in (getattr(self, "flip_x_cb", None), getattr(self, "flip_y_cb", None)):
                if cb is not None:
                    cb.config(state="disabled")   # 2nd fit reuses the 1st fit's orientation
            self._refresh_tiff()
            self._apply_transform()

    # ---------------- montage assembly / nav ----------------

    def _on_montage_changed(self):
        if not self.mrc_is_montage:
            return
        secs = sorted(self.mrc_section_map.keys())
        idx = min(self.montage_var.get(), len(secs) - 1); self.montage_var.set(idx)
        sec = secs[idx]
        self.mrc_current_pieces = self.mrc_section_map.get(sec, [])
        self.montage_info_var.set(self.mrc_mont_info.get(sec, ""))
        self.mrc_image = self._get_montage(sec)
        self._mrc_img_dirty = True
        self._draw_mrc()
        self.status_var.set(f"Montage {idx+1}/{self.mrc_n_sections}  {self.mrc_mont_info.get(sec, '')}")

    def _get_montage(self, sec_idx):
        if sec_idx in self.mrc_montage_cache:
            return self.mrc_montage_cache[sec_idx]
        pieces = self.mrc_section_map[sec_idx]

        def status_cb(msg):
            self.status_var.set(msg); self.update_idletasks()

        mont = self.mrc_reader.assemble_section(
            self.mrc_file_path, self.mrc_img_hw, self.mrc_feather_px,
            pieces=pieces, status_cb=status_cb)
        self.mrc_montage_cache[sec_idx] = mont
        return mont

    # ---------------- drawing (store & draw in DISPLAY frame) ----------------

    def _draw_mrc(self, keep_view=False):
        if self.mrc_image is None:
            return
        if self._mrc_img_dirty:
            xl, yl = self._save_view(self.ax_mrc) if keep_view else (None, None)
            self.ax_mrc.clear(); self._style_ax(self.ax_mrc)
            img = self.mrc_image; h, w = img.shape
            ds = max(1, max(h, w) // 1024)
            disp = _mrc_to_display(img[::ds, ::ds] if ds > 1 else img)
            self.ax_mrc.imshow(disp, cmap="gray", origin="upper", aspect="equal",
                               extent=[-0.5, w - 0.5, h - 0.5, -0.5],
                               vmin=self.bc_mrc.vmin, vmax=self.bc_mrc.vmax)
            if keep_view:
                self._restore_view(self.ax_mrc, xl, yl)
            self._mrc_img_dirty = False; self._mrc_pt_artists = []
        else:
            for a in self._mrc_pt_artists:
                try: a.remove()
                except Exception: pass
            self._mrc_pt_artists = []
        for i, pair in enumerate(self.point_pairs):
            if "mrc" in pair:
                dx, dy = pair["mrc"]                          # DISPLAY coords, drawn as-is
                ln, = self.ax_mrc.plot(dx, dy, "o", color=PT_MRC, markersize=8,
                                       markeredgecolor="white", markeredgewidth=0.8, zorder=5)
                tx = self.ax_mrc.text(dx + 6, dy - 6, str(i + 1), color=PT_MRC,
                                      fontsize=9, fontweight="bold", zorder=6)
                self._mrc_pt_artists.extend([ln, tx])
        if self._last_tform is not None and self.tiff_stack is not None:
            H, W = self.tiff_stack.shape[-2:]
            corners = np.array([[-0.5, -0.5],
                                [W-0.5, -0.5],
                                [W-0.5, H-0.5],
                                [-0.5, H-0.5],
                                [-0.5, -0.5]], dtype=float)
            pts = self._last_tform(corners)
            ln, = self.ax_mrc.plot(pts[:, 0], pts[:, 1], "-", color="white",
                                   linewidth=1.5, alpha=0.9, zorder=7)
            self._mrc_pt_artists.append(ln)
            # annotate the frame with rotation angle and physical size
            M = np.asarray(self._last_tform.params, dtype=float)
            rot = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
            tps = self._tiff_scale()
            lbl = f"rot {rot:+.1f} deg"
            if tps:
                lbl += f"   {W * tps:.0f} x {H * tps:.0f} um"
            tx = self.ax_mrc.text(float(pts[:, 0].min()), float(pts[:, 1].min()) - 10,
                                  lbl, color="white", fontsize=8, zorder=7,
                                  path_effects=[pe.withStroke(linewidth=2, foreground="black")])
            self._mrc_pt_artists.append(tx)
        self.canvas_mrc.draw_idle()

    def _draw_tiff(self, keep_view=False):
        if self.tiff_stack is None:
            return
        c = min(self.channel_var.get(), self.tiff_stack.shape[0] - 1)
        z = min(self.z_var.get(), self.tiff_stack.shape[1] - 1)
        bc = self.bc_tiff_panel.bc(c)
        vmin, vmax = (bc.vmin, bc.vmax) if bc else (0.0, 1.0)
        if self._tiff_img_dirty:
            xl, yl = self._save_view(self.ax_tiff) if keep_view else (None, None)
            self.ax_tiff.clear(); self._style_ax(self.ax_tiff)
            img = self._get_tiff_slice(c, z); h, w = img.shape
            ds = max(1, max(h, w) // 1024)
            disp = img[::ds, ::ds] if ds > 1 else img
            self.ax_tiff.imshow(disp, cmap="gray", origin="upper", aspect="equal",
                                extent=[-0.5, w - 0.5, h - 0.5, -0.5], vmin=vmin, vmax=vmax)
            if keep_view:
                self._restore_view(self.ax_tiff, xl, yl)
            self._tiff_img_dirty = False; self._tiff_pt_artists = []
        else:
            for a in self._tiff_pt_artists:
                try: a.remove()
                except Exception: pass
            self._tiff_pt_artists = []
        for i, pair in enumerate(self.point_pairs):
            if "tiff" in pair:
                dx, dy = pair["tiff"]                          # DISPLAY coords, drawn as-is
                ln, = self.ax_tiff.plot(dx, dy, "o", color=PT_TIFF, markersize=8,
                                        markeredgecolor="white", markeredgewidth=0.8, zorder=5)
                tx = self.ax_tiff.text(dx + 6, dy - 6, str(i + 1), color=PT_TIFF,
                                       fontsize=9, fontweight="bold", zorder=6)
                self._tiff_pt_artists.extend([ln, tx])
        self.canvas_tiff.draw_idle()

    def _refresh_tiff(self):
        self._tiff_img_dirty = True
        self._draw_tiff(keep_view=True)

    @staticmethod
    def _save_view(ax):
        return ax.get_xlim(), ax.get_ylim()

    @staticmethod
    def _restore_view(ax, xl, yl):
        if xl is not None:
            ax.set_xlim(xl); ax.set_ylim(yl)

    # ---------------- clicks (store DISPLAY coords, no flipping) ----------------

    def _on_click_mrc(self, event):
        if event.button != 1 or event.inaxes is not self.ax_mrc or _shift_held(event):
            return
        if self.mrc_image is None or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        for pair in self.point_pairs:
            if "mrc" not in pair:
                pair["mrc"] = (x, y); self._update_tree(); self._draw_mrc(keep_view=True)
                self.status_var.set("MRC point set.\nNow click matching TIFF point."); return
        self._draw_mrc(keep_view=True)
        self.point_pairs.append({"mrc": (x, y)}); self._update_tree(); self._draw_mrc(keep_view=True)
        self.status_var.set(f"MRC point #{len(self.point_pairs)} placed.\nNow click matching TIFF point.")

    def _on_click_tiff(self, event):
        if event.button != 1 or event.inaxes is not self.ax_tiff or _shift_held(event):
            return
        if self.tiff_stack is None or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        for pair in reversed(self.point_pairs):
            if "mrc" in pair and "tiff" not in pair:
                pair["tiff"] = (x, y); self._update_tree(); self._draw_tiff(keep_view=True)
                n = sum(1 for p in self.point_pairs if "mrc" in p and "tiff" in p)
                self.status_var.set(f"Pair complete!  {n} pair(s).\nAdd more or Apply."); return
        self.point_pairs.append({"tiff": (x, y)}); self._update_tree(); self._draw_tiff(keep_view=True)
        self.status_var.set(f"TIFF point #{len(self.point_pairs)} placed.\nNow click matching MRC point.")

    def _update_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, pair in enumerate(self.point_pairs):
            m = "({:.1f}, {:.1f})".format(*pair["mrc"]) if "mrc" in pair else "-"
            t = "({:.1f}, {:.1f})".format(*pair["tiff"]) if "tiff" in pair else "-"
            self.tree.insert("", "end", values=(i + 1, m, t))

    def _remove_last(self):
        if not self.point_pairs:
            self.status_var.set("No landmark pairs to remove."); return
        self.point_pairs.pop(); self._update_tree()
        self._draw_mrc(keep_view=True); self._draw_tiff(keep_view=True)
        self.status_var.set(f"Removed last point.\n{len(self.point_pairs)} remaining.")

    def _clear_points(self):
        self.point_pairs.clear(); self.warped_channels.clear(); self._update_tree()
        self._draw_mrc(keep_view=True); self._draw_tiff(keep_view=True)
        self.status_var.set("Points cleared.")

    def _on_flip_changed(self):
        # Persist the toggle state onto the displayed MRC dataclass.
        self._sync_flip_to_mrc()
        # Picks are stored in the display frame, so a flip change invalidates
        # them.  Clear pairs and any applied transform; re-pick after orienting.
        if self.point_pairs or self.warped_channels:
            self.point_pairs.clear(); self.warped_channels.clear(); self._update_tree()
            self._draw_mrc(keep_view=True)
        self._last_tform = None
        self._loaded_record = None
        self._set_display_rotation(0.0)     # rotation preview was tied to the record's flips
        self._refresh_tiff()
        self.status_var.set("Flip changed. Landmarks cleared -- orient first, then pick.")

    # ---------------- apply / warp ----------------

    def _apply_transform(self):
        if self.mrc_image is None or self.tiff_stack is None:
            messagebox.showwarning("Missing data", "Load both images first."); return

        def status_cb(msg):
            self.status_var.set(msg); self.update_idletasks()

        ttype = self.transform_var.get()
        fx, fy = bool(self.flip_x.get()), bool(self.flip_y.get())
        record = self._loaded_record
        n_pairs = sum(1 for p in self.point_pairs if "mrc" in p and "tiff" in p)
        save_dir = os.path.join(self.site_data.path, "transforms") if self.site_data.path else None

        try:
            if record is not None and n_pairs == 0:
                result = self.correlator.run_reapply(
                    record, self.tiff_stack, self.mrc_image.shape,
                    tiff_pixel_spacing_um=self._tiff_scale(),
                    mrc_pixel_spacing_um=self.mrc_pixel_spacing_um, status_cb=status_cb)
            elif record is not None:
                result = self.correlator.run_reapply_refine(
                    record, self._pairs_for_fit(), ttype, self.tiff_stack, self.mrc_image.shape,
                    flip_x=fx, flip_y=fy,
                    mrc_pixel_spacing_um=self.mrc_pixel_spacing_um,
                    tiff_pixel_spacing_um=self._tiff_scale(), status_cb=status_cb)
            else:
                result = self.correlator.run_fit_and_warp(
                    self._pairs_for_fit(), ttype, self.tiff_stack, self.mrc_image.shape,
                    flip_x=fx, flip_y=fy, status_cb=status_cb,
                    mrc_pixel_spacing_um=self.mrc_pixel_spacing_um,
                    tiff_pixel_spacing_um=self._tiff_scale(),
                    auto_save=True, save_dir=save_dir)
            rec = result["record"]
            self._loaded_record = rec
            self.site_data.set_registration(result, transform_type=ttype,
                                             flip_x=rec.flip_x, flip_y=rec.flip_y)
        except ValueError as e:
            messagebox.showwarning("Not enough points", str(e)); return
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Transform failed", str(e)); return

        self._last_tform = result["transform"]
        self.warped_channels = result["warped_channels"]
        C = len(self.warped_channels); mrc_h, mrc_w = self.mrc_image.shape
        fit_txt = result["fit_info"]["text"]
        H_t, W_t = self.tiff_stack.shape[-2:]                 # (Y, X) of a TIFF slice
        c = self._last_tform(np.array([[W_t / 2.0, H_t / 2.0]]))[0]
        overlay_center_px = (float(c[0]), float(c[1]))        # display-MRC montage pixels
        self.site_data.registration.overlay_center_px = overlay_center_px
        self.overlay_center_px = overlay_center_px            # convenience handle for the script
        self.status_var.set(f"{ttype.capitalize()} applied -- {C} ch warped.\n{fit_txt}\nClick Show Overlay.")
        messagebox.showinfo("Done",
            f"{ttype.capitalize()} from {result['n_pairs']} pairs.\n"
            f"{C} ch warped to grid ({mrc_w} x {mrc_h}).\n\n{fit_txt}\n\n"
            "If RMSE is large, the TIFF orientation probably doesn't match the "
            "MRC yet -- adjust the X/Y flips (which clears points), re-pick, and Apply.")

    # ---------------- export / import ----------------

    def _export_transform(self):
        if self._last_tform is None or self._loaded_record is None:
            messagebox.showwarning("No transform", "Apply a transform before exporting."); return
        save_dir = filedialog.askdirectory(title="Choose folder to save the transform")
        if not save_dir:
            return
        try:
            path = self.correlator.save_transform(self._loaded_record, save_dir=save_dir)
            self.status_var.set(f"Transform saved: {os.path.basename(path)}")
            messagebox.showinfo("Exported", f"Transform saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _import_transform(self):
        if self.mrc_image is None or self.tiff_stack is None:
            messagebox.showwarning("Missing data", "Load both the MRC and the TIFF first."); return
        path = filedialog.askopenfilename(title="Import transform",
            filetypes=[("Transforms", ("*.yaml", "*.yml", "*.csv", "*.txt")), ("All files", "*")])
        if not path:
            return
        try:
            record = self.correlator.load_transform(path)
            if record.transform_type in ("euclidean", "similarity", "affine", "projective"):
                self.transform_var.set(record.transform_type)
            self.flip_x.set(bool(record.flip_x))    # restore the stored TIFF orientation
            self.flip_y.set(bool(record.flip_y))
            self._loaded_record = record
            self.point_pairs.clear(); self._update_tree()
            self._refresh_tiff()
            self._apply_transform()      # re-apply path (no pairs -> coarse re-apply)
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Import error", str(e))

    def _update_scale_info(self):
        tps, mps = self._tiff_scale(), self.mrc_pixel_spacing_um
        t = f"TIF px {tps:.4f} um" if tps else "TIF px ?"
        m = f"MRC px {mps:.4f} um" if mps else "MRC px ?"
        s = f"scale {tps / mps:.2f} MRC px/TIF px" if (tps and mps) else "scale ?"
        self.scale_info_var.set(f"{t}   {m}   {s}")

    def _import_transform_rot_only(self):
        """Import a stored transform but keep ONLY its rotation and flips.

        - scale comes from the pixel sizes (TIFF um/px / MRC um/px), not from
          the stored fit;
        - the TIFF footprint is centred on the current MRC and warped there
          immediately (no landmarks needed);
        - the right panel shows the TIFF flipped + rotated as it lands on the
          MRC; the white rectangle on the MRC shows its size and rotation."""
        if self.mrc_image is None or self.tiff_stack is None:
            messagebox.showwarning("Missing data", "Load both the MRC and the TIFF first."); return
        path = filedialog.askopenfilename(title="Import transform (rotation + flips only)",
            filetypes=[("Transforms", ("*.yaml", "*.yml", "*.csv", "*.txt")), ("All files", "*")])
        if not path:
            return

        def status_cb(msg):
            self.status_var.set(msg); self.update_idletasks()

        try:
            record = self.correlator.load_transform(path)
            # Orientation comes from the record; lock the checkboxes so the
            # preview stays consistent (re-loading a TIFF re-enables them).
            self.flip_x.set(bool(record.flip_x))
            self.flip_y.set(bool(record.flip_y))
            for cb in (getattr(self, "flip_x_cb", None), getattr(self, "flip_y_cb", None)):
                if cb is not None:
                    cb.config(state="disabled")
            self.point_pairs.clear(); self._update_tree()

            tps = self._tiff_scale()
            if not tps:
                messagebox.showwarning("No pixel size",
                    "The TIFF/CZI metadata has no pixel size -- the stored fit "
                    "scale is used instead of the pixel-size ratio.")
            result = self.correlator.run_reapply(
                record, self.tiff_stack, self.mrc_image.shape,
                tiff_pixel_spacing_um=tps,
                mrc_pixel_spacing_um=self.mrc_pixel_spacing_um,
                status_cb=status_cb, center_on_mrc=True)

            rec = result["record"]
            self._loaded_record = rec
            self._last_tform = result["transform"]
            self.warped_channels = result["warped_channels"]
            ttype = rec.transform_type or "similarity"
            if ttype in ("euclidean", "similarity", "affine", "projective"):
                self.transform_var.set(ttype)
            self.site_data.set_registration(result, transform_type=ttype,
                                            flip_x=rec.flip_x, flip_y=rec.flip_y)

            rot = float(result["fit_info"].get("rotation_deg") or 0.0)
            self._set_display_rotation(rot)
            self._refresh_tiff()
            self._draw_mrc(keep_view=True)      # white footprint rectangle
            self._update_scale_info()

            mps = self.mrc_pixel_spacing_um
            scale_txt = (f"{tps / mps:.3f} MRC px/TIF px (px-size ratio)"
                         if (tps and mps) else "stored fit scale")
            self.status_var.set(
                f"Rot/flip import: rot {rot:+.2f} deg, flips "
                f"({bool(rec.flip_x)}, {bool(rec.flip_y)}), scale {scale_txt}.\n"
                "Footprint centred on MRC (white frame). Pick landmarks + Apply "
                "to refine the position.")
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Rot/flip import error", str(e))

    # ---------------- overlay + picker ----------------

    def _show_overlay(self):
        if self.mrc_image is None:
            messagebox.showwarning("No MRC", "Load MRC first."); return
        if not self.warped_channels:
            messagebox.showwarning("No data", "Apply a transform first."); return

        C = len(self.warped_channels)
        roles = self.channel_roles()

        # everything here is in the DISPLAY-MRC frame already; show directly
        mrc_disp = apply_bc(_mrc_to_display(_fast_ds(self.mrc_image)), self.bc_mrc.vmin, self.bc_mrc.vmax)
        chs_disp = []
        for idx, ch in enumerate(self.warped_channels):
            bc = self.bc_tiff_panel.bc(idx)
            vmin, vmax = (bc.vmin, bc.vmax) if bc else (0.0, 1.0)
            chs_disp.append(apply_bc(_fast_ds(ch), vmin, vmax))

        # Panels, in order: MRC | TEM+reflection | one per fluorescence channel
        # | full composite. The plan is shared with the save path below so the
        # files always match what is on screen.
        plan = overlay_panel_plan(roles, C)
        panels = [(mrc_disp if colors is None
                   else composite_overlay(mrc_disp, chs_disp, colors=colors),
                   title, is_gray)
                  for title, colors, is_gray in plan]

        ncols = len(panels)
        win = tk.Toplevel(self); win.title("Overlay result"); win.configure(bg=BG); win.minsize(600, 350)
        fig = Figure(figsize=(3.6 * ncols, 4.4), facecolor=BG)
        axes = [fig.add_subplot(1, ncols, i + 1) for i in range(ncols)]
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.04)

        def sa(ax, t):
            ax.set_facecolor(BG); ax.set_title(t, color=CYA, fontsize=8, pad=4); ax.axis("off")

        for ax, (img, title, is_gray) in zip(axes, panels):
            if is_gray:
                ax.imshow(img, cmap="gray", origin="upper", vmin=0, vmax=1)
            else:
                ax.imshow(img, origin="upper")
            sa(ax, title)

        canvas = FigureCanvasTkAgg(fig, master=win); canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        for ax in axes:
            PanZoomHandler(ax, canvas)

        btns = ttk.Frame(win, padding=(6, 0, 6, 6)); btns.pack(fill="x")
        has_stage = bool(self.mrc_current_pieces)

        def open_stage_picker():
            if not has_stage:
                messagebox.showwarning("No mdoc data", "Load MRC + mdoc to enable stage picking.", parent=win)
                return
            from clem_target_picking import CLEMPicker
            # tell the picker how the montage is displayed so its display<->true works
            
            mrc_summary = self._resolve_latest_mrc() or self.mrc_reader.build_montage_summary(self.mrc_file_path)
            mrc_summary.flip_x = MRCReader.MONTAGE_FLIP_X
            mrc_summary.flip_y = MRCReader.MONTAGE_FLIP_Y
            clem_picker = CLEMPicker(mrc_summary, self.tem, site_data=self.site_data)

            names = list(roles[:len(self.warped_channels)])
            n_z = self.site_data.tiff.num_z_slices if self.site_data.tiff is not None else 1

            # picker works in TRUE frame: un-flip the display-MRC arrays once
            mrc_true = self.mrc_image
            chans_true = [_mrc_display_to_true_2d(ch) for ch in self.warped_channels]
            fx, fy = bool(self.flip_x.get()), bool(self.flip_y.get())

            def warp_slice_true(c, z):
                disp = self.correlator.warp_slice(self.tiff_stack, c, z, self._last_tform,
                                                  self.mrc_image.shape, flip_x=fx, flip_y=fy)
                return _mrc_display_to_true_2d(disp)

            def warp_crop_true(c, z, x0, y0, cw):
                """One (channel, z) slice warped straight into a cw x cw window
                at (x0, y0) in TRUE montage pixels -- same result as cropping
                warp_slice_true, without warping the whole montage first.

                The montage is drawn flipped, so the TRUE window becomes this
                DISPLAY window; deriving it from the true origin (not the
                rounded centre) keeps the two paths exactly aligned.
                """
                H, W = self.mrc_image.shape[:2]
                dx0 = (W - cw - x0) if MRCReader.MONTAGE_FLIP_X else x0
                dy0 = (H - cw - y0) if MRCReader.MONTAGE_FLIP_Y else y0
                crop = self.correlator.warp_crop(
                    self.tiff_stack, c, z, self._last_tform, dx0, dy0, cw,
                    flip_x=fx, flip_y=fy, mrc_shape=self.mrc_image.shape)
                return _mrc_display_to_true_2d(crop)

            StagePickerWindow(win, clem_picker=clem_picker, mrc_reader=self.mrc_reader,
                              mrc_gray_true=mrc_true, channels_gray_true=chans_true,
                              channel_names=names, channel_roles=list(roles),
                              warp_slice_true=warp_slice_true,
                              warp_crop_true=warp_crop_true, n_z=n_z,
                              site_data=self.site_data, title="Stage Position Picker")

        ttk.Button(btns, text="Open Stage Picker", style="Mont.TButton",
                   command=open_stage_picker).pack(side="left", padx=(0, 6), pady=4)
        if not has_stage:
            ttk.Label(btns, text="(load MRC + mdoc to enable)", style="Sm.TLabel",
                      foreground=BG3).pack(side="left", pady=4)

        def save_overlay():
            base = filedialog.asksaveasfilename(title="Choose base filename", defaultextension=".tif",
                filetypes=[("TIFF", "*.tif"), ("PNG", "*.png"), ("All files", "*")])
            if not base:
                return
            root, ext = os.path.splitext(base); ext = (ext or ".tif").lower()
            use_png = ext == ".png"

            def to_u8(a):
                if a.ndim == 2:
                    a = np.stack([a] * 3, axis=-1)
                return (np.clip(a, 0, 1) * 255).astype(np.uint8)

            def write_panel(sfx, arr):
                out = root + sfx + ext
                if use_png:
                    from PIL import Image
                    Image.fromarray(to_u8(arr)).save(out)   # PNG is deflate already
                elif arr.ndim == 2:
                    tiff_write(out, (np.clip(arr, 0, 1) * 65535).astype(np.uint16), photometric="minisblack")
                else:
                    tiff_write(out, to_u8(arr), photometric="rgb")
                return os.path.basename(out)

            try:
                saved = []
                mrc_bc = apply_bc(_mrc_to_display(self.mrc_image), self.bc_mrc.vmin, self.bc_mrc.vmax)
                # Full-resolution versions of the panels shown on screen.
                full_bc = [apply_bc(ch, *((self.bc_tiff_panel.bc(i).vmin, self.bc_tiff_panel.bc(i).vmax)
                                          if self.bc_tiff_panel.bc(i) else (0.0, 1.0)))
                           for i, ch in enumerate(self.warped_channels)]

                # Same plan as the figure, at full resolution.
                for n, (title, colors, _is_gray) in enumerate(plan, start=1):
                    img = (mrc_bc if colors is None
                           else composite_overlay(mrc_bc, full_bc, colors=colors))
                    saved.append(write_panel(f"_{n:02d}_{_panel_slug(title)}", img))
                messagebox.showinfo("Saved", f"{len(saved)} file(s) written:\n" + "\n".join(saved))
            except Exception as e:
                messagebox.showerror("Save error", str(e))

        ttk.Button(btns, text="Save all panels", command=save_overlay,
                   style="Accent.TButton").pack(side="right", pady=4)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch the CLEM UI standalone (offline).")
    parser.add_argument("site_folder", nargs="?",
                        help="Site folder (MRC/mdoc, TIFF/CZI). Omit to open a chooser.")
    parser.add_argument("--coord-key", default="AlignedPieceCoordsVS")
    parser.add_argument("--milling-angle", type=float, default=0.0)
    parser.add_argument("--reuse-transform", action="store_true",
                        help="Re-apply the newest transform saved under "
                             "<site>/transforms on startup (locks the flips).")
    args = parser.parse_args()

    site_folder = args.site_folder
    if not site_folder:
        _root = tk.Tk(); _root.withdraw()
        site_folder = filedialog.askdirectory(title="Choose a site folder")
        _root.destroy()
    if not site_folder:
        raise SystemExit("No site folder selected.")

    site_folder = os.path.normpath(site_folder)
    experiment_root = os.path.dirname(site_folder) or site_folder

    mrc_reader = MRCReader(coord_key=args.coord_key, section=0)
    tem = TEMComm(path=experiment_root, mrc_reader=mrc_reader, offline=True)
    site_data = SiteDataSummary(site_id=os.path.basename(site_folder),
                                path=site_folder, milling_angle=args.milling_angle)
    RegistrationApp(mrc_reader=mrc_reader, site_data=site_data,
                    tem_communication=tem,
                    reuse_transform=args.reuse_transform).mainloop()

