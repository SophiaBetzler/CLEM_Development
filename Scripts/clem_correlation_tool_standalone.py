#!/usr/bin/env python3
"""
CLEM Registration Tool -- standalone
====================================

Offline CLEM correlation: load a TEM map, load a light-microscopy image, pick
landmarks, fit and apply a transform, inspect the overlay, then pick targets and
write image crops.  No SerialEM connection, no navigator, no stage movement.

Supported TEM maps
------------------
* Any complete .mrc -- opened directly, no assembly.  This is the usual case:
  an already-stitched map exported from Tomo5 / AutoTEM or anywhere else.
* SerialEM montages -- a multi-section .mrc plus a .mdoc carrying piece
  coordinates (RefinedPieceCoordinates / AlignedPieceCoordsVS / ... ), which
  are assembled on load.  One file may hold SEVERAL montages, one per
  navigator item; they are listed in the "Montage" dropdown (or with
  --list-sections) and assembled one at a time.  "Refine tiles" re-measures
  the piece positions from the overlaps instead of trusting the mdoc.

Pixel size comes from the mdoc when there is one, otherwise from the MRC
header (voxel_size).  It is only used to scale a re-applied transform, so a
missing value degrades gracefully.

Coordinate model
----------------
Everything the user sees and clicks is in the DISPLAY frame.  The map is drawn
with the montage flip convention (MONTAGE_FLIP_*); the LM image is drawn flipped
per the X/Y checkboxes.  Landmarks are stored exactly as clicked.  The fitted
transform maps display-LM pixels -> display-map pixels.

Dependencies:  numpy matplotlib mrcfile tifffile scikit-image
Optional:      aicspylibczi or czifile (for .czi), pillow (PNG export)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _require(module: str, pip_name: str = None):
    """Import a hard dependency, raising a readable ImportError (never sys.exit,
    so a caller can catch it)."""
    try:
        return __import__(module)
    except ImportError as exc:
        raise ImportError(
            f"Missing dependency '{module}'.  Install with:  "
            f"pip install {pip_name or module}") from exc


_require("mrcfile")
_require("tifffile")
_require("skimage", "scikit-image")

import mrcfile                                          # noqa: E402
import tifffile                                         # noqa: E402
from skimage.transform import (                         # noqa: E402
    ProjectiveTransform, estimate_transform, rotate as _sk_rotate, warp)

import tkinter as tk                                    # noqa: E402
from tkinter import filedialog, messagebox, ttk         # noqa: E402

import matplotlib                                       # noqa: E402
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg   # noqa: E402
from matplotlib.figure import Figure                    # noqa: E402


# --------------------------------------------------------------------------- #
# Theme and conventions
# --------------------------------------------------------------------------- #

BG = "#1e1e2e"; BG2 = "#313244"; BG3 = "#45475a"; FG = "#cdd6f4"
ACC = "#89b4fa"; ACC2 = "#a6e3a1"; RED = "#f38ba8"; CYA = "#89dceb"

CHANNEL_HEX = ["#00ff00", "#ff00ff", "#00ffff", "#ffff00", "#ff8000", "#8000ff"]
CHANNEL_COLORS = [(0., 1., 0.), (1., 0., 1.), (0., 1., 1.),
                  (1., 1., 0.), (1., .5, 0.), (.5, 0., 1.)]
CHANNEL_NAMES = ["green", "magenta", "cyan", "yellow", "orange", "purple"]

# --------------------------------------------------------------------------- #
# Channel roles
# --------------------------------------------------------------------------- #
# Channels are assigned roles rather than colours-by-position, because not every
# dataset carries every channel. The default is positional (ch0 reflection, then
# red, green, blue) and can be overridden per channel from the dropdown in the
# Brightness/Contrast panel.
#
# "reflection" is a reference channel, not a fluorophore: it is EXCLUDED from
# the fluorescence composite and instead gets its own map + reflection overlay,
# drawn in green.
CHANNEL_ROLES = ("reflection", "red", "green", "blue", "off")

ROLE_RGB = {"reflection": (0., 1., 0.),      # green, in its own overlay only
            "red":        (1., 0., 0.),
            "green":      (0., 1., 0.),
            "blue":       (0., 0., 1.),
            "off":        None}

ROLE_HEX = {"reflection": "#00ff00", "red": "#ff0000", "green": "#00ff00",
            "blue": "#0000ff", "off": "#555555"}

DEFAULT_ROLE_ORDER = ("reflection", "red", "green", "blue")
COMPOSITE_ROLES = ("red", "green", "blue")


def default_role(idx):
    """Positional default for channel `idx`; anything beyond blue starts off."""
    return DEFAULT_ROLE_ORDER[idx] if idx < len(DEFAULT_ROLE_ORDER) else "off"


def role_rgb(role):
    return ROLE_RGB.get(role)


def is_composite_role(role):
    return role in COMPOSITE_ROLES


def overlay_panel_plan(roles, n_channels):
    """Which overlay panels to build for a given set of channel roles.

    Returns a list of (title, colors, is_gray). `colors` is the per-channel
    colour list to hand to composite_overlay -- None entries drop that channel
    -- or None for the plain map panel.

    Copes with stacks that carry only some of the channels: a role that is not
    present simply produces no panel.
    """
    roles = list(roles)[:n_channels]
    roles += ["off"] * (n_channels - len(roles))

    plan = [("TEM map", None, True)]

    refl = [i for i, r in enumerate(roles) if r == "reflection"]
    if refl:
        plan.append(("map + reflection (green)",
                     [role_rgb("reflection") if i in refl else None
                      for i in range(n_channels)],
                     False))

    comp = [role_rgb(r) if is_composite_role(r) else None for r in roles]
    for i, col in enumerate(comp):
        if col is None:
            continue
        plan.append((f"map + Ch {i} ({roles[i]})",
                     [col if k == i else None for k in range(n_channels)],
                     False))

    if sum(c is not None for c in comp) > 1:
        plan.append(("composite", comp, False))
    return plan


# Compression for saved overlay panels and crop stacks. "zlib" (deflate) is
# lossless and readable by ImageJ/Fiji, tifffile and bioformats without extra
# packages. Set to None to write uncompressed; "lzw" is an alternative, and
# "zstd" is faster and smaller but needs the imagecodecs package.
TIFF_COMPRESSION = "zlib"


def tiff_write(path, data, **kwargs):
    """tifffile.imwrite with compression, tolerating tifffile API differences.

    Newer tifffile takes compression=; older versions took compress=; and a
    codec may be unavailable. Any of those falls back to uncompressed rather
    than losing the save.
    """
    if TIFF_COMPRESSION:
        for kw in ("compression", "compress"):
            try:
                return tifffile.imwrite(path, data, **{kw: TIFF_COMPRESSION}, **kwargs)
            except TypeError:
                continue            # this tifffile doesn't take that keyword
            except ValueError:
                break               # keyword is fine, codec is not
    return tifffile.imwrite(path, data, **kwargs)


def _panel_slug(title):
    """Filename-safe suffix for a panel title."""
    out = title.lower()
    for a, b in ((" + ", "_"), (" ", "_"), ("(", ""), (")", "")):
        out = out.replace(a, b)
    return out

PT_MAP = "#FF4444"; PT_LM = "#44AAFF"
ZOOM_FACTOR = 1.25

MONTAGE_FLIP_X = False      # how the assembled map is displayed
MONTAGE_FLIP_Y = True
LM_FLIP_X_DEFAULT = False   # starting state of the LM flip checkboxes
LM_FLIP_Y_DEFAULT = True

COORD_KEYS = ("RefinedPieceCoordinates", "AlignedPieceCoordsVS",
              "AlignedPieceCoords", "PieceCoordinates")

MAX_MOSAIC_PX = 8192        # assembled maps are binned to fit inside this
MRC_EXTS = (".mrc", ".rec", ".mrcs", ".map", ".st")


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

@dataclass
class Tile:
    piece_x_px: Optional[float] = None          # position in the assembled map
    piece_y_px: Optional[float] = None
    piece_x_stage_um: Optional[float] = None
    piece_y_stage_um: Optional[float] = None
    piece_z_stage_um: Optional[float] = None
    z_index: Optional[int] = None
    source_path: Optional[str] = None


@dataclass
class MontageSection:
    """One montage inside a multi-montage .mrc.

    A SerialEM montage file can hold several independent montages -- one per
    navigator item, for instance a grid square each -- stacked in the same
    .mrc. They are distinguished by the third component of the piece
    coordinates, and closed off in the mdoc by a [MontSection = N] block.
    """
    index: int
    n_pieces: int = 0
    z_values: list = field(default_factory=list)
    nav_label: Optional[str] = None
    stage_x_um: Optional[float] = None
    stage_y_um: Optional[float] = None
    stage_z_um: Optional[float] = None

    def describe(self):
        """One line for the montage dropdown."""
        bits = [f"#{self.index}"]
        if self.nav_label not in (None, ""):
            bits.append(f"nav {self.nav_label}")
        bits.append(f"{self.n_pieces} pieces")
        if self.stage_x_um is not None and self.stage_y_um is not None:
            bits.append(f"stage ({self.stage_x_um:.0f}, {self.stage_y_um:.0f}) um")
        return "  |  ".join(bits)


@dataclass
class MapSummary:
    """An assembled TEM map, whatever format it came from."""
    path: Optional[str] = None
    section: int = 0                            # which montage of the file
    sections: list = field(default_factory=list)    # every MontageSection found
    map_id: Optional[str] = None
    source: str = "unknown"                     # serialem | single
    image: Optional[Any] = None                 # assembled float32, TRUE frame
    image_height: Optional[int] = None          # of the assembled mosaic
    image_width: Optional[int] = None
    tile_height: Optional[int] = None
    tile_width: Optional[int] = None
    pixel_spacing_um: Optional[float] = None    # AFTER binning
    binning: int = 1
    magnification: Optional[float] = None
    stage_z_um: Optional[float] = None
    stage_tilt_deg: Optional[float] = None
    rotation_deg: Optional[float] = None
    tiles: list = field(default_factory=list)
    flip_x: bool = MONTAGE_FLIP_X
    flip_y: bool = MONTAGE_FLIP_Y
    refinement: Optional[Any] = None            # TileRefinement, if it ran
    info: str = ""


@dataclass
class LMImage:
    path: Optional[str] = None
    stack_czyx: Optional[Any] = None            # float32 (C, Z, Y, X) in [0, 1]
    pixel_spacing_um: Optional[float] = None
    num_channels: int = 0
    num_z: int = 0
    info: str = ""


@dataclass
class Pick:
    pick_id: str
    image_coord_x: float                        # TRUE map pixels
    image_coord_y: float


@dataclass
class TransformRecord:
    matrix: Optional[Any] = None                # 3x3 row-major
    transform_type: Optional[str] = None
    flip_x: bool = False
    flip_y: bool = True
    scale_x: Optional[float] = None
    scale_y: Optional[float] = None
    rotation_deg: Optional[float] = None
    rmse_px: Optional[float] = None
    n_pairs: Optional[int] = None
    mrc_shape: Optional[tuple] = None
    tiff_shape: Optional[tuple] = None
    pixel_spacing_um: Optional[float] = None            # map um/px
    tiff_pixel_spacing_um: Optional[float] = None       # LM um/px
    created_at: Optional[str] = None
    source_path: Optional[str] = None

    def to_dict(self):
        m = self.matrix
        return {
            "created_at": self.created_at,
            "transform_type": self.transform_type,
            "flip_x": bool(self.flip_x), "flip_y": bool(self.flip_y),
            "scale_x": self.scale_x, "scale_y": self.scale_y,
            "rotation_deg": self.rotation_deg, "rmse_px": self.rmse_px,
            "n_pairs": self.n_pairs,
            "mrc_shape": list(self.mrc_shape) if self.mrc_shape else None,
            "tiff_shape": list(self.tiff_shape) if self.tiff_shape else None,
            "pixel_spacing_um": self.pixel_spacing_um,
            "tiff_pixel_spacing_um": self.tiff_pixel_spacing_um,
            "matrix": (np.asarray(m, float).tolist() if m is not None else None),
        }

    @classmethod
    def from_dict(cls, d):
        def f(k):
            v = d.get(k)
            if v in (None, "", "None"):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def b(k, default=False):
            v = d.get(k)
            if isinstance(v, bool):
                return v
            if v in (None, "", "None"):
                return default
            return str(v).strip().lower() in ("1", "true", "yes", "y")

        def shape(k):
            v = d.get(k)
            if v in (None, "", "None"):
                return None
            if isinstance(v, (list, tuple)):
                return tuple(int(float(x)) for x in v)
            return tuple(int(float(x)) for x in re.split(r"[;,\s]+", str(v)) if x)

        n = d.get("n_pairs")
        return cls(
            matrix=d.get("matrix"),
            transform_type=d.get("transform_type") or None,
            flip_x=b("flip_x", False), flip_y=b("flip_y", True),
            scale_x=f("scale_x"), scale_y=f("scale_y"),
            rotation_deg=f("rotation_deg"), rmse_px=f("rmse_px"),
            n_pairs=(int(float(n)) if n not in (None, "", "None") else None),
            mrc_shape=shape("mrc_shape"), tiff_shape=shape("tiff_shape"),
            pixel_spacing_um=f("pixel_spacing_um"),
            tiff_pixel_spacing_um=f("tiff_pixel_spacing_um"),
            created_at=d.get("created_at"),
        )


# --------------------------------------------------------------------------- #
# Small image helpers
# --------------------------------------------------------------------------- #

def normalize(img):
    img = np.nan_to_num(np.asarray(img, dtype=np.float32))
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


def normalize_percentile(img, lo_pct=0.5, hi_pct=99.5):
    """Stretch to [0, 1] on percentiles rather than min/max.

    min/max lets one hot pixel or a dead-black border set the whole range and
    squash the real data into a narrow band, which is what made assembled
    montages look flat and oddly contrasted.
    """
    img = np.nan_to_num(np.asarray(img, dtype=np.float32))
    finite = img[np.isfinite(img)]
    if not finite.size:
        return np.zeros_like(img)
    lo, hi = np.percentile(finite, (lo_pct, hi_pct))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def auto_bc(img, percentiles=(1.0, 99.8)):
    img = np.nan_to_num(np.asarray(img, dtype=np.float32))
    s = img[np.isfinite(img)]
    s = s[s > 0] if np.any(s > 0) else s
    if not s.size:
        return np.zeros_like(img)
    lo, hi = np.percentile(s, percentiles)
    if hi <= lo:
        lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def flip_for_display(arr, flip_x=False, flip_y=False):
    if flip_x:
        arr = np.fliplr(arr)
    if flip_y:
        arr = np.flipud(arr)
    return arr


def map_to_display(arr):
    return flip_for_display(arr, MONTAGE_FLIP_X, MONTAGE_FLIP_Y)


display_to_map = map_to_display            # the flip is its own inverse


def apply_bc(img, vmin, vmax):
    if vmax <= vmin:
        return np.zeros_like(img)
    return np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)


def composite_overlay(base_bc, channels_bc, alpha=0.6, colors=None):
    """Composite channels over the map.

    `colors` is an optional per-channel list of RGB tuples; a None entry means
    "leave this channel out" (role 'off', or reflection when building the
    fluorescence composite). Without it, the legacy positional palette is used.
    """
    rgb = np.empty(base_bc.shape + (3,), dtype=np.float32)
    np.multiply(base_bc, alpha, out=rgb[..., 0])
    rgb[..., 1] = rgb[..., 0]; rgb[..., 2] = rgb[..., 0]
    for idx, ch in enumerate(channels_bc):
        if colors is not None:
            col = colors[idx] if idx < len(colors) else None
        else:
            col = CHANNEL_COLORS[idx % len(CHANNEL_COLORS)]
        if col is None:
            continue
        oma = 1.0 - ch
        for ci, cv in enumerate(col):
            np.multiply(rgb[..., ci], oma, out=rgb[..., ci])
            if cv > 0:
                rgb[..., ci] += ch * cv
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return rgb


def fast_ds(arr, max_px=2048):
    h, w = arr.shape[:2]
    f = max(1, max(h, w) // max_px)
    if f == 1:
        return arr
    return arr[::f, ::f] if arr.ndim == 2 else arr[::f, ::f, :]


def cosine_weight_map(h, w, feather_px):
    feather_px = max(1, int(feather_px))

    def ramp(n):
        r = np.ones(n, dtype=np.float32)
        f = min(feather_px, n // 2)
        if f > 0:
            t = np.linspace(0.0, np.pi / 2, f, dtype=np.float32)
            r[:f] = np.sin(t); r[-f:] = np.sin(t)[::-1]
        return r

    return np.outer(ramp(h), ramp(w)).astype(np.float32)


def _choose_binning(width, height, limit=MAX_MOSAIC_PX):
    """Smallest power-of-two binning that fits the mosaic inside `limit`."""
    b = 1
    while max(width, height) / b > limit:
        b *= 2
    return b


def assemble(placements, tile_h, tile_w, feather_px, binning=1, status_cb=None):
    """Blend tiles into one mosaic.

    placements : list of (loader, x_px, y_px); loader() returns a 2-D array.
                 Coordinates are in FULL-resolution mosaic pixels.
    Returns (mosaic float32, min_x, min_y) with coordinates already binned.
    """
    b = max(1, int(binning))
    th, tw = tile_h // b, tile_w // b
    xs = [p[1] for p in placements]; ys = [p[2] for p in placements]
    min_x, min_y = min(xs), min(ys)
    off = [(int(round((x - min_x) / b)), int(round((y - min_y) / b)))
           for _, x, y in placements]
    H = max(o[1] for o in off) + th + 1
    W = max(o[0] for o in off) + tw + 1

    canvas = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    wmap = cosine_weight_map(th, tw, max(1, feather_px // b))

    for i, ((loader, _, _), (cx, cy)) in enumerate(zip(placements, off)):
        if status_cb is not None:
            status_cb(f"Assembling tile {i + 1}/{len(placements)}...")
        img = loader()
        if b > 1:
            img = img[::b, ::b]
        img = np.asarray(img, dtype=np.float32)
        hh, ww = img.shape[:2]
        hh, ww = min(hh, th), min(ww, tw)
        canvas[cy:cy + hh, cx:cx + ww] += img[:hh, :ww] * wmap[:hh, :ww]
        weight[cy:cy + hh, cx:cx + ww] += wmap[:hh, :ww]

    valid = weight > 0
    canvas[valid] /= weight[valid]
    # Normalise ONCE, over the finished mosaic. Normalising each tile first
    # gives every tile its own gain -- a vacuum tile and a dense tile both get
    # stretched to full range -- which is what produced the patchy, tile-to-tile
    # contrast. Percentiles rather than min/max, so a hot pixel can't set the
    # scale for the whole montage.
    return normalize_percentile(canvas), float(min_x), float(min_y)


# --------------------------------------------------------------------------- #
# Tile position refinement
# --------------------------------------------------------------------------- #
# The mdoc's piece coordinates are where SerialEM believes each tile sits.
# RefinedPieceCoordinates / AlignedPieceCoords* are already the result of
# SerialEM's own alignment and are usually good. Raw PieceCoordinates are
# nominal stage positions and can be off by tens of pixels, which shows up as
# duplicated or broken features at the seams.
#
# This measures the true offset between every overlapping pair of tiles by
# phase correlation of their overlap region, then solves one least-squares
# problem for a set of positions consistent with all of those measurements at
# once. Pairwise shifts alone would not be consistent -- going around a loop of
# four tiles would not return to the start -- so the global solve matters.
#
# Nothing is written to disk: the refined coordinates affect the assembled
# mosaic for this session only, and the mdoc is never touched.

REFINE_MAX_TILE_PX = 1024       # tiles are binned to about this for matching
REFINE_MEMORY_MB = 256          # ceiling on the cached binned tiles
REFINE_MIN_OVERLAP_PX = 16      # smaller overlaps carry too little signal
REFINE_MIN_CORR = 0.25          # Pearson r below this -> the pair is rejected
REFINE_ANCHOR_WEIGHT = 0.05     # pull towards the nominal positions


@dataclass
class TileRefinement:
    """What refinement did, for reporting back to the user."""
    applied: bool = False
    n_tiles: int = 0
    n_pairs: int = 0                # overlapping pairs examined
    n_used: int = 0                 # pairs that survived into the final fit
    n_rejected: int = 0             # matched, then thrown out as inconsistent
    binning: int = 1                # scale the matching ran at
    median_shift_px: float = 0.0    # correction vs the mdoc, full-res px
    max_shift_px: float = 0.0
    rms_residual_px: float = 0.0    # disagreement left between pairs
    reason: str = ""

    @property
    def text(self):
        if not self.applied:
            return f"tile refinement skipped ({self.reason})"
        dropped = f", {self.n_rejected} dropped" if self.n_rejected else ""
        return (f"tiles refined: {self.n_used}/{self.n_pairs} seams matched"
                f"{dropped}, correction median {self.median_shift_px:.1f} px, "
                f"max {self.max_shift_px:.1f} px, "
                f"residual {self.rms_residual_px:.1f} px rms")


def _tukey(n, alpha=0.25):
    """Taper that leaves the middle of the patch alone.

    A patch has hard edges, and an FFT treats them as a discontinuity that
    dominates the correlation. A full Hann window would fix that but throw away
    most of the patch; a Tukey taper only touches the outer `alpha` fraction.
    """
    w = np.ones(int(n), dtype=np.float32)
    m = int(np.floor(alpha * (n - 1) / 2.0))
    if m > 0:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(m + 1) / m)).astype(np.float32)
        w[:m + 1] = ramp
        w[n - m - 1:] = ramp[::-1]
    return w


def _parabolic(cm, c0, cp):
    """Sub-pixel peak offset from three samples straddling the maximum."""
    den = cm - 2.0 * c0 + cp
    if den == 0:
        return 0.0
    return float(np.clip(0.5 * (cm - cp) / den, -1.0, 1.0))


def phase_shift(ref, mov, max_shift=None):
    """(dy, dx) that lines `mov` up with `ref`:  ref(y, x) ~ mov(y - dy, x - dx).

    Phase correlation, so it keys on structure rather than absolute intensity
    and is unbothered by the two tiles having different exposure or gain.
    Returns None when the patches are unusable.
    """
    a = np.asarray(ref, dtype=np.float32)
    b = np.asarray(mov, dtype=np.float32)
    if a.shape != b.shape or min(a.shape) < 8:
        return None
    a = a - a.mean(); b = b - b.mean()
    if not np.any(a) or not np.any(b):
        return None                     # flat patch: nothing to align to

    win = np.outer(_tukey(a.shape[0]), _tukey(a.shape[1]))
    A = np.fft.rfft2(a * win)
    B = np.fft.rfft2(b * win)
    R = A * np.conj(B)
    R /= (np.abs(R) + 1e-12)            # keep the phase, discard the magnitude
    corr = np.fft.irfft2(R, s=a.shape)

    H, W = corr.shape
    sy = np.fft.fftfreq(H) * H          # signed shift for each index
    sx = np.fft.fftfreq(W) * W
    if max_shift is not None:
        allowed = ((np.abs(sy)[:, None] <= max_shift)
                   & (np.abs(sx)[None, :] <= max_shift))
        if not allowed.any():
            return None
        corr = np.where(allowed, corr, -1e30)

    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    if corr[iy, ix] <= -1e29:
        return None
    dy = sy[iy] + _parabolic(corr[(iy - 1) % H, ix], corr[iy, ix],
                             corr[(iy + 1) % H, ix])
    dx = sx[ix] + _parabolic(corr[iy, (ix - 1) % W], corr[iy, ix],
                             corr[iy, (ix + 1) % W])
    return float(dy), float(dx)


def shifted_correlation(a, b, dy, dx):
    """Pearson r between `a` and `b` shifted by (dy, dx), over what overlaps.

    Used as the accept/reject test: a phase correlation always returns a peak,
    even for two patches of featureless vacuum, so the shift it proposes has to
    be checked against the actual pixels before it is trusted.
    """
    dy, dx = int(round(dy)), int(round(dx))
    H, W = a.shape
    y0, y1 = max(0, dy), min(H, H + dy)
    x0, x1 = max(0, dx), min(W, W + dx)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return 0.0
    pa = a[y0:y1, x0:x1].astype(np.float64)
    pb = b[y0 - dy:y1 - dy, x0 - dx:x1 - dx].astype(np.float64)
    pa = pa - pa.mean(); pb = pb - pb.mean()
    den = math.sqrt(float((pa * pa).sum()) * float((pb * pb).sum()))
    return float((pa * pb).sum() / den) if den > 0 else 0.0


def overlapping_pairs(origins, tile_w, tile_h, min_overlap=REFINE_MIN_OVERLAP_PX):
    """Indices of tile pairs that share enough area to be worth matching."""
    pairs = []
    for i in range(len(origins)):
        xi, yi = origins[i]
        for j in range(i + 1, len(origins)):
            xj, yj = origins[j]
            ox = min(xi + tile_w, xj + tile_w) - max(xi, xj)
            oy = min(yi + tile_h, yj + tile_h) - max(yi, yj)
            if ox >= min_overlap and oy >= min_overlap:
                pairs.append((i, j, ox, oy))
    return pairs


def solve_positions(n, measurements, anchor_weight=REFINE_ANCHOR_WEIGHT):
    """Least-squares tile corrections from pairwise offsets.

    `measurements` is a list of (i, j, dy, dx, weight) meaning "tile j sits
    (dy, dx) away from where the mdoc puts it, relative to tile i". Solving all
    of them together produces one consistent set of positions; the anchor rows
    tie the answer to the nominal coordinates, which both fixes the otherwise
    free global translation and keeps any tile that matched nothing where the
    mdoc put it.
    """
    rows = len(measurements) + n
    A = np.zeros((rows, n), dtype=np.float64)
    by = np.zeros(rows, dtype=np.float64)
    bx = np.zeros(rows, dtype=np.float64)
    for r, (i, j, dy, dx, w) in enumerate(measurements):
        A[r, j] = w; A[r, i] = -w
        by[r] = w * dy; bx[r] = w * dx
    for k in range(n):
        A[len(measurements) + k, k] = anchor_weight
    ey = np.linalg.lstsq(A, by, rcond=None)[0]
    ex = np.linalg.lstsq(A, bx, rcond=None)[0]
    return ex, ey


def _choose_refine_binning(tile_w, tile_h, n_tiles, assembly_binning=1):
    """Scale to match at: small enough to be quick, no finer than the output."""
    b = 1
    while max(tile_w, tile_h) / b > REFINE_MAX_TILE_PX:
        b *= 2
    while n_tiles * (tile_w / b) * (tile_h / b) * 4 / 1e6 > REFINE_MEMORY_MB:
        b *= 2
    # Refining finer than the mosaic is drawn at cannot change a single pixel.
    return max(1, b, int(assembly_binning))


def refine_tile_positions(coords, load_tile, tile_w, tile_h,
                          assembly_binning=1, max_shift_px=None,
                          min_corr=REFINE_MIN_CORR, status_cb=None):
    """Correct the mdoc's piece coordinates by matching overlapping tiles.

    coords     : [(x, y)] nominal top-left of each tile, full-res pixels
    load_tile  : load_tile(index) -> 2-D array for that tile
    Returns (refined_coords, TileRefinement).  On any failure the nominal
    coordinates come back unchanged, with the reason recorded.
    """
    n = len(coords)
    info = TileRefinement(n_tiles=n)
    if n < 2:
        info.reason = "only one tile"
        return list(coords), info

    rb = _choose_refine_binning(tile_w, tile_h, n, assembly_binning)
    info.binning = rb
    tw, th = int(tile_w // rb), int(tile_h // rb)
    if min(tw, th) < 16:
        info.reason = "tiles too small to match"
        return list(coords), info

    # Whole-pixel origins at the matching scale; the fraction they drop is
    # recovered by the measurement itself, since each patch is cut relative to
    # its own rounded origin.
    origins = [(int(round(x / rb)), int(round(y / rb))) for x, y in coords]
    pairs = overlapping_pairs(origins, tw, th)
    info.n_pairs = len(pairs)
    if not pairs:
        info.reason = "no tiles overlap"
        return list(coords), info

    if max_shift_px is None:
        max_shift = max(8.0, 0.25 * min(tw, th))
    else:
        max_shift = max(1.0, float(max_shift_px) / rb)

    cache = {}

    def binned(i):
        if i not in cache:
            img = np.asarray(load_tile(i), dtype=np.float32)
            cache[i] = img[::rb, ::rb] if rb > 1 else img
        return cache[i]

    max_area = max(ox * oy for _i, _j, ox, oy in pairs)
    measurements = []
    for k, (i, j, ox, oy) in enumerate(pairs):
        if status_cb is not None:
            status_cb(f"Matching seam {k + 1}/{len(pairs)}...")
        xi, yi = origins[i]; xj, yj = origins[j]
        x0, y0 = max(xi, xj), max(yi, yj)
        x1, y1 = min(xi + tw, xj + tw), min(yi + th, yj + th)
        pi = binned(i)[y0 - yi:y1 - yi, x0 - xi:x1 - xi]
        pj = binned(j)[y0 - yj:y1 - yj, x0 - xj:x1 - xj]
        if pi.shape != pj.shape or min(pi.shape) < 8:
            continue
        est = phase_shift(pi, pj, max_shift=max_shift)
        if est is None:
            continue
        dy, dx = est
        r = shifted_correlation(pi, pj, dy, dx)
        if r < min_corr:
            continue                    # featureless or mismatched seam
        # Weight by how well it matched AND how much area it matched over: a
        # corner-to-corner diagonal shares a fraction of the pixels an edge
        # seam does, and is correspondingly easier to get wrong.
        weight = float(r) * math.sqrt((ox * oy) / max_area)
        measurements.append((i, j, dy, dx, weight))

    if not measurements:
        info.n_used = 0
        info.reason = (f"no seam correlated above r={min_corr:g} "
                       f"({len(pairs)} tried)")
        return list(coords), info

    def _residuals(ex, ey, ms):
        return np.array([math.hypot((ey[j] - ey[i]) - dy, (ex[j] - ex[i]) - dx)
                         for i, j, dy, dx, _w in ms])

    ex, ey = solve_positions(n, measurements)

    # One seam that locked onto the wrong feature drags every tile it touches.
    # It shows up as a residual far above the rest, so drop the outliers and
    # solve again from what is left. A median-based cut-off, because the mean
    # is set by the very measurements being looked for.
    resid = _residuals(ex, ey, measurements)
    if len(measurements) > 3:
        cut = max(1.0, 3.0 * float(np.median(resid)))
        kept = [m for m, rr in zip(measurements, resid) if rr <= cut]
        if 0 < len(kept) < len(measurements):
            info.n_rejected = len(measurements) - len(kept)
            measurements = kept
            ex, ey = solve_positions(n, measurements)
            resid = _residuals(ex, ey, measurements)

    info.n_used = len(measurements)
    # How much the pairwise measurements still disagree after the global fit.
    # A large residual means the tiles cannot all be placed consistently --
    # stage drift or distortion, not something a rigid shift can fix.
    info.rms_residual_px = float(np.sqrt(np.mean(resid ** 2))) * rb

    mag = np.hypot(ex, ey) * rb
    info.median_shift_px = float(np.median(mag))
    info.max_shift_px = float(np.max(mag))
    info.applied = True

    refined = [((origins[i][0] + ex[i]) * rb, (origins[i][1] + ey[i]) * rb)
               for i in range(n)]
    return refined, info


# --------------------------------------------------------------------------- #
# Reader: SerialEM montage (.mrc + .mdoc)
# --------------------------------------------------------------------------- #

def _coerce(val_str):
    parts = val_str.split()
    if not parts:
        return val_str
    try:
        nums = [int(p) if re.fullmatch(r"-?\d+", p) else float(p) for p in parts]
        return nums[0] if len(nums) == 1 else nums
    except ValueError:
        return val_str.strip()


def parse_mdoc(mdoc_path):
    global_info, pieces = {}, []
    current, ctype = global_info, "global"
    with open(mdoc_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"\[ZValue\s*=\s*(\d+)\]", line)
            if m:
                current = {"ZValue": int(m.group(1))}
                pieces.append(current); ctype = "zvalue"; continue
            if line.startswith("["):
                ctype = "text"; continue
            if ctype == "text":
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                current[key.strip()] = _coerce(val.strip())
    for p in pieces:
        for k in COORD_KEYS:
            p.setdefault(k, None)
    return global_info, pieces


def find_mdoc(mrc_path):
    mrc_path = os.fspath(mrc_path)
    directory = os.path.dirname(mrc_path) or "."
    stem, ext = os.path.splitext(mrc_path)
    for c in (mrc_path + ".mdoc", stem + ".mdoc",
              stem + ext.replace(".", "_") + ".mdoc"):
        if os.path.isfile(c) and os.path.getsize(c) > 0:
            return c
    base = os.path.basename(stem)
    try:
        for fn in os.listdir(directory):
            full = os.path.join(directory, fn)
            if (fn.lower().endswith(".mdoc") and fn.startswith(base)
                    and os.path.getsize(full) > 0):
                return full
    except OSError:
        pass
    return None


def _scalar(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0] if len(value) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pick_coord_key(pieces, preferred=None):
    """First coordinate field populated for every piece, preferring `preferred`."""
    order = ([preferred] if preferred else []) + [k for k in COORD_KEYS if k != preferred]
    for k in order:
        if k and pieces and all(p.get(k) is not None for p in pieces):
            return k
    return None


def section_of(piece, key):
    """Which montage a piece belongs to: the 3rd component of its coordinates."""
    c = piece.get(key)
    return int(c[2]) if isinstance(c, (list, tuple)) and len(c) >= 3 else 0


def group_sections(pieces, key):
    """{section index: [pieces]}, in file order."""
    out = {}
    for p in pieces:
        out.setdefault(section_of(p, key), []).append(p)
    return out


def summarize_sections(pieces, key):
    """Describe every montage in the file, without touching the .mrc."""
    sections = []
    for idx, group in sorted(group_sections(pieces, key).items()):
        xs = [float(p["StagePosition"][0]) for p in group
              if isinstance(p.get("StagePosition"), (list, tuple))]
        ys = [float(p["StagePosition"][1]) for p in group
              if isinstance(p.get("StagePosition"), (list, tuple))]
        zs = [z for z in (_scalar(p.get("StageZ", p.get("Z"))) for p in group)
              if z is not None]
        label = next((p.get("NavigatorLabel") for p in group
                      if p.get("NavigatorLabel") not in (None, "")), None)
        sections.append(MontageSection(
            index=idx, n_pieces=len(group),
            z_values=[p.get("ZValue") for p in group],
            nav_label=(str(label) if label is not None else None),
            stage_x_um=(float(np.median(xs)) if xs else None),
            stage_y_um=(float(np.median(ys)) if ys else None),
            stage_z_um=(float(np.median(zs)) if zs else None)))
    return sections


def list_montage_sections(target, coord_key=None):
    """Every montage in the .mrc beside `target`; [] if it isn't a montage.

    Cheap: reads only the mdoc. Use this to offer the user a choice before
    committing to assembling one of them.
    """
    mdoc_path = find_mdoc(resolve_map_path(target))
    if mdoc_path is None:
        return []
    _global_info, pieces = parse_mdoc(mdoc_path)
    key = pick_coord_key(pieces, coord_key)
    if key is None:
        return []
    return summarize_sections(pieces, key)


def read_serialem_montage(mrc_path, coord_key=None, section=0, status_cb=None,
                          refine=False, max_shift_px=None,
                          min_corr=REFINE_MIN_CORR):
    mrc_path = os.fspath(mrc_path)
    mdoc_path = find_mdoc(mrc_path)
    if mdoc_path is None:
        raise FileNotFoundError(f"No .mdoc beside {os.path.basename(mrc_path)}.")

    global_info, pieces_all = parse_mdoc(mdoc_path)
    if not pieces_all:
        raise ValueError("No [ZValue] blocks in the mdoc.")

    key = pick_coord_key(pieces_all, coord_key)
    if key is None:
        raise KeyError(
            "No piece-coordinate field is populated for every section "
            f"(looked for {COORD_KEYS}).  This mdoc has "
            f"{len(pieces_all)} sections -- if they carry TiltAngle it is a "
            "tilt series, not a montage.")

    ps_ang = _scalar(global_info.get("PixelSpacing")) or 10000.0
    pixel_spacing_um = ps_ang / 10000.0

    img_size = global_info.get("ImageSize", [4096, 4096])
    if isinstance(img_size, (int, float)):
        tile_w = tile_h = int(img_size)
    else:
        tile_w, tile_h = int(img_size[0]), int(img_size[1])

    ps = global_info.get("PieceSpacing", [tile_w - 410, tile_h - 410])
    if isinstance(ps, (int, float)):
        ps_x = ps_y = int(ps)
    else:
        ps_x, ps_y = int(ps[0]), int(ps[1])
    feather = max(1, min(tile_w - ps_x, tile_h - ps_y))

    # A montage file may hold several montages. Assemble exactly one of them;
    # the rest are described in the summary so a caller can offer the choice.
    groups = group_sections(pieces_all, key)
    section_infos = summarize_sections(pieces_all, key)
    section = 0 if section is None else int(section)
    if section not in groups:
        wanted, section = section, sorted(groups)[0]
        print(f"[WARN] montage {wanted} is not in this file "
              f"(it has {sorted(groups)}); loading {section} instead.")
    pieces = groups[section]
    if len(groups) > 1:
        print(f"[INFO] {os.path.basename(mrc_path)} holds {len(groups)} montages; "
              f"assembling #{section} ({len(pieces)} pieces).")

    coords = [(float(p[key][0]), float(p[key][1])) for p in pieces]
    min_x = min(c[0] for c in coords); min_y = min(c[1] for c in coords)
    coords = [(c[0] - min_x, c[1] - min_y) for c in coords]

    full_w = int(max(c[0] for c in coords)) + tile_w
    full_h = int(max(c[1] for c in coords)) + tile_h
    binning = _choose_binning(full_w, full_h)
    if binning > 1:
        print(f"[INFO] Mosaic {full_w}x{full_h} px -> binning by {binning}")

    # mmap so only this montage's tiles are read. mrcfile.open() pulls the
    # WHOLE file into memory in one call, which for a nine-montage file is
    # ~80 tiles when nine are wanted -- and on a network share it can fail
    # outright. mmap pages in each tile as assemble() asks for it.
    try:
        _mrc_ctx = mrcfile.mmap(mrc_path, mode="r", permissive=True)
    except Exception as exc:
        print(f"[INFO] mmap unavailable for {os.path.basename(mrc_path)} "
              f"({type(exc).__name__}: {exc}); falling back to a full read.")
        _mrc_ctx = mrcfile.open(mrc_path, mode="r", permissive=True)

    refinement = TileRefinement(n_tiles=len(pieces), reason="not requested")

    with _mrc_ctx as mrc:
        data = mrc.data
        if data is None:
            raise ValueError("MRC contains no image data.")
        # Raw tile values -- do NOT normalise per tile. assemble() normalises
        # the finished mosaic once; per-tile normalisation would give each tile
        # its own gain and produce visible tile-to-tile contrast steps.
        if data.ndim == 2:
            def _tile(_z):
                return np.asarray(data, np.float32)
            usable = list(pieces)
        else:
            n_frames = data.shape[0]

            def _tile(z):
                return np.asarray(data[z], np.float32)
            usable = [p for p in pieces if p["ZValue"] < n_frames]
            if len(usable) < len(pieces):
                print(f"[WARN] {len(pieces) - len(usable)} piece(s) of montage "
                      f"{section} point past the end of the .mrc; skipping them.")
        if not usable:
            raise ValueError(f"Montage {section} has no pieces inside the .mrc.")

        keep = {id(p) for p in usable}
        coords = [c for p, c in zip(pieces, coords) if id(p) in keep]
        pieces = usable

        if refine:
            # Must happen here: the tiles are needed as pixels, and with mmap
            # they are only valid while the file is open.
            coords, refinement = refine_tile_positions(
                coords, lambda i: _tile(pieces[i]["ZValue"]), tile_w, tile_h,
                assembly_binning=binning, max_shift_px=max_shift_px,
                min_corr=min_corr, status_cb=status_cb)
            print(f"[INFO] {refinement.text}")

        placements = [((lambda z=p["ZValue"]: _tile(z)), c[0], c[1])
                      for p, c in zip(pieces, coords)]
        # Inside the with-block: mmap'd data is invalid once the file closes.
        mosaic, _mx, _my = assemble(placements, tile_h, tile_w, feather,
                                    binning=binning, status_cb=status_cb)

    global_z = _scalar(global_info.get("StageZ", global_info.get("Z")))
    tiles = []
    for p, c in zip(pieces, coords):
        sp = p.get("StagePosition")
        sx = float(sp[0]) if isinstance(sp, (list, tuple)) and len(sp) >= 2 else None
        sy = float(sp[1]) if isinstance(sp, (list, tuple)) and len(sp) >= 2 else None
        sz = (float(sp[2]) if isinstance(sp, (list, tuple)) and len(sp) >= 3
              else _scalar(p.get("StageZ", p.get("Z"))) or global_z)
        tiles.append(Tile(piece_x_px=c[0] / binning, piece_y_px=c[1] / binning,
                          piece_x_stage_um=sx, piece_y_stage_um=sy,
                          piece_z_stage_um=sz, z_index=p.get("ZValue"),
                          source_path=mrc_path))

    angles = [_scalar(p.get("RotationAngle")) or 0.0 for p in pieces]
    stage_z = next((t.piece_z_stage_um for t in tiles
                    if t.piece_z_stage_um is not None), global_z)
    map_id = Path(mrc_path).stem
    if len(section_infos) > 1:
        map_id = f"{map_id}[{section}]"      # keep saved files distinguishable
    return MapSummary(
        path=mrc_path, section=section, sections=section_infos,
        map_id=map_id, source="serialem",
        image=mosaic, image_height=mosaic.shape[0], image_width=mosaic.shape[1],
        tile_height=tile_h // binning, tile_width=tile_w // binning,
        pixel_spacing_um=pixel_spacing_um * binning, binning=binning,
        magnification=_scalar(global_info.get("Magnification")),
        stage_z_um=stage_z,
        stage_tilt_deg=_scalar(pieces[0].get("TiltAngle")),
        rotation_deg=float(np.median(angles)) if angles else 0.0,
        tiles=tiles, refinement=refinement,
        info=(f"SerialEM montage {section} of {len(section_infos)}: "
              f"{len(tiles)} pieces, {key}, bin {binning}, "
              f"{pixel_spacing_um * binning:.6f} um/px"
              + (f"\n{refinement.text}" if refine else "")))


def read_single_mrc(mrc_path, status_cb=None):
    """Any .mrc as a plain image -- middle section if it is a stack."""
    mrc_path = os.fspath(mrc_path)
    with mrcfile.open(mrc_path, mode="r", permissive=True) as mrc:
        data = mrc.data
        if data is None:
            raise ValueError("MRC contains no image data.")
        note = ""
        if data.ndim == 3:
            mid = data.shape[0] // 2
            note = f" (section {mid} of {data.shape[0]})"
            data = data[mid]
        elif data.ndim != 2:
            raise ValueError(f"Unsupported MRC ndim={data.ndim}")
        img = normalize(np.asarray(data, np.float32))
        vox = float(mrc.voxel_size.x) if mrc.voxel_size.x else 0.0
    ps_um = vox / 10000.0 if vox else None
    return MapSummary(
        path=mrc_path, map_id=Path(mrc_path).stem, source="single",
        image=img, image_height=img.shape[0], image_width=img.shape[1],
        tile_height=img.shape[0], tile_width=img.shape[1],
        pixel_spacing_um=ps_um,
        info=f"single image {img.shape[1]}x{img.shape[0]}{note}")


def resolve_map_path(target):
    """The .mrc a user's choice refers to (they may have picked the .mdoc)."""
    p = Path(target)
    if p.is_dir():
        raise IsADirectoryError(
            f"{p} is a folder -- point this at an .mrc (or .mdoc) file.")

    if p.suffix.lower() == ".mdoc":
        stem = p.with_suffix("")
        if stem.suffix.lower() in MRC_EXTS and stem.is_file():
            return stem
        for ext in MRC_EXTS:
            if p.with_suffix(ext).is_file():
                return p.with_suffix(ext)
    return p


def load_map(target, coord_key=None, section=0, status_cb=None, refine=False):
    """Load a TEM map.  With an .mdoc beside it the montage is assembled from
    the piece coordinates; otherwise the MRC is used directly as one image.

    `section` selects which montage to assemble when the file holds more than
    one; list_montage_sections() reports what is available.
    `refine` re-measures the tile positions from the image data instead of
    trusting the mdoc -- see refine_tile_positions().
    """
    p = resolve_map_path(target)

    if find_mdoc(p) is not None:
        try:
            return read_serialem_montage(p, coord_key, section, status_cb,
                                         refine=refine)
        except (KeyError, ValueError) as exc:
            print(f"[WARN] {exc}\n[INFO] Falling back to a single image.")
    return read_single_mrc(p, status_cb=status_cb)


# --------------------------------------------------------------------------- #
# Reader: light microscopy (OME-TIFF / plain TIFF / CZI)
# --------------------------------------------------------------------------- #

_TO_UM = {"m": 1e6, "meter": 1e6, "metre": 1e6, "cm": 1e4, "centimeter": 1e4,
          "centimetre": 1e4, "mm": 1e3, "millimeter": 1e3, "millimetre": 1e3,
          "um": 1.0, "µm": 1.0, "micron": 1.0, "microns": 1.0,
          "micrometer": 1.0, "micrometre": 1.0, "nm": 1e-3, "nanometer": 1e-3,
          "nanometre": 1e-3, "inch": 25400.0, "in": 25400.0}


def _to_um(value, unit):
    if value is None or unit is None:
        return None
    f = _TO_UM.get(str(unit).strip().lower())
    return float(value) * f if f is not None else None


def read_tiff_pixel_spacing_um(path):
    """XY pixel spacing in um/px for an OME-TIFF or ImageJ/plain TIFF."""
    with tifffile.TiffFile(os.fspath(path)) as tf:
        ome = tf.ome_metadata
        if ome:
            mx = re.search(r'PhysicalSizeX\s*=\s*"([^"]+)"', ome)
            mu = re.search(r'PhysicalSizeXUnit\s*=\s*"([^"]+)"', ome)
            if mx:
                r = _to_um(float(mx.group(1)), mu.group(1) if mu else "um")
                if r:
                    return r
        ij = tf.imagej_metadata or {}
        unit = ij.get("unit")
        page = tf.pages[0]
        xres = page.tags.get("XResolution")
        if xres is not None:
            raw = xres.value
            if isinstance(raw, (tuple, list)):
                num, den = raw[0], raw[1]
                px_per_unit = (num / den) if den else 0.0
            else:
                px_per_unit = float(raw)
            if px_per_unit:
                if unit is None:
                    ru = page.tags.get("ResolutionUnit")
                    unit = {2: "inch", 3: "cm"}.get(int(ru.value) if ru else 1)
                r = _to_um(1.0 / px_per_unit, unit)
                if r:
                    return r
    print(f"[WARN] {Path(path).name}: no usable pixel size in the TIFF tags "
          "(no OME PhysicalSizeX, no ImageJ 'unit'). Scale-aware re-apply "
          "will be skipped.")
    return None


def _to_czyx(data, axes):
    """Collapse anything that is not C/Z/Y/X, then order as (C, Z, Y, X)."""
    axes = axes.upper()
    for dim in list(axes):
        if dim not in "CZYX":
            idx = axes.index(dim)
            take = data.shape[idx] // 2 if data.shape[idx] > 1 else 0
            if data.shape[idx] > 1:
                print(f"[WARN] collapsing axis '{dim}' (size {data.shape[idx]}) "
                      f"by taking index {take}")
            data = data.take(take, axis=idx)
            axes = axes.replace(dim, "", 1)
    while data.ndim < 2:
        data = data[np.newaxis]; axes = "Y" + axes
    for dim in ("C", "Z"):
        if dim not in axes:
            data = data[np.newaxis]; axes = dim + axes
    data = np.transpose(data, [axes.index(d) for d in "CZYX"])
    out = np.zeros(data.shape, dtype=np.float32)
    for c in range(data.shape[0]):
        out[c] = normalize(data[c])
    return out


def load_lm(path):
    """Load an OME-TIFF, plain/ImageJ TIFF or CZI into an LMImage."""
    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".czi":
        arr, axes = None, None
        try:
            from aicspylibczi import CziFile
            czi = CziFile(path)
            arr, shp = czi.read_image()
            axes = "".join(d for d, _ in shp)
        except Exception:
            arr = None
        if arr is None:
            try:
                from czifile import CziFile as _GohlkeCzi
            except ImportError as exc:
                raise ImportError("Reading .czi needs 'aicspylibczi' or "
                                  "'czifile':  pip install czifile") from exc
            with _GohlkeCzi(path) as czi:
                arr = np.asarray(czi.asarray())
                axes = "".join(czi.axes)
        stack = _to_czyx(np.asarray(arr), axes)
        spacing = None
        try:
            raw = Path(path).read_bytes()
            s, e = raw.find(b"<ImageDocument"), raw.find(b"</ImageDocument>")
            xml = (raw[s:e] if (s != -1 and e != -1) else raw).decode("utf-8", "replace")
            m = re.search(r'<Distance Id="X">\s*<Value>([^<]+)</Value>', xml)
            spacing = float(m.group(1)) * 1e6 if m else None
        except Exception:
            spacing = None
        info = f"czi {tuple(stack.shape)}"
    else:
        with tifffile.TiffFile(path) as tf:
            data = tf.asarray()
            axes = tf.series[0].axes if tf.series else "YX"
        stack = _to_czyx(np.asarray(data), axes)
        spacing = read_tiff_pixel_spacing_um(path)
        info = f"tif {tuple(stack.shape)} axes={axes}"

    c, z, y, x = stack.shape
    print(f"[INFO] LM {Path(path).name}: C={c} Z={z} {x}x{y} px, "
          f"{spacing if spacing else 'unknown'} um/px")
    return LMImage(path=path, stack_czyx=stack, pixel_spacing_um=spacing,
                   num_channels=c, num_z=z, info=info + f"  {spacing} um/px")


# --------------------------------------------------------------------------- #
# Correlation: fit, warp, re-apply, persist
# --------------------------------------------------------------------------- #

MIN_PAIRS = {"euclidean": 2, "similarity": 2, "affine": 3, "projective": 4}


def matrix_of(tform):
    for attr in ("params", "matrix"):
        M = getattr(tform, attr, None)
        if M is not None:
            return np.asarray(M, dtype=float)
    raise TypeError("transform has no matrix")


def apply_matrix(M, pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    h = np.hstack([pts, np.ones((len(pts), 1))])
    out = h @ np.asarray(M, float).T
    w = out[:, 2:3]
    w[w == 0] = 1.0
    return out[:, :2] / w


def image_center(shape):
    h, w = shape[-2:]
    return np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=float)


def diagnostics(M, src=None, dst=None):
    M = np.asarray(M, dtype=float)
    sx = math.hypot(M[0, 0], M[1, 0])
    sy = math.hypot(M[0, 1], M[1, 1])
    rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    rmse, rmse_txt = None, "n/a"
    if src is not None and len(np.asarray(src).reshape(-1, 2)):
        pred = apply_matrix(M, src)
        rmse = float(np.sqrt(np.mean(np.sum(
            (pred - np.asarray(dst, float).reshape(-1, 2)) ** 2, axis=1))))
        rmse_txt = f"{rmse:.2f} px"
    return {"scale_x": sx, "scale_y": sy, "rotation_deg": rot, "rmse_px": rmse,
            "text": f"scale x={sx:.4f} y={sy:.4f}  rot={rot:.2f} deg  RMSE={rmse_txt}"}


def fit_transform(point_pairs, transform_type):
    """Fit DISPLAY-LM -> DISPLAY-MAP from landmark pairs."""
    pairs = [(p["lm"], p["map"]) for p in point_pairs if "lm" in p and "map" in p]
    need = MIN_PAIRS.get(transform_type, 3)
    if len(pairs) < need:
        raise ValueError(f"{transform_type.capitalize()} needs >= {need} pairs "
                         f"(you have {len(pairs)}).")
    src = np.array([p[0] for p in pairs], float).reshape(-1, 2)
    dst = np.array([p[1] for p in pairs], float).reshape(-1, 2)
    M = matrix_of(estimate_transform(transform_type, src, dst))
    return ProjectiveTransform(matrix=M), diagnostics(M, src, dst), len(pairs)


def rescale_about(M, new_scale, src_anchor, dst_anchor):
    """Copy of M with the linear scale set to new_scale (rotation preserved),
    translated so src_anchor -> dst_anchor."""
    M = np.asarray(M, dtype=float)
    A = M[:2, :2]
    stored = float(np.hypot(A[0, 0], A[1, 0]))
    if stored <= 0:
        raise ValueError("transform has no positive scale to rescale")
    A_new = (new_scale / stored) * A
    out = np.eye(3, dtype=float)
    out[:2, :2] = A_new
    out[:2, 2] = np.asarray(dst_anchor, float) - A_new @ np.asarray(src_anchor, float)
    return out, stored


def warp_slice(stack, c, z, tform, map_shape, flip_x=False, flip_y=True):
    img = flip_for_display(stack[c, z], flip_x, flip_y)
    return warp(img, tform.inverse, output_shape=map_shape[-2:], order=1,
                preserve_range=True, mode="constant", cval=0).astype(np.float32)


def warp_crop(stack, c, z, tform, map_shape, x0_true, y0_true, cw,
              flip_x=False, flip_y=True):
    """Warp one (channel, z) LM slice straight into a cw x cw window whose
    top-left corner is (x0_true, y0_true) in TRUE map pixels.

    Bit-identical to warp_slice(...) -> display_to_map -> crop, but it
    interpolates cw**2 pixels instead of the whole map. Writing crops used to
    warp the entire map once per channel per z just to cut a few small windows
    out of it, which is what made it slow.
    """
    H, W = map_shape[-2:]
    # The map is drawn flipped, so a TRUE-space window maps to this DISPLAY
    # window; deriving it from the true origin (not the rounded centre) keeps
    # the two paths exactly aligned.
    x0 = (W - cw - x0_true) if MONTAGE_FLIP_X else x0_true
    y0 = (H - cw - y0_true) if MONTAGE_FLIP_Y else y0_true

    img = flip_for_display(stack[c, z], flip_x, flip_y)
    off = np.asarray([float(x0), float(y0)])
    crop = warp(img, lambda coords: tform.inverse(coords + off),
                output_shape=(cw, cw), order=1, preserve_range=True,
                mode="constant", cval=0).astype(np.float32)

    # Beyond the map edge the old path produced zeros (it cropped after
    # warping into the map grid); preserve that.
    if x0 < 0 or y0 < 0 or x0 + cw > W or y0 + cw > H:
        yy, xx = np.mgrid[0:cw, 0:cw]
        inside = ((xx + x0 >= 0) & (xx + x0 < W) &
                  (yy + y0 >= 0) & (yy + y0 < H))
        crop = np.where(inside, crop, np.float32(0.0))

    return map_to_display(crop)      # flip the small crop into TRUE orientation


def warp_channels(stack, tform, map_shape, flip_x=False, flip_y=True, status_cb=None):
    """Max-over-Z warp of every channel into the DISPLAY-MAP frame."""
    C, Z = stack.shape[:2]
    out = []
    for c in range(C):
        acc = None
        for z in range(Z):
            if status_cb is not None:
                status_cb(f"Warping channel {c + 1}/{C}, z {z + 1}/{Z}...")
            w = warp_slice(stack, c, z, tform, map_shape, flip_x, flip_y)
            acc = w if acc is None else np.maximum(acc, w)
        out.append(acc)
    return out


# --------------------------------------------------------------------------- #
# LM preprocessing  (weak-fluorescence enhancement)
# --------------------------------------------------------------------------- #
# A deliberately simple, linear pipeline for detecting weak fluorescence:
#
#   bad-pixel correction -> mild smoothing -> broad background subtraction
#   -> Gaussian matched filter -> optional Z-max of the matched response
#
# It is for detection and localisation, not for making pretty pictures. Every
# step is linear or a targeted point repair, so peak POSITIONS are preserved
# even though intensities are not; nothing here is safe to treat as
# quantitative signal. No CLAHE, no wholesale median filtering.
#
# Reflection channels are never touched -- they are the reference.


@dataclass
class LMPreprocessSettings:
    """Parameters for LMPreprocessor. Each step can be switched off."""
    enabled: bool = False
    mode: str = "matched"                   # "matched" | "maxproj"

    # step toggles
    remove_bad_pixels: bool = True
    pre_smooth: bool = True
    subtract_background: bool = True
    matched_filter: bool = True

    # step parameters
    bad_pixel_sigma_threshold: float = 11.0   # robust sigmas from the local median
    bad_pixel_z_persistence: int = 3          # planes a pixel must be bad in
    pre_smoothing_sigma: float = 0.7          # px
    background_sigma: float = 35.0            # px
    matched_filter_sigma: float = 2.0         # px; = FWHM / 2.355


class LMPreprocessor:
    """Runs LMPreprocessSettings over one (C, Z, Y, X) light-microscopy stack.

    Each step is a separate method taking and returning a (Z, Y, X) float32
    array, so steps can be run, skipped or tuned individually.
    """

    def __init__(self, settings=None):
        self.settings = settings or LMPreprocessSettings()

    # -- dependency -------------------------------------------------------- #

    @staticmethod
    def _ndimage():
        try:
            from scipy import ndimage
            return ndimage
        except ImportError as exc:
            raise ImportError(
                "LM preprocessing needs scipy  ->  pip install scipy") from exc

    # -- steps ------------------------------------------------------------- #

    def find_bad_pixels(self, zyx):
        """Boolean (Y, X) mask of persistent hot/dead pixels.

        A pixel is a candidate in a plane when it deviates from its local 3x3
        median by more than `bad_pixel_sigma_threshold` robust sigmas (MAD
        based, so bright real features don't inflate the estimate). Only
        candidates recurring in at least `bad_pixel_z_persistence` planes are
        called bad -- that is what separates a detector defect, which sits in
        the same place in every plane, from genuine fluorescence.
        """
        ndi = self._ndimage()
        s = self.settings
        counts = np.zeros(zyx.shape[1:], dtype=np.int32)
        for plane in zyx:
            med = ndi.median_filter(plane, size=3, mode="nearest")
            resid = plane - med
            mad = np.median(np.abs(resid - np.median(resid)))
            sigma = 1.4826 * mad
            if sigma <= 0:
                continue
            counts += (np.abs(resid) > s.bad_pixel_sigma_threshold * sigma)
        persistence = max(1, int(s.bad_pixel_z_persistence))
        return counts >= min(persistence, zyx.shape[0])

    def correct_bad_pixels(self, zyx, mask=None):
        """Replace only the masked pixels with their local 3x3 median."""
        ndi = self._ndimage()
        if mask is None:
            mask = self.find_bad_pixels(zyx)
        if not mask.any():
            return zyx
        out = zyx.copy()
        for i, plane in enumerate(out):
            med = ndi.median_filter(plane, size=3, mode="nearest")
            plane[mask] = med[mask]
            out[i] = plane
        return out

    def presmooth(self, zyx):
        """Mild Gaussian, per plane. Weak on purpose: it must not move peaks."""
        ndi = self._ndimage()
        sigma = float(self.settings.pre_smoothing_sigma)
        if sigma <= 0:
            return zyx
        return np.stack([ndi.gaussian_filter(p, sigma, mode="nearest") for p in zyx])

    def subtract_background(self, zyx):
        """Subtract a broad Gaussian estimate of the slowly varying background.

        The result is kept signed -- clipping here would throw away the noise
        statistics the matched filter and any later thresholding depend on.

        `background_sigma` must be small relative to the image: the Gaussian
        reaches ~4 sigma, so once 8*sigma approaches the frame the estimate
        collapses towards a constant and slow gradients survive instead of
        being removed. Fine for a 2k camera frame at sigma 35; not for a small
        crop.
        """
        ndi = self._ndimage()
        sigma = float(self.settings.background_sigma)
        if sigma <= 0:
            return zyx
        smallest = min(zyx.shape[-2:])
        if 8.0 * sigma >= smallest and not getattr(self, "_bg_sigma_warned", False):
            self._bg_sigma_warned = True
            print(f"[WARN] background_sigma={sigma:g} is large for a "
                  f"{zyx.shape[-2]}x{zyx.shape[-1]} image: the background "
                  f"estimate degenerates towards a constant and gradients will "
                  f"not be removed. Use sigma < {smallest / 8.0:.0f}.")
        return np.stack([p - ndi.gaussian_filter(p, sigma, mode="nearest")
                         for p in zyx])

    def apply_matched_filter(self, zyx):
        """Gaussian matched to the expected feature size (sigma = FWHM/2.355)."""
        ndi = self._ndimage()
        sigma = float(self.settings.matched_filter_sigma)
        if sigma <= 0:
            return zyx
        return np.stack([ndi.gaussian_filter(p, sigma, mode="nearest") for p in zyx])

    # -- whole channel / stack --------------------------------------------- #

    @staticmethod
    def to_display_range(arr, hi_pct=99.9):
        """Rescale a matched-filter response into [0, 1] for display.

        The pipeline output is SIGNED and centred near zero, while the rest of
        the UI expects [0, 1]: the brightness/contrast sliders default to that
        range, and composite_overlay uses the channel value directly as an
        opacity. Feeding it signed, near-zero data makes the noise floor fill
        the display range and the overlay washes out to a solid colour.

        Negative values are below-background, so clip at zero and stretch by a
        high percentile of the positive side -- that keeps real peaks bright
        and leaves the background transparent.
        """
        a = np.nan_to_num(np.asarray(arr, dtype=np.float32))
        pos = a[a > 0]
        hi = float(np.percentile(pos, hi_pct)) if pos.size else 0.0
        if hi <= 0:
            return np.zeros_like(a)
        return np.clip(a / hi, 0.0, 1.0)

    def process_channel(self, zyx):
        """Run the enabled steps over one channel.

        Returns {"corrected", "bgsub", "matched", "maxproj"}; `maxproj` is the
        (Y, X) maximum matched response across Z.
        """
        s = self.settings
        out = np.asarray(zyx, dtype=np.float32)
        if s.remove_bad_pixels:
            out = self.correct_bad_pixels(out)
        corrected = out
        if s.pre_smooth:
            out = self.presmooth(out)
        if s.subtract_background:
            out = self.subtract_background(out)
        bgsub = out
        matched = self.apply_matched_filter(out) if s.matched_filter else out
        return {"corrected": corrected, "bgsub": bgsub, "matched": matched,
                "maxproj": matched.max(axis=0)}

    def process_stack(self, czyx, roles=None, skip_roles=("reflection",),
                      progress=None):
        """Enhance every non-reflection channel of a (C, Z, Y, X) stack.

        `roles` is the per-channel role list; channels whose role is in
        `skip_roles` are copied through untouched, as are channels with no role.
        Returns {"matched": (C, Z, Y, X), "maxproj": (C, Y, X), "processed":
        [bool per channel]}.
        """
        czyx = np.asarray(czyx)
        C = czyx.shape[0]
        roles = list(roles or [])
        matched = np.empty(czyx.shape, dtype=np.float32)
        maxproj = np.empty((C,) + czyx.shape[2:], dtype=np.float32)
        processed = []

        for c in range(C):
            role = roles[c] if c < len(roles) else None
            if role in skip_roles:
                matched[c] = czyx[c].astype(np.float32, copy=True)
                maxproj[c] = matched[c].max(axis=0)
                processed.append(False)
                continue
            if progress is not None:
                progress(c, C, role)
            res = self.process_channel(czyx[c])
            # Back into [0, 1] so the display, the BC sliders and the composite
            # behave the same as they do for a raw channel. Both outputs are
            # scaled by the SAME factor, taken from the per-plane response, so
            # "matched" and "maxproj" stay comparable to each other.
            scaled = self.to_display_range(res["matched"])
            matched[c] = scaled
            maxproj[c] = scaled.max(axis=0)
            processed.append(True)

        return {"matched": matched, "maxproj": maxproj, "processed": processed}

    @staticmethod
    def as_stack(result, mode):
        """The (C, Z, Y, X) stack for a display mode.

        "matched" is the per-plane matched response; "maxproj" repeats each
        channel's Z-maximum across every plane, so the array keeps its shape
        and the Z control stays harmless.
        """
        if mode == "maxproj":
            z = result["matched"].shape[1]
            return np.repeat(result["maxproj"][:, None, :, :], z, axis=1)
        return result["matched"]


def make_record(tform, transform_type, info, n_pairs, map_shape, lm_shape,
                flip_x, flip_y, map_ps, lm_ps):
    return TransformRecord(
        matrix=matrix_of(tform).tolist(), transform_type=transform_type,
        flip_x=bool(flip_x), flip_y=bool(flip_y),
        scale_x=info.get("scale_x"), scale_y=info.get("scale_y"),
        rotation_deg=info.get("rotation_deg"), rmse_px=info.get("rmse_px"),
        n_pairs=n_pairs,
        mrc_shape=tuple(map_shape) if map_shape is not None else None,
        tiff_shape=tuple(lm_shape) if lm_shape is not None else None,
        pixel_spacing_um=map_ps, tiff_pixel_spacing_um=lm_ps,
        created_at=datetime.now().isoformat(timespec="seconds"))


def run_fit_and_warp(point_pairs, transform_type, stack, map_shape,
                     flip_x=False, flip_y=True, map_ps=None, lm_ps=None,
                     status_cb=None):
    tform, info, n_pairs = fit_transform(point_pairs, transform_type)
    warped = warp_channels(stack, tform, map_shape, flip_x, flip_y, status_cb)
    rec = make_record(tform, transform_type, info, n_pairs, map_shape,
                      stack.shape, flip_x, flip_y, map_ps, lm_ps)
    print(f"[INFO] Fit {transform_type} from {n_pairs} pairs -- {info['text']}")
    return {"transform": tform, "fit_info": info, "n_pairs": n_pairs,
            "warped_channels": warped, "record": rec}


def run_reapply(record, stack, map_shape, map_ps=None, lm_ps=None,
                center_on_map=True, status_cb=None):
    """Re-apply a stored transform: rotation and flips from the record, scale
    from the pixel-size ratio, footprint centred on the map.  No registration
    translation is carried over."""
    M = np.asarray(record.matrix, dtype=float)
    A = M[:2, :2]
    stored_scale = float(np.hypot(A[0, 0], A[1, 0]))
    rot_deg = math.degrees(math.atan2(A[1, 0], A[0, 0]))

    want_scale, scale_src = stored_scale, "stored (pixel sizes unavailable)"
    if lm_ps and map_ps:
        want_scale = float(lm_ps) / float(map_ps)
        scale_src = f"{float(lm_ps):.6f} / {float(map_ps):.6f} um/px"

    src_anchor = image_center(stack.shape)
    if center_on_map:
        dst_anchor, anchor_txt = image_center(map_shape), "map centre"
    else:
        dst_anchor = apply_matrix(M, image_center(record.tiff_shape)[None, :])[0]
        anchor_txt = "previous LM centre"

    M2, _ = rescale_about(M, want_scale, src_anchor, dst_anchor)
    tform = ProjectiveTransform(matrix=M2)
    fx, fy = bool(record.flip_x), bool(record.flip_y)
    H, W = stack.shape[-2:]
    map_h, map_w = map_shape[-2:]

    print("[INFO] Re-applying stored transform")
    print(f"       rotation       : {rot_deg:+.3f} deg  (preserved)")
    print(f"       scale stored   : {stored_scale:.6f} map px / LM px")
    print(f"       scale applied  : {want_scale:.6f} map px / LM px  <- {scale_src}")
    print(f"       flip_x, flip_y : {fx}, {fy}")
    print(f"       anchored on    : {anchor_txt}")
    print(f"       LM centre      : ({src_anchor[0]:.1f}, {src_anchor[1]:.1f}) -> "
          f"({dst_anchor[0]:.1f}, {dst_anchor[1]:.1f})")
    print(f"       footprint      : {W}x{H} LM px -> "
          f"{W * want_scale:.0f}x{H * want_scale:.0f} of {map_w}x{map_h} map px")
    if record.transform_type == "projective":
        print("       [WARN] projective terms are dropped on re-apply (affine only)")

    info = diagnostics(M2)
    info["text"] = (f"re-applied: rot {rot_deg:+.2f} deg, scale "
                    f"{stored_scale:.4f} -> {want_scale:.4f}, "
                    f"flips ({fx}, {fy}), centred on {anchor_txt}")
    warped = warp_channels(stack, tform, map_shape, fx, fy, status_cb)
    rec = make_record(tform, record.transform_type, info, record.n_pairs,
                      map_shape, stack.shape, fx, fy, map_ps, lm_ps)
    return {"transform": tform, "fit_info": info, "n_pairs": record.n_pairs or 0,
            "warped_channels": warped, "record": rec}


def save_transform(record, save_dir, filename=None):
    """Write JSON, plus a CSV in the format the pipeline's clem_correlation reads."""
    os.makedirs(save_dir, exist_ok=True)
    stamp = (record.created_at or datetime.now().isoformat(timespec="seconds"))
    stamp = stamp.replace("-", "").replace(":", "").replace("T", "-")
    base = filename or f"transform_{record.transform_type or 'transform'}_{stamp}"
    base = os.path.splitext(base)[0]
    path = os.path.join(save_dir, base + ".json")
    n = 1
    while os.path.exists(path):
        path = os.path.join(save_dir, f"{base}-{n}.json"); n += 1

    d = record.to_dict()
    with open(path, "w") as fh:
        json.dump(d, fh, indent=2)

    csv_path = os.path.splitext(path)[0] + ".csv"
    matrix = d.pop("matrix")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# CLEM LM->MAP transform; DISPLAY LM px -> DISPLAY MAP px"])
        w.writerow(["key", "value"])
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                v = ";".join(str(x) for x in v)
            w.writerow([k, "" if v is None else v])
        if matrix is not None:
            for i, row in enumerate(matrix):
                for j, val in enumerate(row):
                    w.writerow([f"m{i}{j}", repr(float(val))])

    record.source_path = path
    return path


def load_transform(path):
    """Read a transform written here (.json) or by the pipeline (.csv/.yaml)."""
    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as fh:
            rec = TransformRecord.from_dict(json.load(fh))
    elif ext == ".csv":
        kv, mv = {}, {}
        with open(path, newline="") as fh:
            for row in csv.reader(fh):
                if not row:
                    continue
                key = row[0].strip()
                if not key or key.startswith("#") or key.lower() == "key":
                    continue
                val = row[1].strip() if len(row) > 1 else ""
                if len(key) == 3 and key[0] == "m" and key[1:].isdigit():
                    mv[key] = float(val)
                else:
                    kv[key] = val
        kv["matrix"] = ([[mv.get(f"m{i}{j}", 0.0) for j in range(3)]
                         for i in range(3)] if mv else None)
        rec = TransformRecord.from_dict(kv)
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Reading a YAML transform needs PyYAML.") from exc
        with open(path) as fh:
            rec = TransformRecord.from_dict(yaml.safe_load(fh))
    else:
        raise ValueError(f"Unrecognised transform file type: {ext}")
    if rec.matrix is None:
        raise ValueError(f"{Path(path).name} carries no matrix.")
    rec.source_path = path
    return rec


def find_latest_transform(folder):
    """Newest transform file in <folder>/transforms or <folder>/../transforms."""
    if not folder:
        return None
    cands = []
    for d in (Path(folder) / "transforms", Path(folder).parent / "transforms"):
        if d.is_dir():
            for pat in ("*.json", "*.csv", "*.yaml", "*.yml"):
                cands += [p for p in d.glob(pat) if p.is_file()]
    if not cands:
        return None
    newest = max(cands, key=lambda p: p.stat().st_mtime)
    try:
        rec = load_transform(newest)
        print(f"[INFO] Found stored transform: {newest.name}")
        return rec
    except Exception as exc:
        print(f"[WARN] Could not load {newest.name}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# UI widgets
# --------------------------------------------------------------------------- #

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
    """Scroll to zoom, middle-drag or shift-drag to pan."""

    def __init__(self, ax, canvas):
        self.ax, self.canvas, self._pan = ax, canvas, None
        w = canvas.get_tk_widget()
        for seq, cb in (("<MouseWheel>", self._wheel), ("<Button-4>", self._up),
                        ("<Button-5>", self._down), ("<Button-2>", self._press),
                        ("<B2-Motion>", self._drag), ("<ButtonRelease-2>", self._release),
                        ("<Shift-Button-1>", self._press),
                        ("<Shift-B1-Motion>", self._drag),
                        ("<Shift-ButtonRelease-1>", self._release)):
            w.bind(seq, cb, add="+")

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

    def _wheel(self, e):
        self._zoom(e.x, e.y, 1 / ZOOM_FACTOR if e.delta > 0 else ZOOM_FACTOR)

    def _up(self, e):
        self._zoom(e.x, e.y, 1 / ZOOM_FACTOR)

    def _down(self, e):
        self._zoom(e.x, e.y, ZOOM_FACTOR)

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
        self.ax.set_xlim(xl - dtx * (xr - xl) / bb.width,
                         xr - dtx * (xr - xl) / bb.width)
        self.ax.set_ylim(yl + dty * (yl - yr) / bb.height,
                         yr + dty * (yl - yr) / bb.height)
        self._pan = (e.x, e.y)
        self.canvas.draw_idle()

    def _release(self, _e):
        self._pan = None


class BCControls(ttk.Frame):
    """Min/max sliders."""

    def __init__(self, parent, callback, **kw):
        super().__init__(parent, **kw)
        self._cb = callback
        self.vmin_var = tk.DoubleVar(master=parent, value=0.0)
        self.vmax_var = tk.DoubleVar(master=parent, value=1.0)
        for i, (lbl, var, name) in enumerate([("Min", self.vmin_var, "_chg_min"),
                                              ("Max", self.vmax_var, "_chg_max")]):
            row = ttk.Frame(self); row.pack(fill="x", pady=1)
            ttk.Label(row, text=lbl, width=4, anchor="e").pack(side="left")
            ttk.Scale(row, from_=0.0, to=1.0, orient="horizontal", variable=var,
                      command=getattr(self, name)).pack(side="left", fill="x", expand=True)
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
    def vmin(self):
        return self.vmin_var.get()

    @property
    def vmax(self):
        return self.vmax_var.get()

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

    def composite_colors(self):
        """Per-channel RGB for the fluorescence composite; reflection excluded."""
        return [role_rgb(v.get()) if is_composite_role(v.get()) else None
                for v in self._role_vars]

    def indices_with_role(self, role):
        return [i for i, v in enumerate(self._role_vars) if v.get() == role]


# --------------------------------------------------------------------------- #
# Crop picker  (no navigator, no stage movement -- writes crops only)
# --------------------------------------------------------------------------- #

class CropPickerWindow(tk.Toplevel):
    """Composite view of map + warped LM channels.  Click to mark targets, then
    write a multi-channel crop stack around each one."""

    def __init__(self, parent, map_summary, map_true, channels_true, channel_names,
                 warp_slice_true=None, n_z=1, out_dir=None, channel_roles=None,
                 warp_crop_true=None):
        super().__init__(parent)
        self.title("Target Picker  --  crops only")
        self.configure(bg=BG)
        self.map = map_summary
        self._map_true = map_true
        self._chan_true = list(channels_true)
        self._chan_names = list(channel_names)
        # Roles drive the colours here too, so the picker matches the overlay.
        self._chan_roles = (list(channel_roles) if channel_roles is not None
                            else [default_role(i) for i in range(len(self._chan_true))])
        self._warp_slice_true = warp_slice_true
        # Warps straight into a crop window; far cheaper than warping the whole
        # map per (channel, z) when the crops cover less area than the map.
        self._warp_crop_true = warp_crop_true
        self._n_z = max(1, int(n_z))
        self._pix_um = map_summary.pixel_spacing_um or 1.0
        self._out_dir = out_dir or (map_summary.path if map_summary.path else os.getcwd())
        if os.path.isfile(self._out_dir):
            self._out_dir = os.path.dirname(self._out_dir)

        self._H, self._W = self._map_true.shape[:2]
        self._picks, self._artists = [], []
        self._view_target = 2000
        self._render_pending = False

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1250, int(sw * 0.92))}x{min(820, int(sh * 0.88))}")
        self.minsize(760, 460)
        ttk.Style(self).configure("Sm.TLabel", background=BG, foreground=FG,
                                  font=("Segoe UI", 9))
        self._build_ui()

    # ---------------- layout ----------------

    def _build_ui(self):
        main = ttk.Frame(self, padding=6); main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1); main.rowconfigure(0, weight=1)

        fig = Figure(figsize=(6, 5), facecolor=BG)
        self._ax = fig.add_subplot(111); self._ax.set_facecolor(BG)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02)
        self._fig = fig
        self._canvas = FigureCanvasTkAgg(fig, master=main)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._im = self._ax.imshow(np.zeros((1, 1, 3), np.float32), origin="upper",
                                   aspect="equal", interpolation="nearest")
        self._ax.set_xlim(-0.5, self._W - 0.5)
        self._ax.set_ylim(self._H - 0.5, -0.5)
        self._ax.set_autoscale_on(False); self._ax.axis("off")
        PanZoomHandler(self._ax, self._canvas)
        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._ax.callbacks.connect("xlim_changed", lambda _a: self._schedule())
        self._ax.callbacks.connect("ylim_changed", lambda _a: self._schedule())

        side = ttk.Frame(main, padding=(6, 4, 4, 4), width=290)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        for r in range(5):
            side.rowconfigure(r, weight=1 if r == 3 else 0)

        hdr = ttk.Frame(side); hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(hdr, text="TARGETS", foreground=CYA,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(hdr, text=f"{self._pix_um:.6f} um/px", style="Sm.TLabel",
                  foreground=BG3).pack(anchor="w")
        ttk.Label(hdr, text=f"{self.map.image_width} x {self.map.image_height} px",
                  style="Sm.TLabel", foreground=BG3).pack(anchor="w")

        layers = ttk.LabelFrame(side, text="Layers", padding=(4, 2))
        layers.grid(row=1, column=0, sticky="ew", pady=(4, 4))
        self._build_layers(layers)

        # Same layout as the pipeline picker: crop FOV, one Export button, the
        # destructive buttons, then a status line.
        btn = ttk.Frame(side); btn.grid(row=2, column=0, sticky="ew", pady=(2, 4))
        btn.columnconfigure(0, weight=1); btn.columnconfigure(1, weight=1)

        row = ttk.Frame(btn); row.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(row, text="Crop FOV (um):", style="Sm.TLabel").pack(side="left")
        self._fov_var = tk.StringVar(master=self, value="2.0")
        ttk.Entry(row, textvariable=self._fov_var, width=7).pack(side="left", padx=(4, 0))

        ttk.Button(btn, text="Export", style="Accent.TButton",
                   command=self._export).grid(row=1, column=0, columnspan=2,
                                              sticky="ew", pady=(4, 4))

        ttk.Button(btn, text="Remove last", style="Danger.TButton",
                   command=self._remove_last).grid(row=2, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(btn, text="Clear all", style="Danger.TButton",
                   command=self._clear).grid(row=2, column=1, sticky="ew", padx=(2, 0))

        self._export_status = tk.StringVar(master=self, value="")
        ttk.Label(btn, textvariable=self._export_status, style="Sm.TLabel",
                  foreground=BG3, wraplength=270,
                  justify="left").grid(row=3, column=0, columnspan=2,
                                       sticky="ew", pady=(6, 0))

        ttk.Button(btn, text="Choose output folder...",
                   command=self._choose_out).grid(row=4, column=0, columnspan=2,
                                                  sticky="ew", pady=(6, 2))
        self._out_var = tk.StringVar(master=self, value=f"-> {self._out_dir}")
        ttk.Label(btn, textvariable=self._out_var, style="Sm.TLabel",
                  foreground=BG3, wraplength=270,
                  justify="left").grid(row=5, column=0, columnspan=2, sticky="ew")

        tf = ttk.Frame(side); tf.grid(row=3, column=0, sticky="nsew", pady=(4, 0))
        tf.rowconfigure(0, weight=1); tf.columnconfigure(0, weight=1)
        cols = ("#", "X px", "Y px")
        self._tree = ttk.Treeview(tf, columns=cols, show="headings")
        for c, w in zip(cols, (34, 100, 100)):
            self._tree.heading(c, text=c); self._tree.column(c, width=w, anchor="center")
        vsb = ttk.Scrollbar(tf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew"); vsb.grid(row=0, column=1, sticky="ns")

        self._status = tk.StringVar(
            master=self, value="Left-click to mark a target.\nScroll=zoom  Shift+drag=pan")
        ttk.Label(side, textvariable=self._status, wraplength=275, justify="left",
                  style="Sm.TLabel").grid(row=4, column=0, sticky="ew", pady=(4, 0))
        self._schedule()

    def _build_layers(self, container):
        sc = tk.Canvas(container, bg=BG, highlightthickness=0, height=190)
        vsb = ttk.Scrollbar(container, orient="vertical", command=sc.yview)
        sc.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y"); sc.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(sc); sc.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: sc.configure(scrollregion=sc.bbox("all")))

        self._tem_on = tk.BooleanVar(master=self, value=True)
        r = ttk.Frame(inner); r.pack(fill="x", pady=(2, 0))
        tk.Label(r, text="  ", bg="#cccccc", width=2).pack(side="left", padx=(2, 4))
        ttk.Checkbutton(r, text="TEM map", variable=self._tem_on,
                        command=self._schedule).pack(side="left")
        self._tem_bc = BCControls(inner, callback=self._schedule)
        self._tem_bc.pack(fill="x", padx=4)

        self._chan_on, self._chan_bc = [], []
        for i in range(len(self._chan_true)):
            role = (self._chan_roles[i] if i < len(self._chan_roles)
                    else default_role(i))
            name = self._chan_names[i] if i < len(self._chan_names) else role
            # Reflection is a reference channel: available, but off by default
            # so the picker opens on the fluorescence view.
            on = tk.BooleanVar(master=self, value=(role not in ("reflection", "off")))
            r = ttk.Frame(inner); r.pack(fill="x", pady=(4, 0))
            tk.Label(r, text="  ", bg=ROLE_HEX.get(role, CHANNEL_HEX[i % len(CHANNEL_HEX)]),
                     width=2).pack(side="left", padx=(2, 4))
            ttk.Checkbutton(r, text=f"Ch {i} ({name})", variable=on,
                            command=self._schedule).pack(side="left")
            bc = BCControls(inner, callback=self._schedule); bc.pack(fill="x", padx=4)
            self._chan_on.append(on); self._chan_bc.append(bc)

    # ---------------- rendering ----------------

    def _view_window(self):
        xl, yl = self._ax.get_xlim(), self._ax.get_ylim()
        x0, x1 = sorted((float(xl[0]), float(xl[1])))
        y0, y1 = sorted((float(yl[0]), float(yl[1])))
        c0 = int(np.clip(np.floor(x0), 0, self._W)); c1 = int(np.clip(np.ceil(x1) + 1, 0, self._W))
        r0 = int(np.clip(np.floor(y0), 0, self._H)); r1 = int(np.clip(np.ceil(y1) + 1, 0, self._H))
        if c1 <= c0:
            c0, c1 = 0, self._W
        if r1 <= r0:
            r0, r1 = 0, self._H
        s = max(1, int(np.ceil(max(c1 - c0, r1 - r0) / self._view_target)))
        return c0, c1, r0, r1, s

    def _render(self):
        self._render_pending = False
        c0, c1, r0, r1, s = self._view_window()
        cols = np.arange(c0, c1, s); rows = np.arange(r0, r1, s)
        if not len(cols) or not len(rows):
            return
        tcols = (self._W - 1 - cols) if MONTAGE_FLIP_X else cols
        trows = (self._H - 1 - rows) if MONTAGE_FLIP_Y else rows
        rr = np.ix_(trows, tcols)
        shape = (len(rows), len(cols))

        if self._tem_on.get():
            base = apply_bc(self._map_true[rr].astype(np.float32),
                            self._tem_bc.vmin, self._tem_bc.vmax) * 0.6
        else:
            base = np.zeros(shape, np.float32)
        rgb = np.empty(shape + (3,), np.float32)
        rgb[..., 0] = base; rgb[..., 1] = base; rgb[..., 2] = base

        for i, L in enumerate(self._chan_true):
            if not self._chan_on[i].get():
                continue
            a = apply_bc(L[rr].astype(np.float32), self._chan_bc[i].vmin,
                         self._chan_bc[i].vmax)
            oma = 1.0 - a
            role = (self._chan_roles[i] if i < len(self._chan_roles)
                    else default_role(i))
            col = role_rgb(role) or CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            for k, cv in enumerate(col):
                rgb[..., k] *= oma
                if cv > 0:
                    rgb[..., k] += a * cv
        np.clip(rgb, 0.0, 1.0, out=rgb)
        self._im.set_data(rgb)
        self._im.set_extent([c0 - 0.5, c0 + len(cols) * s - 0.5,
                             r0 + len(rows) * s - 0.5, r0 - 0.5])
        self._canvas.draw_idle()

    def _schedule(self, *_):
        if not self._render_pending:
            self._render_pending = True
            self.after_idle(self._render)

    # ---------------- picking ----------------

    def _display_to_true(self, dx, dy):
        px = (self._W - 1 - dx) if MONTAGE_FLIP_X else dx
        py = (self._H - 1 - dy) if MONTAGE_FLIP_Y else dy
        return float(px), float(py)

    def _true_to_display(self, px, py):
        dx = (self._W - 1 - px) if MONTAGE_FLIP_X else px
        dy = (self._H - 1 - py) if MONTAGE_FLIP_Y else py
        return float(dx), float(dy)

    def _on_click(self, event):
        if event.button != 1 or event.inaxes is not self._ax or _shift_held(event):
            return
        if event.xdata is None or event.ydata is None:
            return
        px, py = self._display_to_true(event.xdata, event.ydata)
        pick = Pick(pick_id=str(len(self._picks) + 1),
                    image_coord_x=px, image_coord_y=py)
        self._picks.append(pick)
        self._refresh_tree(); self._redraw_points()
        self._status.set(f"Target #{pick.pick_id}\nmap px: {px:.0f}, {py:.0f}")

    def _redraw_points(self):
        for a in self._artists:
            try:
                a.remove()
            except Exception:
                pass
        self._artists = []
        for i, p in enumerate(self._picks):
            dx, dy = self._true_to_display(p.image_coord_x, p.image_coord_y)
            dot, = self._ax.plot(dx, dy, "o", color=CYA, markersize=6,
                                 markeredgecolor="white", markeredgewidth=0.8, zorder=6)
            txt = self._ax.text(dx + 9, dy - 9, str(i + 1), color=CYA, fontsize=8,
                                fontweight="bold", zorder=7)
            self._artists.extend([dot, txt])
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
            self._picks.pop(); self._refresh_tree(); self._redraw_points()

    def _clear(self):
        self._picks.clear(); self._refresh_tree(); self._redraw_points()

    def _choose_out(self):
        d = filedialog.askdirectory(parent=self, title="Output folder for crops",
                                    initialdir=self._out_dir)
        if d:
            self._out_dir = d
            self._out_var.set(f"-> {d}")

    # ---------------- crop writing ----------------

    def _crop(self, full, px, py, cw, fill=0.0):
        H, W = full.shape[:2]
        half = cw // 2
        x0, y0 = int(round(px)) - half, int(round(py)) - half
        out = np.full((cw, cw), fill, dtype=np.float32)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W, x0 + cw), min(H, y0 + cw)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = full[sy0:sy1, sx0:sx1]
        return out

    def _export(self):
        """Export the picks. Three steps, no dialogs:

          1. write one MRC crop per pick;
          2. write one multichannel TIFF overlay per pick;
          3. save a screenshot of the picker view.

        Crops use the crop FOV entered above. Everything lands in
        <output folder>/picks. This mirrors the pipeline picker's Export button,
        minus its step of returning coordinates to SerialEM -- this tool is
        offline and has no navigator.
        """
        if not self._picks:
            messagebox.showwarning("No targets", "Mark at least one target first.",
                                   parent=self)
            return
        try:
            fov = float(self._fov_var.get())
            if fov <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("Crop FOV", "Enter a positive FOV in micrometres.",
                                   parent=self)
            return

        ps = self.map.pixel_spacing_um
        if not ps or ps <= 0:
            messagebox.showerror("No pixel size",
                                 "The map has no pixel size, so a FOV in "
                                 "micrometres cannot be converted to pixels.",
                                 parent=self)
            return
        cw = max(2, int(round(fov / ps)))
        out_dir = os.path.join(self._out_dir, "picks")
        os.makedirs(out_dir, exist_ok=True)
        n_ch = len(self._chan_true)

        try:
            self._status.set(f"Writing {len(self._picks)} crop(s) at {cw} px...")
            self.update_idletasks()

            # One (channel, z) plane at a time, cropping every target out of it
            # before moving on -- a full map-sized plane per channel per z would
            # otherwise be held in memory all at once.
            stacks = [np.zeros((self._n_z, 1 + n_ch, cw, cw), np.float32)
                      for _ in self._picks]
            for i, p in enumerate(self._picks):
                tem = self._crop(self._map_true, p.image_coord_x, p.image_coord_y, cw)
                for z in range(self._n_z):
                    stacks[i][z, 0] = tem

            # Warping the whole map for every (channel, z) and then cutting
            # small windows out of it is the expensive way round. When the
            # crops cover less area than the map, warp straight into each crop
            # instead -- same pixels, far fewer of them.
            map_px = int(self._map_true.shape[0]) * int(self._map_true.shape[1])
            crop_px = len(self._picks) * cw * cw
            if self._warp_crop_true is not None and crop_px < map_px:
                half = cw // 2
                for i, p in enumerate(self._picks):
                    self._status.set(f"Cropping target {i + 1}/{len(self._picks)}...")
                    self.update_idletasks()
                    x0 = int(round(p.image_coord_x)) - half
                    y0 = int(round(p.image_coord_y)) - half
                    for c in range(n_ch):
                        for z in range(self._n_z):
                            stacks[i][z, c + 1] = self._warp_crop_true(c, z, x0, y0, cw)
            else:
                # Fallback: one full plane at a time (also the path when there
                # is no warp callback, or when the crops cover most of the map).
                for c in range(n_ch):
                    for z in range(self._n_z):
                        self._status.set(f"Cropping channel {c + 1}/{n_ch}, "
                                         f"z {z + 1}/{self._n_z}...")
                        self.update_idletasks()
                        plane = (self._warp_slice_true(c, z)
                                 if self._warp_slice_true is not None else self._chan_true[c])
                        for i, p in enumerate(self._picks):
                            stacks[i][z, c + 1] = self._crop(
                                plane, p.image_coord_x, p.image_coord_y, cw)
                        del plane

            # Same filenames as the pipeline picker writes, so crops from
            # either tool sit together and read the same way.
            n_mrc = n_tif = 0
            for i, p in enumerate(self._picks):
                tem = stacks[i][0, 0]
                stack = stacks[i]
                res = 1.0 / ps
                labels = (["TEM"] + [f"Ch{c}" for c in range(n_ch)]) * self._n_z

                tif_path = os.path.join(out_dir, f"target_overlays_{p.pick_id}.tif")
                tiff_write(tif_path, stack, imagej=True, resolution=(res, res),
                           metadata={"axes": "ZCYX", "unit": "um",
                                     "Labels": labels})
                n_tif += 1

                mrc_path = os.path.join(out_dir, f"crop_fov_{p.pick_id}.mrc")
                with mrcfile.new(mrc_path, overwrite=True) as m:
                    m.set_data(tem.astype(np.float32))
                    m.voxel_size = ps * 10000.0
                    m.update_header_from_data()
                p.view_crop_path = mrc_path
                n_mrc += 1

            # Screenshot of the picks, same as the pipeline picker.
            shot = os.path.join(out_dir, "picker_screenshot.png")
            self._fig.savefig(shot, dpi=150, facecolor=self._fig.get_facecolor(),
                              bbox_inches="tight")

            done = [f"{n_mrc} MRC crop(s)",
                    f"{n_tif} TIFF overlay(s) at {fov} um ({cw}x{cw} px)",
                    f"screenshot {os.path.basename(shot)}"]
            summary = "\n".join(f"  - {d}" for d in done)
            self._export_status.set(f"Exported to {out_dir}")
            self._status.set("Export complete.")
            print(f"[INFO] Exported {n_mrc} mrc + {n_tif} tif crop(s) to {out_dir}")
            messagebox.showinfo("Export complete",
                                f"Written to:\n{out_dir}\n\n{summary}", parent=self)
        except Exception as exc:
            import traceback; traceback.print_exc()
            self._status.set("Export failed.")
            self._export_status.set("Export failed.")
            messagebox.showerror("Export error", str(exc), parent=self)


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #

class RegistrationApp(tk.Tk):

    def __init__(self, map_path=None, lm_path=None, coord_key=None, section=0,
                 out_dir=None, refine=False):
        super().__init__()
        self.title("CLEM Registration Tool  --  standalone")
        self.configure(bg=BG)
        self.minsize(1150, 830)

        self.coord_key = coord_key
        self.section = section
        self.refine = bool(refine)
        self.out_dir = out_dir

        self.map_summary = None
        self.map_image = None          # TRUE frame
        self._sections = []            # montages available in the loaded file
        self._map_target = None        # what the user actually chose
        self._map_key = None           # (path, section) currently displayed
        self.lm = None
        self.lm_stack = None           # (C, Z, Y, X) as used downstream
        self.lm_stack_raw = None       # untouched, as loaded
        self.warped_channels = []      # DISPLAY-MAP frame

        # Weak-fluorescence enhancement. When on, lm_stack holds the enhanced
        # array and everything downstream -- display, fit, overlay, picker --
        # uses it; lm_stack_raw is kept so it can be switched off again.
        self.preprocess = LMPreprocessSettings()
        self._preprocessor = LMPreprocessor(self.preprocess)
        self._enhance_cache = None     # LMPreprocessor.process_stack result
        self._enhance_cache_key = None
        self.point_pairs = []          # DISPLAY-frame landmark pairs

        self.flip_x = tk.BooleanVar(master=self, value=LM_FLIP_X_DEFAULT)
        self.flip_y = tk.BooleanVar(master=self, value=LM_FLIP_Y_DEFAULT)
        self.lm_rotate = tk.BooleanVar(master=self, value=False)

        self._last_tform = None
        self._loaded_record = None
        self._map_dirty = False
        self._lm_dirty = False
        self._map_artists = []
        self._lm_artists = []

        self._build_styles()
        self._build_ui()

        if map_path:
            self.after(100, lambda: self._do_load_map(map_path))
        if lm_path:
            self.after(200, lambda: self._do_load_lm(lm_path))

    # ---------------- styles / layout ----------------

    def _build_styles(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        s.configure("Sm.TLabel", background=BG, foreground=FG, font=("Segoe UI", 9))
        s.configure("TButton", background=ACC, foreground=BG,
                    font=("Segoe UI", 10, "bold"), padding=5)
        s.map("TButton", background=[("active", CYA), ("disabled", BG3)])
        s.configure("Accent.TButton", background=ACC2, foreground=BG,
                    font=("Segoe UI", 11, "bold"), padding=7)
        s.configure("Danger.TButton", background=RED, foreground=BG,
                    font=("Segoe UI", 10, "bold"), padding=5)
        s.configure("Map.TButton", background="#cba6f7", foreground=BG,
                    font=("Segoe UI", 10, "bold"), padding=5)
        s.configure("TLabelframe", background=BG, relief="groove")
        s.configure("TLabelframe.Label", background=BG, foreground=CYA,
                    font=("Segoe UI", 10, "bold"))
        s.configure("TCheckbutton", background=BG, foreground=FG)
        s.configure("TScale", background=BG, troughcolor=BG2, sliderlength=14)
        s.configure("Treeview", background=BG2, foreground=FG, fieldbackground=BG2,
                    rowheight=22)
        s.configure("Treeview.Heading", background=BG3, foreground=FG)
        s.map("Treeview", background=[("selected", ACC)])

    def _build_ui(self):
        PAD = 8
        top = ttk.Frame(self, padding=(PAD, PAD, PAD, 0)); top.pack(fill="x")
        ttk.Button(top, text="Load TEM map", style="Map.TButton",
                   command=self._load_map_file).pack(side="left", padx=2)
        self.map_info_var = tk.StringVar(master=self, value="No map loaded")
        ttk.Label(top, textvariable=self.map_info_var, width=34,
                  style="Sm.TLabel").pack(side="left", padx=4)
        # A montage .mrc can hold several montages (one per navigator item).
        # This lists them and swaps between them without reopening the file.
        ttk.Label(top, text="Montage:", style="Sm.TLabel").pack(side="left")
        self.section_var = tk.StringVar(master=self, value="")
        self.section_combo = ttk.Combobox(top, textvariable=self.section_var,
                                          width=30, state="disabled", values=[])
        self.section_combo.pack(side="left", padx=(4, 0))
        self.section_combo.bind("<<ComboboxSelected>>", self._on_section_change)
        # Off by default: matching every seam costs a second pass over the
        # tiles, and SerialEM's own aligned coordinates are usually fine.
        self.refine_var = tk.BooleanVar(master=self, value=bool(self.refine))
        ttk.Checkbutton(top, text="Refine tiles", variable=self.refine_var,
                        command=self._on_refine_toggle).pack(side="left", padx=(6, 0))
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(top, text="Load LM image",
                   command=self._load_lm_file).pack(side="left", padx=2)
        self.lm_info_var = tk.StringVar(master=self, value="No LM image loaded")
        ttk.Label(top, textvariable=self.lm_info_var, width=34,
                  style="Sm.TLabel").pack(side="left", padx=4)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(top, text="Ch:", style="Sm.TLabel").pack(side="left")
        self.channel_var = tk.IntVar(master=self, value=0)
        self.channel_spin = ttk.Spinbox(top, from_=0, to=0, width=4,
                                        textvariable=self.channel_var,
                                        command=self._refresh_lm)
        self.channel_spin.pack(side="left", padx=2)
        ttk.Label(top, text="Z:", style="Sm.TLabel").pack(side="left")
        self.z_var = tk.IntVar(master=self, value=0)
        self.z_spin = ttk.Spinbox(top, from_=0, to=0, width=4, textvariable=self.z_var,
                                  command=self._refresh_lm)
        self.z_spin.pack(side="left", padx=2)
        # ---- weak-fluorescence enhancement -------------------------------- #
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)
        self.enhance_var = tk.BooleanVar(master=self, value=False)
        ttk.Checkbutton(top, text="Enhance", variable=self.enhance_var,
                        command=self._on_enhance_toggle).pack(side="left")
        self.enhance_mode_var = tk.StringVar(master=self, value="matched")
        self.enhance_mode = ttk.Combobox(
            top, textvariable=self.enhance_mode_var, width=14, state="readonly",
            values=["matched", "maxproj"])
        self.enhance_mode.pack(side="left", padx=(4, 0))
        self.enhance_mode.bind("<<ComboboxSelected>>",
                               lambda _e: self._on_enhance_toggle())

        ttk.Label(top, text="  scroll=zoom  shift-drag=pan  left-click=landmark",
                  style="Sm.TLabel", foreground=BG3).pack(side="right", padx=6)

        panels = ttk.Frame(self, padding=PAD); panels.pack(fill="both", expand=True)
        panels.columnconfigure(0, weight=1); panels.columnconfigure(1, weight=1)
        panels.rowconfigure(0, weight=1)
        self.canvas_map, self.ax_map, self.bc_map = self._make_map_panel(panels)
        self.canvas_lm, self.ax_lm, self.bc_lm_panel = self._make_lm_panel(panels)

        bot = ttk.Frame(self, padding=(PAD, 0, PAD, PAD)); bot.pack(fill="x")
        pf = ttk.LabelFrame(bot, text="Landmark pairs (display pixels)", padding=4)
        pf.pack(side="left", fill="both", expand=True)
        cols = ("#", "MAP (x, y)", "LM (x, y)")
        self.tree = ttk.Treeview(pf, columns=cols, show="headings", height=3)
        for c in cols:
            self.tree.heading(c, text=c); self.tree.column(c, width=130, anchor="center")
        sb = ttk.Scrollbar(pf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True); sb.pack(side="left", fill="y")

        bf = ttk.Frame(bot, padding=(10, 0, 0, 0)); bf.pack(side="left", fill="y")
        self.status_var = tk.StringVar(
            master=self, value="Load a map and an LM image, flip to match, then pick landmarks.")
        ttk.Label(bf, textvariable=self.status_var, wraplength=230, justify="left",
                  style="Sm.TLabel").pack(pady=(0, 4), anchor="w")
        row = ttk.Frame(bf); row.pack(fill="x", pady=(0, 4))
        ttk.Label(row, text="Transform:", style="Sm.TLabel").pack(side="left")
        self.transform_var = tk.StringVar(master=self, value="similarity")
        ttk.Combobox(row, textvariable=self.transform_var, width=11, state="readonly",
                     values=["euclidean", "similarity", "affine", "projective"]
                     ).pack(side="left", padx=(4, 0))

        bg = ttk.Frame(bf); bg.pack(fill="x")
        bg.columnconfigure(0, weight=1); bg.columnconfigure(1, weight=1)
        ttk.Button(bg, text="Remove last", style="Danger.TButton",
                   command=self._remove_last).grid(row=0, column=0, sticky="ew",
                                                   padx=(0, 2), pady=2)
        ttk.Button(bg, text="Clear all", style="Danger.TButton",
                   command=self._clear_points).grid(row=0, column=1, sticky="ew",
                                                    padx=(2, 0), pady=2)
        ttk.Button(bg, text="Apply Transform", style="Accent.TButton",
                   command=self._apply_transform).grid(row=1, column=0, columnspan=2,
                                                       sticky="ew", pady=(4, 2))
        ttk.Button(bg, text="Re-apply stored",
                   command=self._reapply_stored).grid(row=2, column=0, columnspan=2,
                                                      sticky="ew", pady=2)
        ttk.Button(bg, text="Show Overlay",
                   command=self._show_overlay).grid(row=3, column=0, columnspan=2,
                                                    sticky="ew", pady=2)
        ttk.Button(bg, text="Export", command=self._export_transform).grid(
            row=4, column=0, sticky="ew", padx=(0, 2), pady=2)
        ttk.Button(bg, text="Import", command=self._import_transform).grid(
            row=4, column=1, sticky="ew", padx=(2, 0), pady=2)

    @staticmethod
    def _style_ax(ax):
        ax.set_facecolor(BG); ax.tick_params(colors=BG3)
        for sp in ax.spines.values():
            sp.set_edgecolor(BG3)

    def _make_map_panel(self, parent):
        lf = ttk.LabelFrame(parent, text="TEM map  --  left-click to place a landmark",
                            padding=4)
        lf.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        lf.rowconfigure(0, weight=1); lf.columnconfigure(0, weight=1)
        fig = Figure(figsize=(5, 3.2), facecolor=BG)
        ax = fig.add_subplot(111); self._style_ax(ax)
        fig.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.03)
        canvas = FigureCanvasTkAgg(fig, master=lf)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        bcf = ttk.LabelFrame(lf, text="Brightness / Contrast (map)", padding=(4, 2))
        bcf.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        bc = BCControls(bcf, callback=self._on_bc_map); bc.pack(fill="x")
        PanZoomHandler(ax, canvas)
        canvas.mpl_connect("button_press_event", self._on_click_map)
        return canvas, ax, bc

    def _make_lm_panel(self, parent):
        lf = ttk.LabelFrame(parent, text="LM image  --  flip to match, then left-click",
                            padding=4)
        lf.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        lf.rowconfigure(0, weight=1); lf.columnconfigure(0, weight=1)
        fig = Figure(figsize=(5, 3.2), facecolor=BG)
        ax = fig.add_subplot(111); self._style_ax(ax)
        fig.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.03)
        canvas = FigureCanvasTkAgg(fig, master=lf)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        flip = ttk.Frame(lf, padding=(4, 2)); flip.grid(row=1, column=0, sticky="ew")
        ttk.Label(flip, text="Flip:", style="Sm.TLabel").pack(side="left", padx=(0, 4))
        self.flip_x_cb = ttk.Checkbutton(flip, text="X", variable=self.flip_x,
                                         command=self._on_flip_changed)
        self.flip_x_cb.pack(side="left", padx=4)
        self.flip_y_cb = ttk.Checkbutton(flip, text="Y", variable=self.flip_y,
                                         command=self._on_flip_changed)
        self.flip_y_cb.pack(side="left", padx=4)
        self.rotate_cb = ttk.Checkbutton(flip, text="Rotate to fit (display only)",
                                         variable=self.lm_rotate,
                                         command=self._on_rotate_changed,
                                         state="disabled")
        self.rotate_cb.pack(side="left", padx=(12, 4))

        outer = ttk.LabelFrame(lf, text="Brightness / Contrast (per channel)",
                               padding=(4, 2))
        outer.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        sc = tk.Canvas(outer, bg=BG, highlightthickness=0, height=80)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=sc.yview)
        sc.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y"); sc.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(sc); sc.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: sc.configure(scrollregion=sc.bbox("all")))
        panel = ChannelBCPanel(inner); panel.pack(fill="x")
        PanZoomHandler(ax, canvas)
        canvas.mpl_connect("button_press_event", self._on_click_lm)
        return canvas, ax, panel

    # ---------------- loading ----------------

    def _status(self, msg):
        self.status_var.set(msg); self.update_idletasks()

    def _load_map_file(self):
        p = filedialog.askopenfilename(
            title="Open TEM map",
            filetypes=[("MRC / mdoc", ("*.mrc", "*.rec", "*.mrcs", "*.map", "*.st", "*.mdoc")),
                       ("All files", "*")])
        if p:
            self._do_load_map(p)

    def _do_load_map(self, target, section=None):
        """Load one montage. `section` defaults to the current selection."""
        try:
            self._status(f"Loading map from {os.path.basename(str(target))}...")

            # Read the mdoc first: it is cheap and tells us how many montages
            # the file holds, so the dropdown is right even if assembly fails.
            try:
                self._sections = list_montage_sections(target, self.coord_key)
            except Exception as exc:
                print(f"[WARN] Could not list montages: {exc}")
                self._sections = []
            if section is None:
                # Re-loading the same file keeps the montage on screen; a
                # different file starts at its first montage, since montage
                # numbers mean nothing across files.
                same_file = (self._map_key is None
                             or str(target) == str(self._map_target))
                section = self.section if same_file else 0
            self._populate_sections(section)

            self.refine = bool(self.refine_var.get())
            if self.refine:
                self._status("Matching tile seams...")
                self.config(cursor="watch"); self.update_idletasks()
            try:
                summary = load_map(target, self.coord_key, section,
                                   status_cb=self._status, refine=self.refine)
            finally:
                self.config(cursor="")

            # A different montage -- or the same one with the tiles moved by
            # refinement -- is a different pixel grid, so landmarks and any
            # warped channels no longer mean anything. Dropping them here is
            # better than letting a stale overlay be silently reused.
            applied = bool(summary.refinement and summary.refinement.applied)
            new_key = (str(summary.path), summary.section, applied)
            if getattr(self, "_map_key", None) not in (None, new_key):
                self._reset_correlation("Map changed -- landmarks cleared.")

            self._map_key = new_key
            self._map_target = target
            self.section = summary.section
            self.map_summary = summary
            self.map_image = summary.image
            if summary.sections:
                self._sections = summary.sections
                self._populate_sections(summary.section)
            self._map_dirty = True
            self.bc_map.reset()
            n = len(summary.sections)
            suffix = f"  montage {summary.section} of {n}" if n > 1 else ""
            if applied:
                suffix += "  refined"
            self.map_info_var.set(f"{summary.map_id}  [{summary.source}]{suffix}")
            self._draw_map()
            print(f"[INFO] {summary.info}")
            self._status(summary.info)
            if self.out_dir is None:
                base = summary.path
                self.out_dir = base if os.path.isdir(base) else os.path.dirname(base)
        except Exception as exc:
            import traceback; traceback.print_exc()
            messagebox.showerror("Map load error", f"{type(exc).__name__}: {exc}")

    def _populate_sections(self, current):
        """Fill the montage dropdown; disabled when there is nothing to choose."""
        labels = [s.describe() for s in self._sections]
        self.section_combo.config(values=labels)
        if len(labels) > 1:
            idx = next((i for i, s in enumerate(self._sections)
                        if s.index == current), 0)
            self.section_combo.config(state="readonly")
            self.section_var.set(labels[idx])
        else:
            self.section_combo.config(state="disabled")
            self.section_var.set(labels[0] if labels else "single image")

    def _on_refine_toggle(self):
        """Re-assemble the montage on screen with refinement on or off.

        Nothing is written either way -- this only changes where the tiles are
        laid down in the mosaic that this session works from.
        """
        target = getattr(self, "_map_target", None)
        if target is None:
            self.refine = bool(self.refine_var.get())
            return                      # takes effect on the next load
        section = self.map_summary.section if self.map_summary else None
        self._do_load_map(target, section=section)

    def _on_section_change(self, _evt=None):
        """The user picked a different montage from the same file."""
        target = getattr(self, "_map_target", None)
        if target is None or not self._sections:
            return
        try:
            idx = self.section_combo.current()
        except Exception:
            idx = -1
        if idx < 0 or idx >= len(self._sections):
            return
        wanted = self._sections[idx].index
        if self.map_summary is not None and wanted == self.map_summary.section:
            return
        self._do_load_map(target, section=wanted)

    def _reset_correlation(self, message):
        """Forget the fit and everything derived from it."""
        self.point_pairs.clear()
        self.warped_channels = []
        self._last_tform = None
        self._loaded_record = None
        self.lm_rotate.set(False)
        self.rotate_cb.config(state="disabled")
        self._update_tree()
        self._lm_dirty = True
        self._draw_lm()
        self._status(message)

    def _load_lm_file(self):
        p = filedialog.askopenfilename(
            title="Open LM image",
            filetypes=[("Light microscopy", ("*.tif", "*.tiff", "*.czi")),
                       ("All files", "*")])
        if p:
            self._do_load_lm(p)

    def _do_load_lm(self, path):
        try:
            self._status(f"Loading {os.path.basename(path)}...")
            lm = load_lm(path)
            self.lm = lm
            self.lm_stack_raw = lm.stack_czyx
            self.lm_stack = lm.stack_czyx
            self._enhance_cache = None
            self._enhance_cache_key = None
            self.warped_channels = []
            self._loaded_record = None
            self._last_tform = None
            self.lm_rotate.set(False)
            self.rotate_cb.config(state="disabled")
            for cb in (self.flip_x_cb, self.flip_y_cb):
                cb.config(state="normal")
            C, Z = self.lm_stack.shape[:2]
            self.channel_spin.config(to=max(0, C - 1)); self.channel_var.set(0)
            self.z_spin.config(to=max(0, Z - 1)); self.z_var.set(0)
            self.flip_x.set(LM_FLIP_X_DEFAULT); self.flip_y.set(LM_FLIP_Y_DEFAULT)
            # Keep role assignments when reloading a stack with the same
            # channel count; otherwise fall back to positional defaults.
            prev_roles = self.bc_lm_panel.roles()
            keep = prev_roles if len(prev_roles) == C else None
            self.bc_lm_panel.build(C, self._on_bc_lm, roles=keep,
                                   on_role_change=self._on_channel_role_change)
            self.lm_info_var.set(f"{os.path.basename(path)}  {lm.info}")
            self._lm_dirty = True
            if self.enhance_var.get():
                self._apply_enhancement()      # re-run for the new stack
            self._draw_lm()
            self._status("LM image loaded. Flip to match the map, then pick landmarks.")
        except Exception as exc:
            import traceback; traceback.print_exc()
            messagebox.showerror("LM load error", f"{type(exc).__name__}: {exc}")

    # ---------------- drawing ----------------

    def _on_bc_map(self):
        imgs = self.ax_map.get_images()
        if imgs:
            imgs[0].set_clim(self.bc_map.vmin, self.bc_map.vmax)
            self.canvas_map.draw_idle()

    def _on_bc_lm(self, channel_idx):
        cur = (min(self.channel_var.get(), self.lm_stack.shape[0] - 1)
               if self.lm_stack is not None else 0)
        if channel_idx != cur:
            return
        bc = self.bc_lm_panel.bc(cur)
        imgs = self.ax_lm.get_images()
        if bc and imgs:
            imgs[0].set_clim(bc.vmin, bc.vmax); self.canvas_lm.draw_idle()

    def _on_channel_role_change(self, channel_idx):
        """A channel was reassigned to a different role.

        The single-channel LM preview is greyscale, so nothing on the main
        window needs repainting; the roles are read when an overlay is built.
        Enhancement follows the roles, though -- which channels count as
        reflection has just changed -- so it is recomputed when active.
        """
        role = self.bc_lm_panel.role(channel_idx)
        self.status_var.set(f"Ch {channel_idx} set to '{role}'.\n"
                            f"Rebuild the overlay to see it.")
        if self.enhance_var.get():
            self._apply_enhancement()
            self._lm_dirty = True
            self._draw_lm(keep_view=True)

    # ---------------- weak-fluorescence enhancement ----------------

    def _on_enhance_toggle(self):
        """Checkbox or mode changed: swap lm_stack between raw and enhanced.

        The warped channels behind the overlay and the picker were produced
        from whatever lm_stack held at the time, so they are now stale. Re-warp
        with the existing transform, otherwise the overlay silently keeps
        showing the old data and the enhancement looks like it did nothing.
        """
        if self.lm_stack_raw is None:
            return
        self._apply_enhancement()
        self._lm_dirty = True
        self._draw_lm(keep_view=True)

        if self._last_tform is not None:
            self._status("Re-warping channels with the current transform...")
            self.update_idletasks()
            try:
                self._warp_with_current_transform()
            except Exception as exc:
                import traceback; traceback.print_exc()
                self.warped_channels = []
                self._status(f"Re-warp failed ({exc}); press Apply Transform.")

    def _warp_with_current_transform(self):
        """Re-run the warp for the current lm_stack using self._last_tform."""
        fx, fy = bool(self.flip_x.get()), bool(self.flip_y.get())
        self.warped_channels = warp_channels(
            self.lm_stack, self._last_tform, self.map_image.shape,
            flip_x=fx, flip_y=fy,
            status_cb=lambda m: (self._status(m), self.update_idletasks()))
        n = len(self.warped_channels)
        self._status(f"{n} channel(s) re-warped. Click Show Overlay.")

    def _apply_enhancement(self):
        """Point self.lm_stack at either the raw or the enhanced array.

        Everything downstream -- the fit, the warp, the overlay and the picker
        -- reads self.lm_stack, so enabling this changes what they all see. The
        raw array is kept in self.lm_stack_raw, so switching off is free.

        Reflection channels are never enhanced. The expensive part depends only
        on the stack and the role assignment, so it is cached and switching
        between "matched" and "maxproj" is instant.
        """
        self.preprocess.enabled = bool(self.enhance_var.get())
        self.preprocess.mode = self.enhance_mode_var.get()

        if not self.preprocess.enabled:
            self.lm_stack = self.lm_stack_raw
            self._status("Enhancement off -- using the raw LM channels.")
            return

        roles = self.bc_lm_panel.roles()
        key = (id(self.lm_stack_raw), tuple(roles),
               self.preprocess.bad_pixel_sigma_threshold,
               self.preprocess.bad_pixel_z_persistence,
               self.preprocess.pre_smoothing_sigma,
               self.preprocess.background_sigma,
               self.preprocess.matched_filter_sigma,
               self.preprocess.remove_bad_pixels, self.preprocess.pre_smooth,
               self.preprocess.subtract_background, self.preprocess.matched_filter)

        if self._enhance_cache is None or self._enhance_cache_key != key:
            try:
                self.config(cursor="watch"); self.update_idletasks()
                def _progress(c, total, role):
                    self._status(f"Enhancing channel {c + 1}/{total} ({role})...")
                    self.update_idletasks()
                self._enhance_cache = self._preprocessor.process_stack(
                    self.lm_stack_raw, roles=roles, progress=_progress)
                self._enhance_cache_key = key
            except ImportError as exc:
                self.enhance_var.set(False)
                self.preprocess.enabled = False
                self.lm_stack = self.lm_stack_raw
                messagebox.showerror("Enhancement unavailable", str(exc))
                return
            except Exception as exc:
                import traceback; traceback.print_exc()
                self.enhance_var.set(False)
                self.preprocess.enabled = False
                self.lm_stack = self.lm_stack_raw
                messagebox.showerror("Enhancement error", f"{type(exc).__name__}: {exc}")
                return
            finally:
                self.config(cursor="")

        self.lm_stack = LMPreprocessor.as_stack(self._enhance_cache,
                                                self.preprocess.mode)
        n = sum(self._enhance_cache["processed"])
        what = ("matched filter" if self.preprocess.mode == "matched"
                else "max matched response across Z")
        self._status(f"Enhancement on -- {what}, {n} fluorescence channel(s); "
                     f"reflection untouched. Re-apply the transform to update "
                     f"the overlay.")

    @staticmethod
    def _save_view(ax):
        return ax.get_xlim(), ax.get_ylim()

    @staticmethod
    def _restore_view(ax, xl, yl):
        if xl is not None:
            ax.set_xlim(xl); ax.set_ylim(yl)

    def _draw_map(self, keep_view=False):
        if self.map_image is None:
            return
        if self._map_dirty:
            xl, yl = self._save_view(self.ax_map) if keep_view else (None, None)
            self.ax_map.clear(); self._style_ax(self.ax_map)
            img = self.map_image
            h, w = img.shape
            ds = max(1, max(h, w) // 1024)
            disp = map_to_display(img[::ds, ::ds] if ds > 1 else img)
            self.ax_map.imshow(disp, cmap="gray", origin="upper", aspect="equal",
                               extent=[-0.5, w - 0.5, h - 0.5, -0.5],
                               vmin=self.bc_map.vmin, vmax=self.bc_map.vmax)
            if keep_view:
                self._restore_view(self.ax_map, xl, yl)
            self._map_dirty = False; self._map_artists = []
        else:
            for a in self._map_artists:
                try:
                    a.remove()
                except Exception:
                    pass
            self._map_artists = []

        for i, pair in enumerate(self.point_pairs):
            if "map" in pair:
                dx, dy = pair["map"]
                ln, = self.ax_map.plot(dx, dy, "o", color=PT_MAP, markersize=8,
                                       markeredgecolor="white", markeredgewidth=0.8,
                                       zorder=5)
                tx = self.ax_map.text(dx + 6, dy - 6, str(i + 1), color=PT_MAP,
                                      fontsize=9, fontweight="bold", zorder=6)
                self._map_artists.extend([ln, tx])

        # White outline of the LM footprint as it overlays the map.
        if self._last_tform is not None and self.lm_stack is not None:
            H, W = self.lm_stack.shape[-2:]
            corners = np.array([[-0.5, -0.5], [W - 0.5, -0.5], [W - 0.5, H - 0.5],
                                [-0.5, H - 0.5], [-0.5, -0.5]], dtype=float)
            pts = self._last_tform(corners)
            ln, = self.ax_map.plot(pts[:, 0], pts[:, 1], "-", color="white",
                                   linewidth=1.5, alpha=0.9, zorder=7)
            dot, = self.ax_map.plot(pts[0, 0], pts[0, 1], "s", color="white",
                                    markersize=4, zorder=8)
            self._map_artists.extend([ln, dot])
        self.canvas_map.draw_idle()

    def _display_rotation_deg(self):
        rec = self._loaded_record
        if rec is not None and getattr(rec, "rotation_deg", None):
            return float(rec.rotation_deg)
        if self._last_tform is not None:
            A = matrix_of(self._last_tform)[:2, :2]
            return math.degrees(math.atan2(A[1, 0], A[0, 0]))
        return None

    def _get_lm_slice(self, c, z):
        """DISPLAY slice: flipped per the checkboxes and, if 'Rotate to fit' is on,
        rotated for preview only.  Never used for picking or for the fit."""
        img = flip_for_display(self.lm_stack[c, z], self.flip_x.get(), self.flip_y.get())
        if self.lm_rotate.get():
            ang = self._display_rotation_deg()
            if ang:
                img = _sk_rotate(img, -ang, resize=True, preserve_range=True,
                                 order=1, cval=0).astype(np.float32)
        return img

    def _draw_lm(self, keep_view=False):
        if self.lm_stack is None:
            return
        c = min(self.channel_var.get(), self.lm_stack.shape[0] - 1)
        z = min(self.z_var.get(), self.lm_stack.shape[1] - 1)
        bc = self.bc_lm_panel.bc(c)
        vmin, vmax = (bc.vmin, bc.vmax) if bc else (0.0, 1.0)
        if self._lm_dirty:
            xl, yl = self._save_view(self.ax_lm) if keep_view else (None, None)
            self.ax_lm.clear(); self._style_ax(self.ax_lm)
            img = self._get_lm_slice(c, z)
            h, w = img.shape
            ds = max(1, max(h, w) // 1024)
            disp = img[::ds, ::ds] if ds > 1 else img
            self.ax_lm.imshow(disp, cmap="gray", origin="upper", aspect="equal",
                              extent=[-0.5, w - 0.5, h - 0.5, -0.5],
                              vmin=vmin, vmax=vmax)
            if keep_view:
                self._restore_view(self.ax_lm, xl, yl)
            self._lm_dirty = False; self._lm_artists = []
        else:
            for a in self._lm_artists:
                try:
                    a.remove()
                except Exception:
                    pass
            self._lm_artists = []

        for i, pair in enumerate(self.point_pairs):
            if "lm" in pair and not self.lm_rotate.get():
                dx, dy = pair["lm"]
                ln, = self.ax_lm.plot(dx, dy, "o", color=PT_LM, markersize=8,
                                      markeredgecolor="white", markeredgewidth=0.8,
                                      zorder=5)
                tx = self.ax_lm.text(dx + 6, dy - 6, str(i + 1), color=PT_LM,
                                     fontsize=9, fontweight="bold", zorder=6)
                self._lm_artists.extend([ln, tx])
        self.canvas_lm.draw_idle()

    def _refresh_lm(self):
        self._lm_dirty = True
        self._draw_lm(keep_view=True)

    # ---------------- landmarks ----------------

    def _on_click_map(self, event):
        if event.button != 1 or event.inaxes is not self.ax_map or _shift_held(event):
            return
        if self.map_image is None or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        for pair in self.point_pairs:
            if "map" not in pair:
                pair["map"] = (x, y)
                self._update_tree(); self._draw_map(keep_view=True)
                self.status_var.set("Map point set.\nNow click the matching LM point.")
                return
        self.point_pairs.append({"map": (x, y)})
        self._update_tree(); self._draw_map(keep_view=True)
        self.status_var.set(f"Map point #{len(self.point_pairs)} placed.\n"
                            "Now click the matching LM point.")

    def _on_click_lm(self, event):
        if event.button != 1 or event.inaxes is not self.ax_lm or _shift_held(event):
            return
        if self.lm_stack is None or event.xdata is None:
            return
        if self.lm_rotate.get():
            self.status_var.set("Rotated preview is display-only.\n"
                                "Uncheck 'Rotate to fit' to pick.")
            return
        x, y = float(event.xdata), float(event.ydata)
        for pair in reversed(self.point_pairs):
            if "map" in pair and "lm" not in pair:
                pair["lm"] = (x, y)
                self._update_tree(); self._draw_lm(keep_view=True)
                n = sum(1 for p in self.point_pairs if "map" in p and "lm" in p)
                self.status_var.set(f"Pair complete.  {n} pair(s).\nAdd more or Apply.")
                return
        self.point_pairs.append({"lm": (x, y)})
        self._update_tree(); self._draw_lm(keep_view=True)
        self.status_var.set(f"LM point #{len(self.point_pairs)} placed.\n"
                            "Now click the matching map point.")

    def _update_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, pair in enumerate(self.point_pairs):
            m = "({:.1f}, {:.1f})".format(*pair["map"]) if "map" in pair else "-"
            t = "({:.1f}, {:.1f})".format(*pair["lm"]) if "lm" in pair else "-"
            self.tree.insert("", "end", values=(i + 1, m, t))

    def _remove_last(self):
        if not self.point_pairs:
            self.status_var.set("No landmark pairs to remove."); return
        self.point_pairs.pop(); self._update_tree()
        self._draw_map(keep_view=True); self._draw_lm(keep_view=True)
        self.status_var.set(f"Removed. {len(self.point_pairs)} remaining.")

    def _clear_points(self):
        self.point_pairs.clear(); self.warped_channels.clear(); self._update_tree()
        self._draw_map(keep_view=True); self._draw_lm(keep_view=True)
        self.status_var.set("Landmarks cleared.")

    def _on_flip_changed(self):
        if self.point_pairs or self.warped_channels:
            self.point_pairs.clear(); self.warped_channels.clear(); self._update_tree()
            self._draw_map(keep_view=True)
        self._last_tform = None
        self._loaded_record = None
        self._refresh_lm()
        self.status_var.set("Flip changed. Landmarks cleared -- orient first, then pick.")

    def _on_rotate_changed(self):
        if self.lm_rotate.get() and self.point_pairs:
            self.point_pairs.clear(); self._update_tree()
            self._draw_map(keep_view=True)
            self.status_var.set("Landmarks cleared -- the rotated preview is "
                                "display-only.")
        self._refresh_lm()

    # ---------------- transform ----------------

    def _lm_scale(self):
        return self.lm.pixel_spacing_um if self.lm is not None else None

    def _map_scale(self):
        return self.map_summary.pixel_spacing_um if self.map_summary is not None else None

    def _apply_transform(self, quiet=False):
        if self.map_image is None or self.lm_stack is None:
            messagebox.showwarning("Missing data", "Load both a map and an LM image.")
            return
        ttype = self.transform_var.get()
        fx, fy = bool(self.flip_x.get()), bool(self.flip_y.get())
        n_pairs = sum(1 for p in self.point_pairs if "map" in p and "lm" in p)
        try:
            if self._loaded_record is not None and n_pairs == 0:
                result = run_reapply(self._loaded_record, self.lm_stack,
                                     self.map_image.shape,
                                     map_ps=self._map_scale(), lm_ps=self._lm_scale(),
                                     center_on_map=True, status_cb=self._status)
            else:
                result = run_fit_and_warp(self.point_pairs, ttype, self.lm_stack,
                                          self.map_image.shape, flip_x=fx, flip_y=fy,
                                          map_ps=self._map_scale(),
                                          lm_ps=self._lm_scale(),
                                          status_cb=self._status)
                try:
                    save_dir = os.path.join(
                        self.out_dir or os.path.dirname(self.map_summary.path or "."),
                        "transforms")
                    p = save_transform(result["record"], save_dir)
                    print(f"[INFO] Transform saved: {p}")
                except Exception as exc:
                    print(f"[WARN] Could not auto-save the transform: {exc}")
        except ValueError as exc:
            messagebox.showwarning("Not enough points", str(exc)); return
        except Exception as exc:
            import traceback; traceback.print_exc()
            messagebox.showerror("Transform failed", str(exc)); return

        self._loaded_record = result["record"]
        self._last_tform = result["transform"]
        self.warped_channels = result["warped_channels"]
        self.rotate_cb.config(state="normal")
        fit_txt = result["fit_info"]["text"]
        self.status_var.set(f"{fit_txt}\nClick Show Overlay.")
        self._draw_map(keep_view=True)
        if not quiet:
            messagebox.showinfo(
                "Transform applied",
                f"{len(self.warped_channels)} channel(s) warped onto the map.\n\n"
                f"{fit_txt}\n\nIf the RMSE is large the LM orientation probably "
                "does not match yet -- adjust the flips, re-pick and apply again.")

    def _reapply_stored(self):
        folder = None
        if self.map_summary is not None and self.map_summary.path:
            folder = (self.map_summary.path if os.path.isdir(self.map_summary.path)
                      else os.path.dirname(self.map_summary.path))
        rec = find_latest_transform(folder or self.out_dir)
        if rec is None:
            messagebox.showwarning("No transform",
                                   "No saved transform found in a 'transforms' "
                                   "folder next to the map.")
            return
        self._loaded_record = rec
        self.point_pairs.clear(); self._update_tree()
        self.flip_x.set(bool(rec.flip_x)); self.flip_y.set(bool(rec.flip_y))
        self._refresh_lm()
        self._apply_transform()

    def _export_transform(self):
        if self._loaded_record is None:
            messagebox.showwarning("No transform", "Apply a transform first."); return
        d = filedialog.askdirectory(title="Folder to save the transform")
        if not d:
            return
        try:
            p = save_transform(self._loaded_record, d)
            self.status_var.set(f"Saved {os.path.basename(p)}")
            messagebox.showinfo("Exported", f"Transform written to:\n{p}\n"
                                            "(.json plus a .csv for the pipeline)")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def _import_transform(self):
        if self.map_image is None or self.lm_stack is None:
            messagebox.showwarning("Missing data", "Load both a map and an LM image "
                                                   "before importing a transform.")
            return
        p = filedialog.askopenfilename(
            title="Import transform",
            filetypes=[("Transforms", ("*.json", "*.csv", "*.yaml", "*.yml")),
                       ("All files", "*")])
        if not p:
            return
        try:
            rec = load_transform(p)
            if rec.transform_type in MIN_PAIRS:
                self.transform_var.set(rec.transform_type)
            self._loaded_record = rec
            self.flip_x.set(bool(rec.flip_x)); self.flip_y.set(bool(rec.flip_y))
            self.point_pairs.clear(); self._update_tree()
            self._refresh_lm()
            self._apply_transform()
        except Exception as exc:
            import traceback; traceback.print_exc()
            messagebox.showerror("Import error", str(exc))

    # ---------------- overlay + picker ----------------

    def _show_overlay(self):
        if self.map_image is None:
            messagebox.showwarning("No map", "Load a map first."); return
        if not self.warped_channels:
            messagebox.showwarning("No overlay", "Apply a transform first."); return

        C = len(self.warped_channels)
        map_disp = apply_bc(map_to_display(fast_ds(self.map_image)),
                            self.bc_map.vmin, self.bc_map.vmax)
        chans = []
        for idx, ch in enumerate(self.warped_channels):
            bc = self.bc_lm_panel.bc(idx)
            vmin, vmax = (bc.vmin, bc.vmax) if bc else (0.0, 1.0)
            chans.append(apply_bc(fast_ds(ch), vmin, vmax))
        # Panels, in order: map | map+reflection | one per fluorescence channel
        # | composite. Reflection is a reference channel, so it gets its own
        # green overlay and is kept out of the composite. The plan is shared
        # with _save_panels, so the files always match what is on screen.
        roles = self.bc_lm_panel.roles()
        plan = overlay_panel_plan(roles, C)
        panels = [(map_disp if colors is None
                   else composite_overlay(map_disp, chans, colors=colors),
                   title, is_gray)
                  for title, colors, is_gray in plan]

        ncols = len(panels)
        win = tk.Toplevel(self); win.title("Overlay"); win.configure(bg=BG)
        win.minsize(640, 380)
        fig = Figure(figsize=(3.6 * ncols, 4.4), facecolor=BG)
        axes = [fig.add_subplot(1, ncols, i + 1) for i in range(ncols)]
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.04)

        def sa(ax, t):
            ax.set_facecolor(BG); ax.set_title(t, color=CYA, fontsize=8, pad=4)
            ax.axis("off")

        for ax, (img, title, is_gray) in zip(axes, panels):
            if is_gray:
                ax.imshow(img, cmap="gray", origin="upper", vmin=0, vmax=1)
            else:
                ax.imshow(img, origin="upper")
            sa(ax, title)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True); canvas.draw()
        for ax in axes:
            PanZoomHandler(ax, canvas)

        btns = ttk.Frame(win, padding=(6, 0, 6, 6)); btns.pack(fill="x")
        ttk.Button(btns, text="Open Target Picker (crops)", style="Map.TButton",
                   command=lambda: self._open_picker(win)).pack(side="left",
                                                                padx=(0, 6), pady=4)
        ttk.Button(btns, text="Save panels", style="Accent.TButton",
                   command=lambda: self._save_panels(map_disp, chans, plan)).pack(side="right",
                                                                                  pady=4)

    def _open_picker(self, parent):
        fx, fy = bool(self.flip_x.get()), bool(self.flip_y.get())
        tform = self._last_tform
        stack = self.lm_stack
        map_shape = self.map_image.shape

        def warp_slice_true(c, z):
            return display_to_map(warp_slice(stack, c, z, tform, map_shape, fx, fy))

        def warp_crop_true(c, z, x0, y0, cw):
            return warp_crop(stack, c, z, tform, map_shape, x0, y0, cw, fx, fy)

        chans_true = [display_to_map(ch) for ch in self.warped_channels]
        roles = self.bc_lm_panel.roles()
        names = [roles[i] if i < len(roles) else default_role(i)
                 for i in range(len(self.warped_channels))]
        CropPickerWindow(parent, map_summary=self.map_summary,
                         map_true=self.map_image, channels_true=chans_true,
                         channel_names=names, channel_roles=list(names),
                         warp_slice_true=warp_slice_true,
                         warp_crop_true=warp_crop_true,
                         n_z=(self.lm.num_z if self.lm else 1),
                         out_dir=self.out_dir)

    def _save_panels(self, map_disp, chans, plan=None):
        base = filedialog.asksaveasfilename(title="Base filename",
                                            defaultextension=".tif",
                                            filetypes=[("TIFF", "*.tif"),
                                                       ("PNG", "*.png")])
        if not base:
            return
        root, ext = os.path.splitext(base); ext = (ext or ".tif").lower()

        def to_u8(a):
            if a.ndim == 2:
                a = np.stack([a] * 3, axis=-1)
            return (np.clip(a, 0, 1) * 255).astype(np.uint8)

        def write(sfx, arr):
            out = root + sfx + ext
            if ext == ".png":
                from PIL import Image
                Image.fromarray(to_u8(arr)).save(out)   # PNG is deflate already
            elif arr.ndim == 2:
                tiff_write(out, (np.clip(arr, 0, 1) * 65535).astype(np.uint16),
                           photometric="minisblack")
            else:
                tiff_write(out, to_u8(arr), photometric="rgb")
            return os.path.basename(out)

        try:
            # Same panel plan as the figure, so files match the display.
            if plan is None:
                plan = overlay_panel_plan(self.bc_lm_panel.roles(), len(chans))
            saved = []
            for n, (title, colors, _is_gray) in enumerate(plan, start=1):
                img = (map_disp if colors is None
                       else composite_overlay(map_disp, chans, colors=colors))
                saved.append(write(f"_{n:02d}_{_panel_slug(title)}", img))
            messagebox.showinfo("Saved", "\n".join(saved))
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Standalone CLEM registration tool (no SerialEM).")
    ap.add_argument("map", nargs="?",
                    help="TEM map: an .mrc (or .mdoc) file.")
    ap.add_argument("lm", nargs="?", help="LM image (.tif/.tiff/.czi).")
    ap.add_argument("--coord-key", default=None,
                    help="Preferred mdoc piece-coordinate field "
                         "(default: first one populated).")
    ap.add_argument("--section", type=int, default=0,
                    help="Which montage to assemble when the file holds "
                         "several (default 0).")
    ap.add_argument("--list-sections", action="store_true",
                    help="List the montages in the map file and exit.")
    ap.add_argument("--refine-tiles", action="store_true",
                    help="Re-measure tile positions from the overlaps instead "
                         "of trusting the mdoc (start with it on).")
    ap.add_argument("--out", default=None,
                    help="Output folder for transforms and crops "
                         "(default: beside the map).")
    args = ap.parse_args(argv)

    if args.list_sections:
        if not args.map:
            ap.error("--list-sections needs a map file.")
        secs = list_montage_sections(args.map, args.coord_key)
        if not secs:
            print(f"{args.map}: not a SerialEM montage (no usable mdoc piece "
                  f"coordinates) -- it loads as a single image.")
        else:
            print(f"{args.map}: {len(secs)} montage(s)")
            for s in secs:
                print(f"  {s.describe()}")
        return

    app = RegistrationApp(map_path=args.map, lm_path=args.lm,
                          coord_key=args.coord_key, section=args.section,
                          out_dir=args.out, refine=args.refine_tiles)
    app.mainloop()
    try:
        app.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()