"""
Grid Calibration Tool
=====================
Workflow
--------
  1. Load CZI + a5proj
  2. Pick Grid Centre   – left-click
  3. Pick +X Direction  – left-click
  4. Compute            – fills table
  5. Export CSV

Controls: scroll=zoom  middle-drag=pan  left-click=pick  right-click=undo
Dependencies: pip install numpy pillow matplotlib
              System libzstd required for compressed CZI files (apt install libzstd1)
"""

import os, sys, re, struct, math, csv, ctypes, ctypes.util
import xml.etree.ElementTree as ET
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

try:
    from PIL import Image as _PIL
except ImportError:
    sys.exit("pip install pillow")

try:
    import zstandard as _zstd_pkg
    _ZSTD_PY_CTX = _zstd_pkg.ZstdDecompressor()
except ImportError:
    # zstandard is the primary zstd backend (pure Python, works on all platforms).
    # Without it, compressed CZI files (ZEN 3.x) will show as black images.
    # We still try the ctypes/libzstd fallback at runtime, but warn early.
    _ZSTD_PY_CTX = None
    print(
        "\n*** WARNING: 'zstandard' package not found. ***\n"
        "Compressed CZI files will fail to load.\n"
        "Fix with:  pip install zstandard\n"
        "Then restart the script.\n",
        file=sys.stderr
    )

# ── palette ────────────────────────────────────────────────────────────────────
BG   = "#1e1e2e"; BG2 = "#313244"; BG3 = "#45475a"
FG   = "#cdd6f4"; ACC = "#89b4fa"; ACC2 = "#a6e3a1"
RED  = "#f38ba8"; CYA = "#89dceb"; YEL  = "#f9e2af"; MAG = "#cba6f7"
ZOOM_FACTOR = 1.25


# ══════════════════════════════════════════════════════════════════════════════
#  Pan / Zoom  (sets limits only – never calls redraw itself)
# ══════════════════════════════════════════════════════════════════════════════
class PanZoom:
    def __init__(self, ax, canvas, on_view_change):
        self.ax  = ax
        self.cv  = canvas
        self._cb = on_view_change   # called after every limit change
        self._p0 = None
        w = canvas.get_tk_widget()
        w.bind("<MouseWheel>",      self._wheel,    add="+")
        w.bind("<Button-4>",        self._sup,      add="+")
        w.bind("<Button-5>",        self._sdn,      add="+")
        w.bind("<Button-2>",        self._pstart,   add="+")
        w.bind("<B2-Motion>",       self._pmove,    add="+")
        w.bind("<ButtonRelease-2>", self._pend,     add="+")

    def _in(self, tx, ty):
        h  = self.cv.get_tk_widget().winfo_height()
        bb = self.ax.get_window_extent()
        return bb.x0 <= tx <= bb.x1 and bb.y0 <= (h-ty) <= bb.y1

    def _data(self, tx, ty):
        h = self.cv.get_tk_widget().winfo_height()
        return self.ax.transData.inverted().transform((tx, h-ty))

    def _zoom(self, tx, ty, f):
        if not self._in(tx, ty): return
        cx, cy = self._data(tx, ty)
        xl, xr = self.ax.get_xlim(); yl, yr = self.ax.get_ylim()
        self.ax.set_xlim(cx+(xl-cx)*f, cx+(xr-cx)*f)
        self.ax.set_ylim(cy+(yl-cy)*f, cy+(yr-cy)*f)
        self._cb()

    def _wheel(self, e):
        self._zoom(e.x, e.y, 1/ZOOM_FACTOR if e.delta > 0 else ZOOM_FACTOR)
    def _sup(self, e): self._zoom(e.x, e.y, 1/ZOOM_FACTOR)
    def _sdn(self, e): self._zoom(e.x, e.y,   ZOOM_FACTOR)

    def _pstart(self, e):
        if self._in(e.x, e.y):
            self._p0 = (e.x, e.y,
                        self.ax.get_xlim(), self.ax.get_ylim())
    def _pmove(self, e):
        if self._p0 is None: return
        x0,y0,xl,yl = self._p0
        bb = self.ax.get_window_extent()
        if bb.width < 1 or bb.height < 1: return
        dx = (e.x-x0)/bb.width  * (xl[1]-xl[0])
        dy = (e.y-y0)/bb.height * (yl[1]-yl[0])
        self.ax.set_xlim(xl[0]-dx, xl[1]-dx)
        self.ax.set_ylim(yl[0]+dy, yl[1]+dy)
        self._cb()
    def _pend(self, e): self._p0 = None


# ══════════════════════════════════════════════════════════════════════════════
#  Zstd decompressor (used by read_czi for compressed CZI files)
#
#  Key facts learned from debugging:
#    - The zstd frame magic (28 B5 2F FD) starts inside the subblock XML
#      metadata buffer tail, so meta+data must be concatenated before searching.
#    - RFC 8878 FCS field sizes: flag 0→0, 1→1, 2→4, 3→8 bytes (not 2!).
#    - Use ZSTD_decompressStream (streaming API) so the embedded frame
#      boundary is detected correctly.
#    - comp=6 in ZEN 3.x = zstd with NO byte-plane split: decompressed bytes
#      are plain little-endian uint16.  (File metadata confirms "Zstd1".)
#    - comp=7 = zstd WITH byte-plane split: [high_bytes|low_bytes] → uint16.
# ══════════════════════════════════════════════════════════════════════════════

_ZSTD_MAGIC = bytes([0x28, 0xB5, 0x2F, 0xFD])

# ── zstd decompressor ─────────────────────────────────────────────────────────
#
# On macOS the ctypes/streaming-API path can silently return 0 bytes due to
# ABI differences in how struct pointers are passed to libzstd on ARM64.
# The pure-Python 'zstandard' package is always correct on every platform,
# so it is tried FIRST.  The ctypes path is kept as a fallback for systems
# where zstandard is not installed but libzstd is present.
#
# Install zstandard:  pip install zstandard
# Install libzstd:    brew install zstd  /  apt install libzstd1

def _zstd_decompress(blob: bytes) -> bytes:
    """
    Decompress one zstd frame embedded anywhere inside *blob*.
    The zstd frame magic (28 B5 2F FD) is searched for first; everything
    before it (e.g. the CZI subblock XML) is ignored.
    """
    idx = blob.find(_ZSTD_MAGIC)
    if idx < 0:
        raise ValueError("zstd magic (28 B5 2F FD) not found in subblock buffer")
    frame = blob[idx:]

    # ── Primary: pure-Python zstandard package ────────────────────────────
    # Correctly handles embedded frames on every OS and CPU architecture.
    if _ZSTD_PY_CTX is not None:
        return _ZSTD_PY_CTX.decompress(frame, max_output_size=8 * 1024 * 1024)

    # ── Fallback: ctypes / system libzstd ────────────────────────────────
    # Try common locations, including macOS Homebrew paths that are not in
    # PATH when the app is launched from Finder or an IDE.
    global _ZSTD_CTYPES
    if _ZSTD_CTYPES is None:
        candidates = [ctypes.util.find_library("zstd")]
        if sys.platform == "darwin":
            candidates += [
                "/opt/homebrew/lib/libzstd.dylib",   # Apple Silicon Homebrew
                "/usr/local/lib/libzstd.dylib",       # Intel Homebrew
                "/opt/local/lib/libzstd.dylib",       # MacPorts
            ]
            conda = os.environ.get("CONDA_PREFIX", "")
            if conda:
                candidates.append(os.path.join(conda, "lib", "libzstd.dylib"))
        for name in candidates:
            if not name:
                continue
            try:
                lib = ctypes.CDLL(name)
                lib.ZSTD_createDStream.restype  = ctypes.c_void_p
                lib.ZSTD_createDStream.argtypes = []
                lib.ZSTD_initDStream.restype    = ctypes.c_size_t
                lib.ZSTD_initDStream.argtypes   = [ctypes.c_void_p]
                lib.ZSTD_freeDStream.restype    = ctypes.c_size_t
                lib.ZSTD_freeDStream.argtypes   = [ctypes.c_void_p]
                lib.ZSTD_isError.restype        = ctypes.c_uint
                lib.ZSTD_isError.argtypes       = [ctypes.c_size_t]

                class _In(ctypes.Structure):
                    _fields_ = [("src",  ctypes.c_void_p),
                                 ("size", ctypes.c_size_t),
                                 ("pos",  ctypes.c_size_t)]
                class _Out(ctypes.Structure):
                    _fields_ = [("dst",  ctypes.c_void_p),
                                 ("size", ctypes.c_size_t),
                                 ("pos",  ctypes.c_size_t)]
                lib.ZSTD_decompressStream.restype  = ctypes.c_size_t
                lib.ZSTD_decompressStream.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(_Out),
                    ctypes.POINTER(_In),
                ]
                _ZSTD_CTYPES = (lib, _In, _Out)
                break
            except OSError:
                continue

    if _ZSTD_CTYPES is not None:
        lib, InBuf, OutBuf = _ZSTD_CTYPES
        # Parse Frame Content Size from header (RFC 8878 §3.1.1.1.1)
        fhd        = frame[4]
        fcs_flag   = (fhd >> 6) & 3
        single_seg = (fhd >> 5) & 1
        dict_flag  = fhd & 3
        hoff = 5
        if not single_seg:
            hoff += 1
        hoff += [0, 1, 2, 4][dict_flag]
        fcs_nb = [0, 1, 4, 8][fcs_flag]
        if single_seg and fcs_flag == 0:
            fcs_nb = 1
        fcs = 0
        if fcs_nb:
            fcs = int.from_bytes(frame[hoff:hoff + fcs_nb], "little")
            if fcs_flag == 1:
                fcs += 256
        out_size = max(fcs, 1) + 1024
        src_buf = (ctypes.c_char * len(frame)).from_buffer_copy(frame)
        dst_buf = ctypes.create_string_buffer(out_size)
        ds  = lib.ZSTD_createDStream()
        lib.ZSTD_initDStream(ds)
        ib  = InBuf(src=ctypes.cast(src_buf, ctypes.c_void_p),
                    size=len(frame), pos=0)
        ob  = OutBuf(dst=ctypes.cast(dst_buf, ctypes.c_void_p),
                     size=out_size, pos=0)
        ret = lib.ZSTD_decompressStream(ds, ctypes.byref(ob), ctypes.byref(ib))
        lib.ZSTD_freeDStream(ds)
        if ob.pos > 0:
            return bytes(dst_buf.raw[:ob.pos])
        # ob.pos == 0 means decompressor wrote nothing — ABI problem on this platform.
        # Raise so the user sees a clear message instead of a black image.
        raise RuntimeError(
            f"libzstd decompression wrote 0 bytes (ret={ret}). "
            f"Install zstandard instead:  pip install zstandard"
        )

    raise RuntimeError(
        "Cannot decompress zstd-compressed CZI tiles.\n\n"
        "Run:  pip install zstandard\n\n"
        "Then restart the application."
    )

_ZSTD_CTYPES = None  # (lib, InBuf, OutBuf) — loaded on first use if zstandard absent


# ══════════════════════════════════════════════════════════════════════════════
#  File readers
# ══════════════════════════════════════════════════════════════════════════════

def read_czi(path):
    """
    Read a Zeiss CZI file and return (rgb, px_um, stage_cx, stage_cy).

    rgb      : uint8 (1024, 1024, 3) composite — R=fluo+trans, G=fluo, B=fluo/2
    px_um    : pixel spacing in µm
    stage_cx : stage X position of the image centre (µm)
    stage_cy : stage Y position of the image centre (µm)

    Supports compression types:
      0  Uncompressed uint16
      6  Zstd, raw little-endian uint16 (ZEN 3.x "Zstd1")
      7  Zstd, byte-plane split (rare)
    """
    fsize = os.path.getsize(path)

    # ── 1. Walk segments ──────────────────────────────────────────────────
    segs = []
    with open(path, "rb") as f:
        off = 0
        while off + 32 < fsize:
            f.seek(off)
            sid   = f.read(16).rstrip(b"\x00").decode("ascii", errors="replace")
            alloc = struct.unpack("<q", f.read(8))[0]
            f.read(8)
            if alloc <= 0: break
            segs.append((sid, off, alloc))
            off += 32 + alloc

    # ── 2. Read ZISRAWDIRECTORY → Z, C, compression per subblock ──────────
    #
    # IMPORTANT: for compressed (zstd) subblocks the metadata blob does NOT
    # contain a parseable fixed-format dim table — the blob starts with zstd
    # XML.  Z/C must be read from the directory, not re-parsed from the meta.
    #
    info_map = {}   # file_pos → (z, c, compression)
    with open(path, "rb") as f:
        for sid, off, alloc in segs:
            if sid != "ZISRAWDIRECTORY": continue
            f.seek(off + 32)
            n = struct.unpack("<i", f.read(4))[0]; f.read(124)
            for _ in range(n):
                f.read(2)                                 # schema
                f.read(4)                                 # pixel_type
                fpos = struct.unpack("<q", f.read(8))[0]  # file_pos
                f.read(4)                                  # file_part
                comp = struct.unpack("<i", f.read(4))[0]  # compression
                f.read(6)                                  # pyramid + spare
                nd   = struct.unpack("<i", f.read(4))[0]
                dims = {}
                for _ in range(nd):
                    did   = f.read(4).rstrip(b"\x00").decode("ascii", errors="replace")
                    start = struct.unpack("<i", f.read(4))[0]
                    f.read(4); f.read(4); f.read(4)        # size, stored, float
                    dims[did] = start
                info_map[fpos] = (dims.get("Z", 0), dims.get("C", 0), comp)

    # ── 3. Pixel spacing from ZISRAWMETADATA ─────────────────────────────
    px_um = 2.495703125   # Zeiss LSM default; overridden below if found
    with open(path, "rb") as f:
        for sid, off, alloc in segs:
            if sid != "ZISRAWMETADATA": continue
            f.seek(off + 32)
            xs = struct.unpack("<i", f.read(4))[0]; f.read(252)
            xml = f.read(xs).decode("utf-8", errors="replace")
            try:
                root = ET.fromstring(xml)
                for dist in root.findall(".//Distance"):
                    if dist.get("Id") == "X":
                        px_um = float(dist.findtext("Value", "0")) * 1e6
            except Exception:
                pass
            break

    # ── 4. Read and decode each subblock ─────────────────────────────────
    #
    # ZISRAWSUBBLOCK payload layout (after 32-byte segment header):
    #   INT32    meta_size
    #   INT32    attach_size
    #   INT64    data_size
    #   BYTE[256] reserved
    #   BYTE[meta_size]  XML metadata    ← stage position (XML)
    #   BYTE[data_size]  pixel data      ← compressed or raw
    #
    # Z/C are taken from info_map (directory), not the meta blob.
    #
    planes   = {}
    stage_cx = stage_cy = None

    with open(path, "rb") as f:
        for sid, off, alloc in segs:
            if sid != "ZISRAWSUBBLOCK": continue
            z, c, comp = info_map.get(off, (0, 0, 0))

            f.seek(off + 32)
            msz = struct.unpack("<I", f.read(4))[0]  # meta size
            f.read(4)                                 # attach_size (skip)
            dsz = struct.unpack("<q", f.read(8))[0]  # data size
            f.read(256)                               # reserved
            meta = bytes(f.read(msz))
            data = bytes(f.read(dsz))

            # Stage position from XML in first subblock
            if stage_cx is None:
                xs = meta.find(b"<")
                if xs >= 0:
                    s  = meta[xs:].decode("utf-8", errors="replace")
                    mx = re.search(r"<StageXPosition>([^<]+)", s)
                    my = re.search(r"<StageYPosition>([^<]+)", s)
                    if mx and my:
                        stage_cx = float(mx.group(1))
                        stage_cy = float(my.group(1))

            # Decode pixels → float32 (1024, 1024)
            N = 1024 * 1024
            try:
                if comp == 6:
                    # Zstd, raw LE uint16 — no byte-plane split
                    # (zstd frame magic lives in the meta tail; pass meta+data)
                    raw = _zstd_decompress(meta + data)
                    arr = np.frombuffer(raw[:N * 2], dtype="<u2") \
                            .reshape(1024, 1024).astype(np.float32)
                elif comp == 7:
                    # Zstd, byte-plane split: [high_bytes | low_bytes]
                    raw = _zstd_decompress(meta + data)
                    a   = np.frombuffer(raw, np.uint8)
                    arr = ((a[:N].astype(np.uint16) << 8) | a[N:2*N]) \
                            .reshape(1024, 1024).astype(np.float32)
                else:
                    # Uncompressed uint16 little-endian
                    arr = np.frombuffer(data[:N * 2], dtype="<u2") \
                            .reshape(1024, 1024).astype(np.float32)
            except Exception as e:
                # Re-raise so the CZI error dialog is shown to the user.
                # Silent warnings leave the image black with no explanation.
                raise RuntimeError(
                    f"Failed to decode tile Z={z} C={c} "
                    f"(compression={comp}): {e}") from e

            key = (z, c)
            planes[key] = np.maximum(planes.get(key, np.zeros_like(arr)), arr)

    # ── 5. Max-Z projection + composite (identical to original) ──────────
    def norm(a):
        # Low clip: p0.35 of all pixels
        p_lo = np.percentile(a, 0.35)
        # High clip: p99.65 of non-saturated pixels.
        # Using all pixels anchors the high end at 65535 (hard saturation),
        # which compresses all real fluorescence signal into the bottom ~30%
        # of the 8-bit range.  Excluding saturated pixels gives ~3× better contrast.
        non_sat = a[a < 65535] if (a == 65535).any() else a
        p_hi = np.percentile(non_sat, 99.65) if non_sat.size > 0 else a.max()
        d = p_hi - p_lo
        return np.clip((a - p_lo) / d * 255, 0, 255).astype(np.uint8) if d > 0 \
               else np.zeros_like(a, np.uint8)

    zs  = sorted({k[0] for k in planes})
    mc0 = np.zeros((1024, 1024), np.float32)
    mc1 = np.zeros((1024, 1024), np.float32)
    for z in zs:
        if (z, 0) in planes: mc0 = np.maximum(mc0, planes[(z, 0)])
        if (z, 1) in planes: mc1 = np.maximum(mc1, planes[(z, 1)])
    n0, n1 = norm(mc0), norm(mc1)
    rgb = np.zeros((1024, 1024, 3), np.uint8)
    rgb[:, :, 0] = np.clip(n0.astype(np.int32) + n1.astype(np.int32), 0, 255).astype(np.uint8)
    rgb[:, :, 1] = n0
    rgb[:, :, 2] = (n0 // 2).astype(np.uint8)
    return rgb, px_um, stage_cx or 0.0, stage_cy or 0.0, mc0, mc1


def read_a5proj(path):
    root = ET.parse(path).getroot()
    out  = []
    for img in root.iter("FrameServedImage"):
        name = img.findtext("Name","").strip()
        if "Airy" not in name: continue
        fname = img.findtext("FileName","").strip().split("\\")[-1]
        tf    = img.find("ParentTransform")
        if tf is None: continue
        # M41/M42 = image CENTRE in Atlas5 canvas µm (CenterLocalX/Y = 0.5).
        # M11 = full FOV width in µm (local axis [0,1] maps to M11 µm of canvas).
        # M22 = -M11: square image with Y-axis flipped in Atlas5 canvas.
        m41 = float(tf.findtext("M41","0"))
        m42 = float(tf.findtext("M42","0"))
        m11 = abs(float(tf.findtext("M11","1")))
        out.append(dict(name=name, file=fname,
                        stage_x=m41 + m11/2, stage_y=m42 - m11/2, fov_um=round(m11, 1)))
    return sorted(out, key=lambda x: x["name"])


# ══════════════════════════════════════════════════════════════════════════════
#  Application
# ══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):

    IDLE=0; CENTRE=1; XAXIS=2

    def __init__(self):
        super().__init__()
        self.title("Grid Calibration Tool")
        self.configure(bg=BG)
        self.minsize(1180, 740)

        # data
        self._rgb      = None
        self._mc0      = None   # raw float32 max-Z channel 0 (fluorescence)
        self._mc1      = None   # raw float32 max-Z channel 1 (transmitted)
        self._px_um    = 2.495703125
        self._stage_cx = 0.0
        self._stage_cy = 0.0
        self._airy     = []

        # brightness / contrast (slider values 0-255; mapped to 16-bit range)
        self._bright   = 0
        self._contrast = 255
        self._bc_lo    = 0.0
        self._bc_hi    = 65535.0

        # calibration
        self._step      = self.IDLE
        self._centre_px = None   # (imgx, imgy)
        self._xdir_px   = None   # (imgx, imgy)
        self._angle_deg = 0.0
        self._results   = []
        self._sel_idx   = None

        # view state – updated by pan/zoom callback and before every redraw
        self._xlim = (-0.5, 1023.5)
        self._ylim = (1023.5, -0.5)

        self._build_styles()
        self._build_ui()

    # ── styles ────────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure("TFrame",            background=BG)
        s.configure("TLabel",            background=BG, foreground=FG, font=("Segoe UI",10))
        s.configure("Sm.TLabel",         background=BG, foreground=FG, font=("Segoe UI",9))
        s.configure("TButton",           background=ACC, foreground=BG, font=("Segoe UI",10,"bold"), padding=5)
        s.map("TButton",   background=[("active",CYA),("disabled",BG3)])
        s.configure("Accent.TButton",    background=ACC2, foreground=BG, font=("Segoe UI",11,"bold"), padding=7)
        s.configure("Danger.TButton",    background=RED, foreground=BG, font=("Segoe UI",10,"bold"), padding=5)
        s.configure("Step.TButton",      background=MAG, foreground=BG, font=("Segoe UI",10,"bold"), padding=6)
        s.configure("Active.TButton",    background=YEL, foreground=BG, font=("Segoe UI",10,"bold"), padding=6)
        s.map("Step.TButton",   background=[("active",CYA),("disabled",BG3)])
        s.map("Active.TButton", background=[("active",CYA),("disabled",BG3)])
        s.configure("TLabelframe",       background=BG, relief="groove")
        s.configure("TLabelframe.Label", background=BG, foreground=CYA, font=("Segoe UI",10,"bold"))
        s.configure("Treeview",          background=BG2, foreground=FG, fieldbackground=BG2, rowheight=22)
        s.configure("Treeview.Heading",  background=BG3, foreground=FG)
        s.map("Treeview", background=[("selected",ACC)])
        s.configure("TScale", background=BG, troughcolor=BG3, sliderlength=16)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        top = ttk.Frame(self, padding=(8,6,8,0)); top.pack(fill="x")
        ttk.Button(top, text="Load CZI + a5proj",
                   style="Accent.TButton",
                   command=self._load_both).pack(side="left", padx=3)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(top, text="Load CZI",    command=self._load_czi).pack(side="left", padx=3)
        ttk.Button(top, text="Load a5proj", command=self._load_a5proj).pack(side="left", padx=3)
        self._info_var = tk.StringVar(value="No files loaded")
        ttk.Label(top, textvariable=self._info_var,
                  style="Sm.TLabel", foreground=BG3).pack(side="left", padx=8)
        ttk.Label(top, text="scroll=zoom  middle-drag=pan  left=pick  right=undo",
                  style="Sm.TLabel", foreground=BG3).pack(side="right", padx=6)

        body = ttk.Frame(self, padding=(8,6)); body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1); body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        # image panel
        img_lf = ttk.LabelFrame(body,
            text="5× Overview  —  composite blend · Max-Z",
            padding=4)
        img_lf.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        img_lf.rowconfigure(0, weight=1); img_lf.columnconfigure(0, weight=1)

        self._fig = Figure(figsize=(7,6), facecolor=BG)
        self._ax  = self._fig.add_subplot(111)
        self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._ax.set_facecolor(BG); self._ax.axis("off")

        self._canvas = FigureCanvasTkAgg(self._fig, master=img_lf)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._canvas.draw()

        # pan/zoom: callback just saves new limits, then calls _redraw
        self._pz = PanZoom(self._ax, self._canvas, self._on_view_change)
        self._canvas.mpl_connect("button_press_event",  self._on_click)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)

        # brightness / contrast panel — below the image
        bc_lf = ttk.LabelFrame(body,
            text="Display  —  Ch0 fluorescence (G channel)  +  Ch1 transmitted (mixed into R)",
            padding=(8, 4))
        bc_lf.grid(row=1, column=0, sticky="ew", padx=(0,6), pady=(4,0))
        bc_lf.columnconfigure(1, weight=1)

        ttk.Label(bc_lf, text="Black pt", style="Sm.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0,6))
        self._bright_var = tk.IntVar(value=0)
        self._bright_scale = ttk.Scale(
            bc_lf, from_=0, to=254, orient="horizontal",
            variable=self._bright_var, command=self._on_bc_change)
        self._bright_scale.grid(row=0, column=1, sticky="ew")
        self._bright_lbl = ttk.Label(bc_lf, text="  0", style="Sm.TLabel",
                                      foreground=CYA, width=5)
        self._bright_lbl.grid(row=0, column=2)

        ttk.Label(bc_lf, text="White pt", style="Sm.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0,6))
        self._contrast_var = tk.IntVar(value=255)
        self._contrast_scale = ttk.Scale(
            bc_lf, from_=1, to=255, orient="horizontal",
            variable=self._contrast_var, command=self._on_bc_change)
        self._contrast_scale.grid(row=1, column=1, sticky="ew")
        self._contrast_lbl = ttk.Label(bc_lf, text="255", style="Sm.TLabel",
                                        foreground=CYA, width=5)
        self._contrast_lbl.grid(row=1, column=2)

        ttk.Button(bc_lf, text="Auto", command=self._bc_auto).grid(
            row=0, column=3, rowspan=2, padx=(8,0))

        # right panel spans both rows so it fills the full height
        right = ttk.Frame(body, width=300)
        right.grid(row=0, column=1, sticky="nsew", rowspan=2)
        right.columnconfigure(0, weight=1)
        self._build_right(right)

        # status bar
        sf = ttk.Frame(self, padding=(8,2,8,4)); sf.pack(fill="x")
        ttk.Separator(sf, orient="horizontal").pack(fill="x", pady=(0,3))
        self._sv = tk.StringVar(value="Load files to begin.")
        ttk.Label(sf, textvariable=self._sv,
                  style="Sm.TLabel", foreground=CYA).pack(side="left")

    def _build_right(self, p):
        r = 0
        # step 1
        lf = ttk.LabelFrame(p, text="Step 1 · Grid Centre", padding=(8,4))
        lf.grid(row=r, column=0, sticky="ew", pady=(0,6)); r+=1
        ttk.Label(lf, text="Left-click the central grid feature.\nThis becomes (0,0).",
                  style="Sm.TLabel", justify="left").pack(anchor="w")
        self._btn_cen = ttk.Button(lf, text="Pick Centre",
                                    style="Step.TButton", command=self._act_centre)
        self._btn_cen.pack(fill="x", pady=(6,2))
        self._cen_var = tk.StringVar(value="Centre:  not set")
        ttk.Label(lf, textvariable=self._cen_var,
                  style="Sm.TLabel", foreground=CYA).pack(anchor="w")

        # step 2
        lf2 = ttk.LabelFrame(p, text="Step 2 · +X Direction", padding=(8,4))
        lf2.grid(row=r, column=0, sticky="ew", pady=(0,6)); r+=1
        ttk.Label(lf2, text="Click a point in the +X direction\nfrom the centre.",
                  style="Sm.TLabel", justify="left").pack(anchor="w")
        self._btn_xax = ttk.Button(lf2, text="Pick +X Direction",
                                    style="Step.TButton", command=self._act_xaxis)
        self._btn_xax.pack(fill="x", pady=(6,2))
        self._xax_var = tk.StringVar(value="X-axis:  not set")
        self._ang_var = tk.StringVar(value="Angle:   —")
        ttk.Label(lf2, textvariable=self._xax_var,
                  style="Sm.TLabel", foreground=CYA).pack(anchor="w")
        ttk.Label(lf2, textvariable=self._ang_var,
                  style="Sm.TLabel", foreground=YEL).pack(anchor="w")

        # step 3
        lf3 = ttk.LabelFrame(p, text="Step 3 · Compute & Export", padding=(8,4))
        lf3.grid(row=r, column=0, sticky="ew", pady=(0,6)); r+=1
        self._btn_cmp = ttk.Button(lf3, text="Compute Grid Coordinates",
                                    style="Accent.TButton",
                                    command=self._compute, state="disabled")
        self._btn_cmp.pack(fill="x", pady=(0,4))
        self._btn_exp = ttk.Button(lf3, text="Export CSV",
                                    command=self._export, state="disabled")
        self._btn_exp.pack(fill="x", pady=(0,4))
        ttk.Button(lf3, text="Reset Calibration",
                   style="Danger.TButton", command=self._reset).pack(fill="x")

        # results table
        lf4 = ttk.LabelFrame(p, text="Grid-Relative Distances (µm)", padding=(4,4))
        lf4.grid(row=r, column=0, sticky="nsew"); r+=1
        p.rowconfigure(r-1, weight=1)
        lf4.rowconfigure(0, weight=1); lf4.columnconfigure(0, weight=1)
        cols = ("Name","Grid X (µm)","Grid Y (µm)")
        self._tree = ttk.Treeview(lf4, columns=cols, show="headings")
        for c,w in zip(cols,[90,100,100]):
            self._tree.heading(c, text=c); self._tree.column(c, width=w, anchor="center")
        vsb = ttk.Scrollbar(lf4, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self._tree.bind("<<TreeviewSelect>>", self._on_sel)

    # ── view change callback (pan/zoom) ───────────────────────────────────
    def _on_view_change(self):
        self._xlim = self._ax.get_xlim()
        self._ylim = self._ax.get_ylim()
        self._redraw()

    # ── THE ONLY DRAW FUNCTION ────────────────────────────────────────────
    def _redraw(self):
        ax = self._ax
        ax.cla()
        ax.set_facecolor(BG)
        ax.axis("off")

        if self._rgb is not None:
            ax.imshow(self._rgb, origin="upper", aspect="equal",
                      extent=[-0.5,1023.5,1023.5,-0.5], zorder=0,
                      interpolation="nearest")

        # always restore stored view
        ax.set_xlim(self._xlim)
        ax.set_ylim(self._ylim)

        # Airy boxes — shown before compute; after compute the result loop redraws them
        result_names = {r["name"] for r in self._results}
        for a in self._airy:
            if a["name"] in result_names:
                continue   # result loop will draw this one
            px, py = self._s2p(a["stage_x"], a["stage_y"])
            hw     = (a["fov_um"] / 2) / self._px_um
            ax.add_patch(mpatches.Rectangle(
                (px-hw, py-hw), hw*2, hw*2,
                lw=0.9, edgecolor=ACC, facecolor=ACC+"18", zorder=3))
            ax.plot(px, py, "o", color=ACC, ms=4, mec=BG, mew=0.6, zorder=4)

        # grid centre
        if self._centre_px:
            cx,cy = self._centre_px; arm=20
            kw = dict(color=ACC2, lw=2.0, zorder=6,
                      path_effects=[pe.Stroke(linewidth=4,foreground=BG), pe.Normal()])
            ax.plot([cx-arm,cx+arm],[cy,cy],   **kw)
            ax.plot([cx,cx],[cy-arm,cy+arm],   **kw)
            ax.plot(cx, cy, "o", color=ACC2, ms=9, mec=BG, mew=1.2, zorder=7)
            ax.text(cx+14, cy-14, "ORIGIN", color=ACC2,
                    fontsize=8, fontweight="bold", zorder=8,
                    path_effects=[pe.Stroke(linewidth=2,foreground=BG), pe.Normal()])

        # +X click marker
        if self._xdir_px:
            px,py = self._xdir_px
            ax.plot(px, py, "D", color=CYA, ms=8, mec=BG, mew=1.2, zorder=6)
            ax.text(px+10, py-10, "+X dir", color=CYA,
                    fontsize=8, fontweight="bold", zorder=7,
                    path_effects=[pe.Stroke(linewidth=2,foreground=BG), pe.Normal()])

        # +X and +Y axis arrows from centre
        if self._centre_px and self._xdir_px:
            cx, cy = self._centre_px
            arm    = 1024 * 0.14
            rad    = math.radians(self._angle_deg)
            for dang, col, lbl in [(0, CYA, "+X"), (math.pi/2, YEL, "+Y")]:
                a  = rad + dang
                ex = cx + arm * math.cos(a)
                ey = cy - arm * math.sin(a)
                ax.plot([cx, ex], [cy, ey], color=BG, lw=5,
                        solid_capstyle="round", zorder=9)
                ax.annotate("",
                    xy=(ex, ey), xytext=(cx, cy), zorder=10,
                    arrowprops=dict(
                        arrowstyle="->, head_width=0.35, head_length=0.45",
                        color=col, lw=2.5, shrinkA=0, shrinkB=0))
                ax.text(ex+7, ey-7, lbl, color=col,
                        fontsize=11, fontweight="bold", zorder=11,
                        path_effects=[pe.Stroke(linewidth=3, foreground=BG),
                                      pe.Normal()])

        # result markers — blue boxes; selected box turns green
        for i, res in enumerate(self._results):
            px, py = self._s2p(res["stage_x"], res["stage_y"])
            hw     = (res["fov_um"] / 2) / self._px_um
            if i == self._sel_idx:
                ax.add_patch(mpatches.Rectangle(
                    (px-hw, py-hw), hw*2, hw*2,
                    lw=2.0, edgecolor=ACC2, facecolor=ACC2+"40", zorder=8))
                ax.plot(px, py, "o", color=ACC2, ms=8,
                        mec="white", mew=1.2, zorder=9)
                ax.text(px+hw+4, py, res["name"], color=ACC2,
                        fontsize=8, fontweight="bold", zorder=10,
                        path_effects=[pe.Stroke(linewidth=2, foreground=BG),
                                      pe.Normal()])
            else:
                ax.add_patch(mpatches.Rectangle(
                    (px-hw, py-hw), hw*2, hw*2,
                    lw=0.9, edgecolor=ACC, facecolor=ACC+"18", zorder=3))
                ax.plot(px, py, "o", color=ACC, ms=4,
                        mec=BG, mew=0.6, zorder=4)

        self._canvas.draw()

    # ── coordinate helpers ────────────────────────────────────────────────
    def _s2p(self, sx, sy):
        return ((sx-self._stage_cx)/self._px_um+512,
                (sy-self._stage_cy)/self._px_um+512)

    def _p2s(self, px, py):
        return (self._stage_cx+(px-512)*self._px_um,
                self._stage_cy+(py-512)*self._px_um)

    def _p2g(self, px, py):
        if self._centre_px is None or self._xdir_px is None: return None,None
        cx_s,cy_s = self._p2s(*self._centre_px)
        sx,sy     = self._p2s(px,py)
        t         = math.radians(self._angle_deg)
        dsx,dsy   = sx-cx_s, sy-cy_s
        return ( dsx*math.cos(t)+dsy*math.sin(t),
                -dsx*math.sin(t)+dsy*math.cos(t))

    # ── loading ───────────────────────────────────────────────────────────
    def _load_both(self):
        p1 = filedialog.askopenfilename(title="Open CZI",
            filetypes=[("CZI","*.czi"),("All","*.*")])
        if not p1: return
        p2 = filedialog.askopenfilename(title="Open a5proj",
            filetypes=[("a5proj","*.a5proj"),("All","*.*")])
        if not p2: return
        self._do_czi(p1); self._do_a5(p2)

    def _load_czi(self):
        p = filedialog.askopenfilename(title="Open CZI",
            filetypes=[("CZI","*.czi"),("All","*.*")])
        if p: self._do_czi(p)

    def _load_a5proj(self):
        p = filedialog.askopenfilename(title="Open a5proj",
            filetypes=[("a5proj","*.a5proj"),("All","*.*")])
        if p: self._do_a5(p)

    def _do_czi(self, path):
        self._sv.set(f"Loading {os.path.basename(path)} …")
        self.update_idletasks()
        try:
            rgb,pxum,scx,scy,mc0,mc1 = read_czi(path)
            self._rgb=rgb; self._mc0=mc0; self._mc1=mc1
            self._px_um=pxum
            self._stage_cx=scx; self._stage_cy=scy
            self._xlim=(-0.5,1023.5); self._ylim=(1023.5,-0.5)
            self._bc_auto(redraw=False)
            self._update_info()
            self._redraw()
            self._sv.set(f"CZI loaded — stage ({scx:.0f},{scy:.0f}) µm")
        except Exception as e:
            messagebox.showerror("CZI error", str(e))

    def _do_a5(self, path):
        try:
            self._airy = read_a5proj(path)
            self._update_info(); self._check_ready(); self._redraw()
            self._sv.set(f"a5proj loaded — {len(self._airy)} Airy images")
        except Exception as e:
            messagebox.showerror("a5proj error", str(e))

    # ── brightness / contrast ─────────────────────────────────────────────
    def _apply_bc(self):
        if self._mc0 is None:
            return
        lo = float(self._bright)
        hi = float(self._contrast)
        if hi <= lo:
            hi = lo + 1.0
        # Map slider 0-255 back to 16-bit range using the auto-stretch anchors
        span = self._bc_hi - self._bc_lo
        lo16 = self._bc_lo + span * lo  / 255.0
        hi16 = self._bc_lo + span * hi  / 255.0
        if hi16 <= lo16:
            hi16 = lo16 + 1.0
        def s16(a):
            return np.clip((a - lo16) / (hi16 - lo16) * 255,
                           0, 255).astype(np.uint8)
        n0 = s16(self._mc0)
        n1 = s16(self._mc1) if self._mc1 is not None else np.zeros_like(n0)
        rgb = np.zeros((1024, 1024, 3), np.uint8)
        rgb[:,:,0] = np.clip(n0.astype(np.int32)+n1.astype(np.int32),
                             0,255).astype(np.uint8)
        rgb[:,:,1] = n0
        rgb[:,:,2] = (n0//2).astype(np.uint8)
        self._rgb = rgb

    def _bc_auto(self, redraw=True):
        if self._mc0 is None:
            return
        a = self._mc0
        lo = float(np.percentile(a, 0.35))
        non_sat = a[a < 65535] if (a == 65535).any() else a
        hi = float(np.percentile(non_sat, 99.65) if non_sat.size > 0 else a.max())
        self._bc_lo = lo
        self._bc_hi = max(hi, lo + 1.0)
        self._bright   = 0;   self._contrast = 255
        self._bright_var.set(0);  self._contrast_var.set(255)
        self._bright_lbl.configure(text="  0")
        self._contrast_lbl.configure(text="255")
        self._apply_bc()
        if redraw:
            self._redraw()

    def _on_bc_change(self, _=None):
        b = int(self._bright_var.get())
        c = int(self._contrast_var.get())
        if b >= c:
            c = min(b + 1, 255)
            self._contrast_var.set(c)
        self._bright   = b;  self._contrast = c
        self._bright_lbl.configure(text=f"{b:3d}")
        self._contrast_lbl.configure(text=f"{c:3d}")
        self._apply_bc()
        self._redraw()

    def _update_info(self):
        parts=[]
        if self._rgb is not None:
            parts.append(f"CZI ✓ ({self._stage_cx:.0f},{self._stage_cy:.0f}) µm")
        if self._airy:
            parts.append(f"a5proj ✓ {len(self._airy)} images")
        self._info_var.set("  |  ".join(parts) if parts else "No files loaded")

    # ── calibration steps ─────────────────────────────────────────────────
    def _set_step(self, step):
        self._step = step
        self._btn_cen.configure(
            style="Active.TButton" if step==self.CENTRE else "Step.TButton")
        self._btn_xax.configure(
            style="Active.TButton" if step==self.XAXIS  else "Step.TButton")

    def _check_ready(self):
        ok = self._centre_px and self._xdir_px and self._airy
        self._btn_cmp.configure(state="normal" if ok else "disabled")

    def _act_centre(self):
        if self._rgb is None:
            messagebox.showwarning("No image","Load a CZI first."); return
        self._set_step(self.CENTRE)
        self._sv.set("Click the central TEM grid feature to set (0,0).")

    def _act_xaxis(self):
        if self._rgb is None:
            messagebox.showwarning("No image","Load a CZI first."); return
        if not self._centre_px:
            messagebox.showwarning("No centre","Set the grid centre first."); return
        self._xdir_px = None
        self._set_step(self.XAXIS)
        self._xax_var.set("X-axis:  click a point in +X direction")
        self._sv.set("Click a point in the +X direction from the centre.")

    # ── click ─────────────────────────────────────────────────────────────
    def _on_click(self, event):
        if event.inaxes is not self._ax or event.xdata is None: return
        px,py = event.xdata, event.ydata

        if event.button == 3:
            if self._step == self.XAXIS:
                self._xdir_px=None
                self._xax_var.set("X-axis:  not set"); self._ang_var.set("Angle:   —")
                self._redraw()
            elif self._step == self.CENTRE:
                self._centre_px=None
                self._cen_var.set("Centre:  not set"); self._redraw()
            return

        if event.button != 1: return

        if self._step == self.CENTRE:
            self._centre_px = (px, py)
            sx,sy = self._p2s(px,py)
            self._cen_var.set(
                f"Centre: img({px:.0f},{py:.0f})  stage({sx:.1f},{sy:.1f}) µm")
            self._set_step(self.IDLE); self._check_ready(); self._redraw()
            self._sv.set("Origin set. Now click 'Pick +X Direction'.")

        elif self._step == self.XAXIS:
            self._xdir_px   = (px, py)
            cx,cy           = self._centre_px
            self._angle_deg = math.degrees(math.atan2(-(py-cy), px-cx))
            self._xax_var.set(f"X-axis: centre→({px:.0f},{py:.0f})")
            self._ang_var.set(f"Angle: {self._angle_deg:.3f}° (CCW from image +X)")
            self._set_step(self.IDLE); self._check_ready(); self._redraw()
            self._sv.set(f"+X set at {self._angle_deg:.2f}°. Click 'Compute'.")

    # ── motion ────────────────────────────────────────────────────────────
    def _on_motion(self, event):
        if event.inaxes is not self._ax or event.xdata is None: return
        px,py = event.xdata, event.ydata
        sx,sy = self._p2s(px,py)
        gx,gy = self._p2g(px,py)
        if gx is not None:
            self._sv.set(
                f"Stage ({sx:.1f},{sy:.1f}) µm   "
                f"Grid ({gx:+.1f},{gy:+.1f}) µm   "
                f"dist {math.sqrt(gx**2+gy**2):.1f} µm")
        else:
            self._sv.set(f"Stage ({sx:.1f},{sy:.1f}) µm")

    # ── compute ───────────────────────────────────────────────────────────
    def _compute(self):
        cx_s,cy_s = self._p2s(*self._centre_px)
        t         = math.radians(self._angle_deg)
        ct,st     = math.cos(t), math.sin(t)
        self._results = []
        for a in self._airy:
            dsx = a["stage_x"]-cx_s; dsy = a["stage_y"]-cy_s
            self._results.append(dict(
                name=a["name"], file=a["file"],
                stage_x=a["stage_x"], stage_y=a["stage_y"],
                fov_um=a["fov_um"],
                grid_x= dsx*ct+dsy*st,
                grid_y=-dsx*st+dsy*ct))
        self._refresh_tree()
        self._btn_exp.configure(state="normal")
        self._redraw()
        self._sv.set(
            f"Computed {len(self._results)} positions. "
            f"Origin: stage ({cx_s:.1f},{cy_s:.1f}) µm  "
            f"Angle: {self._angle_deg:.3f}°")

    def _refresh_tree(self):
        self._tree.delete(*self._tree.get_children())
        for r in self._results:
            self._tree.insert("","end",values=(
                r["name"],f"{r['grid_x']:+.1f}",f"{r['grid_y']:+.1f}"))

    # ── tree selection ────────────────────────────────────────────────────
    def _on_sel(self, _e):
        sel = self._tree.selection()
        if not sel:
            self._sel_idx = None; self._redraw(); return
        idx = self._tree.index(sel[0])
        if idx >= len(self._results): return
        self._sel_idx = idx
        r    = self._results[idx]
        px,py = self._s2p(r["stage_x"], r["stage_y"])
        xl,xr = self._ax.get_xlim(); yl,yr = self._ax.get_ylim()
        hw_x  = (xr-xl)/2;  hw_y = abs(yr-yl)/2
        self._xlim = (px-hw_x, px+hw_x)
        self._ylim = (py+hw_y, py-hw_y)
        self._redraw()

    # ── export ────────────────────────────────────────────────────────────
    def _export(self):
        if not self._results:
            messagebox.showwarning("Empty","Run Compute first."); return
        path = filedialog.asksaveasfilename(
            title="Save CSV", defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("All","*.*")])
        if not path: return
        try:
            cx_s,cy_s = self._p2s(*self._centre_px)
            with open(path,"w",newline="") as fh:
                w = csv.writer(fh)
                # w.writerow(["# grid_centre_stage_x_um", f"{cx_s:.4f}"])
                # w.writerow(["# grid_centre_stage_y_um", f"{cy_s:.4f}"])
                # w.writerow(["# grid_angle_deg",         f"{self._angle_deg:.6f}"])
                # w.writerow(["# pixel_size_um",          f"{self._px_um:.6f}"])
                # w.writerow([])
                w.writerow(["name","grid_x_um","grid_y_um"])
                for r in self._results:
                    w.writerow([r["name"],f"{r['grid_x']:+.3f}",f"{r['grid_y']:+.3f}"])
            messagebox.showinfo("Exported",
                f"Saved {len(self._results)} rows to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    # ── reset ─────────────────────────────────────────────────────────────
    def _reset(self):
        self._centre_px=None; self._xdir_px=None
        self._angle_deg=0.0;  self._results=[];  self._sel_idx=None
        self._cen_var.set("Centre:  not set")
        self._xax_var.set("X-axis:  not set"); self._ang_var.set("Angle:   —")
        self._btn_cmp.configure(state="disabled")
        self._btn_exp.configure(state="disabled")
        self._tree.delete(*self._tree.get_children())
        self._set_step(self.IDLE); self._redraw()
        self._sv.set("Calibration reset.")


if __name__ == "__main__":
    App().mainloop()