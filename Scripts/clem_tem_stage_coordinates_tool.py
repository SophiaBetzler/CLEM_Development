"""
CSV + MRC mdoc Coordinate Converter
=====================================
Workflow
--------
  1. Load CSV              - auto-detects first two numeric cols as X/Y (um),
                             first non-numeric col as label
  2. Load MRC + mdoc       - picks either file; sibling found automatically by
                             shared filename stem. Assembles montage from tiles.
  3. Pick Image Centre     - left-click the feature that is (0,0) in your CSV
  4. Pick +X Direction     - left-click any point in the +X direction
  5. (Optional) Pick +Y    - left-click any point in the +Y direction; if you
                             skip it, +Y defaults to 90 deg CCW from +X
  6. Convert               - fills result table with stage coordinates (um)
  7. Export CSV            - save converted coordinates
  8. Export PNG            - save the current view or full image as PNG

Coordinate convention
---------------------
  Input CSV  : X, Y in um, expressed in the CSV file's own coordinate frame.
               The drawn +X / +Y arrows show where that frame's axes point on
               the image.
  Output     : stage X (um) = image horizontal (+X = image right)
               stage Y (um) = image vertical   (+Y = image up / image -Y)
  The transform rotates the user-defined CSV axis frame into image space and
  shifts the origin to the chosen centre's interpolated stage position, then
  maps image pixels to stage um.

  Flip stage X / Flip stage Y mirror the *predicted stage* coordinates about
  the origin label (per axis).  They do NOT change the CSV coordinate frame or
  its drawn +X / +Y axes; use them when the grid is physically flipped between
  imaging modalities.

Mouse controls
--------------
  Scroll wheel       - zoom centred on cursor
  Middle-click drag  - pan
  Left-click         - place point (step-dependent)
  Right-click        - undo last point in current step

Dependencies
------------
  pip install numpy matplotlib mrcfile
  tkinter: part of stdlib (python3-tk on Debian/Ubuntu)
"""

import os, sys, re, csv, math
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patheffects as pe

try:
    import mrcfile
    _HAVE_MRCFILE = True
except ImportError:
    _HAVE_MRCFILE = False
    print("WARNING: mrcfile not installed - MRC display disabled.\n"
          "         pip install mrcfile", file=sys.stderr)

# -- palette -------------------------------------------------------------------
BG   = "#1e1e2e"; BG2 = "#313244"; BG3 = "#45475a"; FG  = "#cdd6f4"
ACC  = "#89b4fa"; ACC2= "#a6e3a1"; RED = "#f38ba8"; CYA = "#89dceb"
YEL  = "#f9e2af"; MAG = "#cba6f7"; ORG = "#fab387"
ZOOM_FACTOR = 1.25

# Large montages are displayed downsampled to at most this many px on the long
# axis (for speed/memory).  Coordinates stay in FULL-resolution pixels, so the
# picked centre and predicted stage positions remain exact regardless.
MAX_DISP_PX = 6000


# ==============================================================================
#  Pan / Zoom
# ==============================================================================
class PanZoom:
    def __init__(self, ax, canvas, on_view_change):
        self.ax = ax; self.cv = canvas; self._cb = on_view_change; self._p0 = None
        w = canvas.get_tk_widget()
        w.bind("<MouseWheel>",      self._wheel,  add="+")
        w.bind("<Button-4>",        self._sup,    add="+")
        w.bind("<Button-5>",        self._sdn,    add="+")
        w.bind("<Button-2>",        self._pstart, add="+")
        w.bind("<B2-Motion>",       self._pmove,  add="+")
        w.bind("<ButtonRelease-2>", self._pend,   add="+")

    def _in(self, tx, ty):
        h = self.cv.get_tk_widget().winfo_height(); bb = self.ax.get_window_extent()
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

    def _wheel(self, e): self._zoom(e.x, e.y, 1/ZOOM_FACTOR if e.delta>0 else ZOOM_FACTOR)
    def _sup(self, e):   self._zoom(e.x, e.y, 1/ZOOM_FACTOR)
    def _sdn(self, e):   self._zoom(e.x, e.y,   ZOOM_FACTOR)

    def _pstart(self, e):
        if self._in(e.x, e.y):
            self._p0 = (e.x, e.y, self.ax.get_xlim(), self.ax.get_ylim())

    def _pmove(self, e):
        if self._p0 is None: return
        x0, y0, xl, yl = self._p0
        bb = self.ax.get_window_extent()
        if bb.width < 1 or bb.height < 1: return
        dx = (e.x-x0)/bb.width  * (xl[1]-xl[0])
        dy = (e.y-y0)/bb.height * (yl[1]-yl[0])
        self.ax.set_xlim(xl[0]-dx, xl[1]-dx)
        self.ax.set_ylim(yl[0]+dy, yl[1]+dy)
        self._cb()

    def _pend(self, e): self._p0 = None


# ==============================================================================
#  mdoc parser
# ==============================================================================
def _coerce(val_str):
    parts = val_str.split()
    if not parts: return val_str
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


# ==============================================================================
#  CSV loader  -  auto-detect X, Y, label columns
# ==============================================================================
def load_csv(path):
    """Return (headers, rows, col_x, col_y, col_lbl).
    col_x / col_y  = first two numeric columns (or None).
    col_lbl        = first non-numeric column (or None).
    """
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        sample = fh.read(4096); fh.seek(0)
        try:    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        except: dialect = csv.excel
        reader  = csv.DictReader(fh, dialect=dialect)
        headers = list(reader.fieldnames or [])
        for row in reader:
            rows.append({k: v.strip() for k, v in row.items()})

    numeric_cols = []
    text_cols    = []
    for h in headers:
        vals = [r[h] for r in rows if r.get(h, "").strip()]
        if not vals: continue
        try:    [float(v) for v in vals]; numeric_cols.append(h)
        except: text_cols.append(h)

    col_x   = numeric_cols[0] if len(numeric_cols) >= 1 else None
    col_y   = numeric_cols[1] if len(numeric_cols) >= 2 else None
    col_lbl = text_cols[0]    if text_cols else None

    return headers, rows, col_x, col_y, col_lbl


# ==============================================================================
#  MRC montage assembler
# ==============================================================================
def _cosine_weight(h, w, feather):
    feather = max(1, int(feather))
    def ramp(n):
        r = np.ones(n, np.float32)
        f = min(feather, n // 2)
        if f > 0:
            t = np.linspace(0, math.pi/2, f, dtype=np.float32)
            r[:f] = np.sin(t); r[-f:] = np.sin(t)[::-1]
        return r
    return np.outer(ramp(h), ramp(w))


def assemble_montage(mrc_path, pieces, img_h, img_w, feather_px,
                     max_disp=MAX_DISP_PX, status_cb=None):
    """Assemble the montage for DISPLAY at reduced resolution, but report the
    FULL-resolution canvas size so all coordinate maths stays in full-res px.

    Returns (display_array, (full_h, full_w), ds) where ds is the integer
    downsample factor applied to the displayed array.  Tiles are read and
    downsampled one at a time (memory-mapped), so peak memory is ~one tile
    rather than the whole stack or a full-resolution canvas.
    """
    max_x = max_y = 0
    for p in pieces:
        c = p.get("PieceCoordinates", [0, 0, 0])
        if isinstance(c, (list, tuple)):
            max_x = max(max_x, int(c[0]) + img_w)
            max_y = max(max_y, int(c[1]) + img_h)
    if max_x == 0: max_x = img_w
    if max_y == 0: max_y = img_h
    full_w, full_h = int(max_x), int(max_y)

    ds = max(1, int(math.ceil(max(full_w, full_h) / float(max_disp))))
    cw = -(-full_w // ds)   # ceil division -> display canvas width
    ch = -(-full_h // ds)   # display canvas height

    canvas  = np.zeros((ch, cw), np.float64)
    weights = np.zeros((ch, cw), np.float64)

    n = len(pieces)
    with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
        n_frames = mrc.data.shape[0] if mrc.data.ndim == 3 else 1
        for i, p in enumerate(pieces):
            z = p.get("ZValue", i)
            if z >= n_frames: continue
            tile = (mrc.data[z] if mrc.data.ndim == 3
                    else mrc.data).astype(np.float64)
            if ds > 1:
                tile = tile[::ds, ::ds]
            dh, dw = tile.shape
            wmap = _cosine_weight(dh, dw, max(1, feather_px // ds)).astype(np.float64)

            c  = p.get("PieceCoordinates", [0, 0, 0])
            cx = int(c[0]) if isinstance(c, (list, tuple)) else 0
            cy = int(c[1]) if isinstance(c, (list, tuple)) else 0
            ox = int(round(cx / ds)); oy = int(round(cy / ds))

            sy0 = max(0, -oy);     sx0 = max(0, -ox)
            sy1 = min(dh, ch - oy)
            sx1 = min(dw, cw - ox)
            if sy1 <= sy0 or sx1 <= sx0: continue
            dy0 = oy+sy0; dx0 = ox+sx0
            dy1 = dy0+(sy1-sy0); dx1 = dx0+(sx1-sx0)
            tc = tile[sy0:sy1, sx0:sx1]
            wc = wmap[sy0:sy1, sx0:sx1]
            canvas [dy0:dy1, dx0:dx1] += tc * wc
            weights[dy0:dy1, dx0:dx1] += wc

            if status_cb and i % max(1, n//10) == 0:
                status_cb(f"Assembling tile {i+1}/{n} (1/{ds} res) ...")

    valid = weights > 0
    canvas[valid] /= weights[valid]
    arr = canvas.astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    else:
        arr[:] = 0
    return arr, (full_h, full_w), ds


def load_mrc_image(mrc_path, mdoc_global=None, mdoc_pieces=None, status_cb=None):
    """Return (display_image, info_str, (full_h, full_w)).

    The display image may be downsampled for very large data, but full_h/full_w
    are always the FULL-resolution canvas size so the caller can map clicks back
    to full-resolution pixel coordinates.  On error returns (None, msg, None).
    """
    if not _HAVE_MRCFILE:
        return None, "mrcfile not installed", None
    try:
        with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
            shape  = mrc.data.shape
            vox    = mrc.voxel_size
            px_ang = float(getattr(vox, 'x', 10.0))
    except Exception as e:
        return None, str(e), None

    px_nm = px_ang / 10.0

    def _prep_single(arr2d):
        """Normalise + downsample a single 2-D frame; return (disp, (fh, fw), ds)."""
        fh, fw = arr2d.shape
        ds = max(1, int(math.ceil(max(fh, fw) / float(MAX_DISP_PX))))
        data = (arr2d[::ds, ::ds] if ds > 1 else arr2d).astype(np.float32)
        lo, hi = np.percentile(data, 1), np.percentile(data, 99)
        if hi > lo: data = np.clip((data-lo)/(hi-lo), 0, 1)
        else:       data[:] = 0
        return data, (fh, fw), ds

    if len(shape) == 2 or (len(shape) == 3 and shape[0] == 1):
        with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
            raw = (mrc.data[0] if mrc.data.ndim == 3 else mrc.data)
            disp, (fh, fw), ds = _prep_single(raw)
        extra = f"  (display 1/{ds})" if ds > 1 else ""
        return disp, (f"single frame {fw}x{fh}{extra}, px={px_nm:.3f} nm"), (fh, fw)

    if mdoc_pieces and len(mdoc_pieces) > 0:
        img_size = (mdoc_global or {}).get("ImageSize", [shape[-1], shape[-2]])
        if isinstance(img_size, (int, float)):
            img_w = img_h = int(img_size)
        else:
            img_w, img_h = int(img_size[0]), int(img_size[1])

        ps = (mdoc_global or {}).get("PieceSpacing",
                                     [img_w - img_w//10, img_h - img_h//10])
        if isinstance(ps, (int, float)): ps_x = ps_y = int(ps)
        else:                            ps_x, ps_y = int(ps[0]), int(ps[1])
        feather = max(4, min(img_w - ps_x, img_h - ps_y, img_w//8))

        try:
            img, (fh, fw), ds = assemble_montage(mrc_path, mdoc_pieces,
                                                 img_h, img_w, feather,
                                                 status_cb=status_cb)
            H, W = img.shape
            extra = f"  (display 1/{ds} = {W}x{H})" if ds > 1 else ""
            return img, (f"montage {fw}x{fh}{extra} from {len(mdoc_pieces)} "
                         f"tiles, px={px_nm:.3f} nm"), (fh, fw)
        except Exception as e:
            print(f"Montage assembly failed: {e} - showing middle slice",
                  file=sys.stderr)

    mid = shape[0] // 2
    with mrcfile.mmap(mrc_path, mode="r", permissive=True) as mrc:
        disp, (fh, fw), ds = _prep_single(mrc.data[mid])
    extra = f"  (display 1/{ds})" if ds > 1 else ""
    return disp, (f"middle slice {mid}/{shape[0]} {fw}x{fh}{extra}, "
                  f"px={px_nm:.3f} nm"), (fh, fw)


# ==============================================================================
#  Main Application
# ==============================================================================
class App(tk.Tk):
    IDLE = 0; PICK_CENTRE = 1; PICK_XDIR = 2; PICK_YDIR = 3

    def __init__(self):
        super().__init__()
        self.title("CSV . mdoc  Coordinate Converter")
        self.configure(bg=BG)
        self.minsize(1100, 760)

        # data
        self._csv_rows     = []
        self._csv_path     = ""
        self._col_x        = None
        self._col_y        = None
        self._col_lbl      = None

        self._mdoc_global  = {}
        self._mdoc_pieces  = []
        self._mdoc_path    = ""
        self._px_nm        = 1.0
        self._stage_fit    = None          # pixel->stage similarity fit (or None)

        self._mrc_image    = None
        self._mrc_path     = ""

        # calibration
        self._step         = self.IDLE
        self._centre_img   = None
        self._xdir_img     = None
        self._ydir_img     = None          # optional +Y side pick (for indicator)
        self._ydir_sign    = 1             # +1 = +Y is 90 deg CCW from +X, -1 = CW
        self._angle_deg    = 0.0           # +X axis angle, CCW from image +X
        self._angle_y_deg  = 90.0          # +Y axis angle = +X +/- 90 (always perp)
        self._centre_stage = (0.0, 0.0)

        # results
        self._results      = []
        self._sel_idx      = None

        # display options
        self._show_labels_var = tk.BooleanVar(value=True)
        self._label_size_var  = tk.IntVar(value=8)

        # flip predicted coordinates (image/stage horizontal & vertical)
        self._flip_h_var      = tk.BooleanVar(value=False)
        self._flip_v_var      = tk.BooleanVar(value=False)

        # view
        self._img_h = 512; self._img_w = 512
        self._xlim  = (-0.5, 511.5)
        self._ylim  = (511.5, -0.5)

        self._csv_img_pts  = []

        self._build_styles()
        self._build_ui()

    # -- styles ----------------------------------------------------------------
    def _build_styles(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure("TFrame",            background=BG)
        s.configure("TLabel",            background=BG, foreground=FG,
                    font=("Segoe UI", 10))
        s.configure("Sm.TLabel",         background=BG, foreground=FG,
                    font=("Segoe UI", 9))
        s.configure("TButton",           background=ACC, foreground=BG,
                    font=("Segoe UI", 10, "bold"), padding=5)
        s.map("TButton", background=[("active", CYA), ("disabled", BG3)])
        s.configure("Accent.TButton",    background=ACC2, foreground=BG,
                    font=("Segoe UI", 11, "bold"), padding=7)
        s.configure("Danger.TButton",    background=RED, foreground=BG,
                    font=("Segoe UI", 10, "bold"), padding=5)
        s.configure("Step.TButton",      background=MAG, foreground=BG,
                    font=("Segoe UI", 10, "bold"), padding=6)
        s.configure("Active.TButton",    background=YEL, foreground=BG,
                    font=("Segoe UI", 10, "bold"), padding=6)
        s.configure("Mont.TButton",      background=CYA, foreground=BG,
                    font=("Segoe UI", 10, "bold"), padding=5)
        s.map("Step.TButton",   background=[("active", CYA), ("disabled", BG3)])
        s.map("Active.TButton", background=[("active", CYA), ("disabled", BG3)])
        s.map("Mont.TButton",   background=[("active", ACC2), ("disabled", BG3)])
        s.configure("TLabelframe",       background=BG, relief="groove")
        s.configure("TLabelframe.Label", background=BG, foreground=CYA,
                    font=("Segoe UI", 10, "bold"))
        s.configure("Treeview",          background=BG2, foreground=FG,
                    fieldbackground=BG2, rowheight=22)
        s.configure("Treeview.Heading",  background=BG3, foreground=FG,
                    font=("Segoe UI", 9, "bold"))
        s.map("Treeview", background=[("selected", ACC)])
        s.configure("TScale",       background=BG, troughcolor=BG3, sliderlength=14)
        s.configure("TSeparator",   background=BG3)
        s.configure("TCheckbutton", background=BG, foreground=FG,
                    font=("Segoe UI", 9))

    # -- UI --------------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self, padding=(8, 6, 8, 0)); top.pack(fill="x")

        ttk.Button(top, text="Load CSV",
                   command=self._load_csv).pack(side="left", padx=3)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=4)
        ttk.Button(top, text="Load MRC + mdoc",
                   style="Mont.TButton",
                   command=self._load_mrc_and_mdoc).pack(side="left", padx=3)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=4)
        ttk.Button(top, text="Load mdoc",
                   command=self._load_mdoc).pack(side="left", padx=3)
        ttk.Button(top, text="Load MRC",
                   command=self._load_mrc).pack(side="left", padx=3)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

        self._info_var = tk.StringVar(value="No files loaded")
        ttk.Label(top, textvariable=self._info_var,
                  style="Sm.TLabel", foreground=BG3).pack(side="left", padx=4)
        ttk.Label(top, text="scroll=zoom  mid-drag=pan  left=pick  right=undo",
                  style="Sm.TLabel", foreground=BG3).pack(side="right", padx=6)

        body = ttk.Frame(self, padding=(8, 6)); body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1); body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        # canvas
        img_lf = ttk.LabelFrame(body,
            text="Image / Stage Canvas  -  assembled MRC montage or blank grid",
            padding=4)
        img_lf.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        img_lf.rowconfigure(0, weight=1); img_lf.columnconfigure(0, weight=1)

        self._fig = Figure(figsize=(7, 6), facecolor=BG)
        self._ax  = self._fig.add_subplot(111)
        self._fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._ax.set_facecolor(BG); self._ax.axis("off")
        self._canvas = FigureCanvasTkAgg(self._fig, master=img_lf)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._canvas.draw()
        self._pz = PanZoom(self._ax, self._canvas, self._on_view_change)
        self._canvas.mpl_connect("button_press_event",  self._on_click)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)

        # B/C + label + flip controls
        bc_lf = ttk.LabelFrame(body, text="Display", padding=(8, 4))
        bc_lf.grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=(4, 0))
        bc_lf.columnconfigure(1, weight=1)

        # brightness/contrast
        ttk.Label(bc_lf, text="Black pt", style="Sm.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0,6))
        self._bc_lo_var = tk.DoubleVar(value=0.0)
        ttk.Scale(bc_lf, from_=0.0, to=0.99, orient="horizontal",
                  variable=self._bc_lo_var,
                  command=self._on_bc_change).grid(row=0, column=1, sticky="ew")
        self._bc_lo_lbl = ttk.Label(bc_lf, text="0.00", style="Sm.TLabel",
                                     foreground=CYA, width=5)
        self._bc_lo_lbl.grid(row=0, column=2)
        ttk.Label(bc_lf, text="White pt", style="Sm.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0,6))
        self._bc_hi_var = tk.DoubleVar(value=1.0)
        ttk.Scale(bc_lf, from_=0.01, to=1.0, orient="horizontal",
                  variable=self._bc_hi_var,
                  command=self._on_bc_change).grid(row=1, column=1, sticky="ew")
        self._bc_hi_lbl = ttk.Label(bc_lf, text="1.00", style="Sm.TLabel",
                                     foreground=CYA, width=5)
        self._bc_hi_lbl.grid(row=1, column=2)
        ttk.Button(bc_lf, text="Auto", command=self._bc_auto).grid(
            row=0, column=3, rowspan=2, padx=(8, 0))

        # separator
        ttk.Separator(bc_lf, orient="vertical").grid(
            row=0, column=4, rowspan=2, sticky="ns", padx=(14, 10))

        # POI label toggle
        ttk.Checkbutton(bc_lf, text="Show POI names",
                        variable=self._show_labels_var,
                        command=self._redraw).grid(row=0, column=5, sticky="w")

        # font size +/- buttons
        fs_frame = ttk.Frame(bc_lf)
        fs_frame.grid(row=1, column=5, sticky="w")
        ttk.Label(fs_frame, text="Label size:", style="Sm.TLabel").pack(side="left")
        ttk.Button(fs_frame, text="-", width=2,
                   command=lambda: self._adj_label_size(-1)).pack(side="left", padx=(4,0))
        self._label_size_lbl = ttk.Label(fs_frame, text=" 8 pt",
                                          style="Sm.TLabel", foreground=CYA, width=5)
        self._label_size_lbl.pack(side="left", padx=2)
        ttk.Button(fs_frame, text="+", width=2,
                   command=lambda: self._adj_label_size(+1)).pack(side="left")

        # separator
        ttk.Separator(bc_lf, orient="vertical").grid(
            row=0, column=6, rowspan=2, sticky="ns", padx=(14, 10))

        # flip predicted coordinates (image/stage space)
        flip_frame = ttk.Frame(bc_lf)
        flip_frame.grid(row=0, column=7, rowspan=2, sticky="w")
        ttk.Label(flip_frame, text="Flip predicted STAGE coords:",
                  style="Sm.TLabel", foreground=CYA).pack(anchor="w")
        ttk.Checkbutton(flip_frame, text="Flip stage X  (horizontal)",
                        variable=self._flip_h_var,
                        command=self._on_flip_change).pack(anchor="w")
        ttk.Checkbutton(flip_frame, text="Flip stage Y  (vertical)",
                        variable=self._flip_v_var,
                        command=self._on_flip_change).pack(anchor="w")

        right = ttk.Frame(body, width=310)
        right.grid(row=0, column=1, sticky="nsew", rowspan=2)
        right.columnconfigure(0, weight=1)
        self._build_right(right)

        sf = ttk.Frame(self, padding=(8, 2, 8, 4)); sf.pack(fill="x")
        ttk.Separator(sf, orient="horizontal").pack(fill="x", pady=(0, 3))
        self._sv = tk.StringVar(value="Load a CSV and MRC+mdoc to begin.")
        ttk.Label(sf, textvariable=self._sv,
                  style="Sm.TLabel", foreground=CYA).pack(side="left")

        self._redraw()

    def _build_right(self, p):
        r = 0
        # info
        lf0 = ttk.LabelFrame(p, text="Loaded Files", padding=(8, 4))
        lf0.grid(row=r, column=0, sticky="ew", pady=(0, 6)); r += 1
        self._file_info_var = tk.StringVar(value="No files loaded")
        ttk.Label(lf0, textvariable=self._file_info_var,
                  style="Sm.TLabel", foreground=CYA,
                  justify="left", wraplength=285).pack(anchor="w")

        # step 1
        lf1 = ttk.LabelFrame(p, text="Step 1  .  Image Centre", padding=(8, 4))
        lf1.grid(row=r, column=0, sticky="ew", pady=(0, 6)); r += 1
        ttk.Label(lf1,
                  text="Left-click the point on the image that corresponds\n"
                       "to the origin of your CSV coordinate system.",
                  style="Sm.TLabel", justify="left").pack(anchor="w")
        self._btn_cen = ttk.Button(lf1, text="Pick Centre",
                                    style="Step.TButton", command=self._act_centre)
        self._btn_cen.pack(fill="x", pady=(6, 2))
        self._cen_var = tk.StringVar(value="Centre:  not set")
        ttk.Label(lf1, textvariable=self._cen_var,
                  style="Sm.TLabel", foreground=CYA).pack(anchor="w")

        # step 2
        lf2 = ttk.LabelFrame(p, text="Step 2  .  +X Direction", padding=(8, 4))
        lf2.grid(row=r, column=0, sticky="ew", pady=(0, 6)); r += 1
        ttk.Label(lf2,
                  text="Click any point that lies in the +X direction\n"
                       "of your CSV coordinate system.",
                  style="Sm.TLabel", justify="left").pack(anchor="w")
        self._btn_xax = ttk.Button(lf2, text="Pick +X Direction",
                                    style="Step.TButton", command=self._act_xdir)
        self._btn_xax.pack(fill="x", pady=(6, 2))
        self._xax_var = tk.StringVar(value="X-axis:  not set")
        self._ang_var = tk.StringVar(value="Angle:   -")
        ttk.Label(lf2, textvariable=self._xax_var,
                  style="Sm.TLabel", foreground=CYA).pack(anchor="w")
        ttk.Label(lf2, textvariable=self._ang_var,
                  style="Sm.TLabel", foreground=YEL).pack(anchor="w")

        # step 2b  .  +Y Direction (optional)
        lf2b = ttk.LabelFrame(p, text="Step 2b  .  +Y Direction  (optional)",
                              padding=(8, 4))
        lf2b.grid(row=r, column=0, sticky="ew", pady=(0, 6)); r += 1
        ttk.Label(lf2b,
                  text="Click a point to choose which side +Y points to.\n"
                       "+Y always stays perpendicular to +X; the click only\n"
                       "selects the CCW (+90) or CW (-90) direction.",
                  style="Sm.TLabel", justify="left").pack(anchor="w")
        self._btn_yax = ttk.Button(lf2b, text="Pick +Y Side",
                                    style="Step.TButton", command=self._act_ydir)
        self._btn_yax.pack(fill="x", pady=(6, 2))
        ttk.Button(lf2b, text="Reset +Y to perpendicular",
                   command=self._clear_ydir).pack(fill="x", pady=(0, 2))
        self._yax_var = tk.StringVar(value="Y-axis:  perpendicular (default)")
        self._angy_var = tk.StringVar(value="Y angle: +90 from +X")
        ttk.Label(lf2b, textvariable=self._yax_var,
                  style="Sm.TLabel", foreground=CYA).pack(anchor="w")
        ttk.Label(lf2b, textvariable=self._angy_var,
                  style="Sm.TLabel", foreground=YEL).pack(anchor="w")

        # step 3
        lf3 = ttk.LabelFrame(p, text="Step 3  .  Convert & Export", padding=(8, 4))
        lf3.grid(row=r, column=0, sticky="ew", pady=(0, 6)); r += 1
        self._btn_conv = ttk.Button(lf3, text="Convert Coordinates",
                                     style="Accent.TButton",
                                     command=self._convert, state="disabled")
        self._btn_conv.pack(fill="x", pady=(0, 4))
        self._btn_exp = ttk.Button(lf3, text="Export CSV",
                                    command=self._export, state="disabled")
        self._btn_exp.pack(fill="x", pady=(0, 4))
        # PNG export buttons
        self._btn_png = ttk.Button(lf3, text="Export PNG  (current view)",
                                    style="Mont.TButton",
                                    command=self._export_png, state="disabled")
        self._btn_png.pack(fill="x", pady=(0, 4))
        self._btn_png_full = ttk.Button(lf3, text="Export PNG  (full image)",
                                         style="Mont.TButton",
                                         command=self._export_png_full, state="disabled")
        self._btn_png_full.pack(fill="x", pady=(0, 4))
        ttk.Button(lf3, text="Reset Calibration",
                   style="Danger.TButton", command=self._reset).pack(fill="x")

        # results table
        lf4 = ttk.LabelFrame(p, text="Converted Coordinates  (um)", padding=(4, 4))
        lf4.grid(row=r, column=0, sticky="nsew"); r += 1
        p.rowconfigure(r-1, weight=1)
        lf4.rowconfigure(0, weight=1); lf4.columnconfigure(0, weight=1)
        cols = ("Label", "CSV X", "CSV Y", "Stage X", "Stage Y")
        self._tree = ttk.Treeview(lf4, columns=cols, show="headings")
        for c, w in zip(cols, [80, 75, 75, 90, 90]):
            self._tree.heading(c, text=c); self._tree.column(c, width=w, anchor="center")
        vsb = ttk.Scrollbar(lf4, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self._tree.bind("<<TreeviewSelect>>", self._on_sel)

    # -- label size control ----------------------------------------------------
    def _adj_label_size(self, delta):
        new_sz = max(5, min(24, self._label_size_var.get() + delta))
        self._label_size_var.set(new_sz)
        self._label_size_lbl.configure(text=f"{new_sz:2d} pt")
        self._redraw()

    # -- flip control ----------------------------------------------------------
    def _on_flip_change(self):
        """Flips mirror the PREDICTED points about the origin label (per axis),
        moving both the on-image markers and their stage coordinates.  The CSV
        frame and its drawn axes are untouched, so re-project the preview and
        re-run any conversion."""
        if self._centre_img and self._xdir_img:
            self._project_csv_to_image()
        if self._results:
            self._convert()
        else:
            self._redraw()
        fh = self._flip_h_var.get(); fv = self._flip_v_var.get()
        state = (("X " if fh else "") + ("Y " if fv else "")).strip()
        self._sv.set(f"Stage flip: {state}" if state else "Stage flip: off")

    # -- view ------------------------------------------------------------------
    def _on_view_change(self):
        self._xlim = self._ax.get_xlim()
        self._ylim = self._ax.get_ylim()
        self._redraw()

    # -- axis basis ------------------------------------------------------------
    def _axis_basis(self):
        """Return (ux, uy, vx, vy): the unit image-space (screen, y-down)
        direction vectors for CSV +X and CSV +Y.

        This describes the CSV file's OWN coordinate frame as registered on the
        image (origin + picked +X direction + perpendicular +Y side).  It is
        used both for the CSV->image projection and for drawing the +X / +Y
        arrows.  It is deliberately INDEPENDENT of the stage-coordinate flip
        toggles - flipping a grid between modalities must not move the CSV
        frame, only the predicted stage output (see _predict_stage).

        A CSV displacement (cx_nm, cy_nm) maps to an image-pixel displacement
            dx_px = (cx_nm*ux + cy_nm*vx) / px_nm
            dy_px = (cx_nm*uy + cy_nm*vy) / px_nm
        """
        tx = math.radians(self._angle_deg)
        ux =  math.cos(tx); uy = -math.sin(tx)
        ty = math.radians(self._angle_y_deg)
        vx =  math.cos(ty); vy = -math.sin(ty)
        return ux, uy, vx, vy

    # -- draw ------------------------------------------------------------------
    def _draw_markers(self, ax, xlim=None, ylim=None):
        """Draw all result markers + labels onto ax.
        If xlim/ylim are provided (current view), use arm scaled to view width;
        otherwise scale to image width (for full export).
        """
        show_lbl = self._show_labels_var.get()
        lbl_sz   = self._label_size_var.get()
        view_w   = (xlim[1]-xlim[0]) if xlim else self._img_w

        # CSV preview dots (before conversion)
        if self._csv_img_pts and not self._results:
            pts = np.array(self._csv_img_pts)
            ax.scatter(pts[:,0], pts[:,1], s=30, color=ORG,
                       edgecolors=BG, linewidths=0.6, zorder=4, alpha=0.8)

        # result markers + POI name labels
        for i, res in enumerate(self._results):
            ix, iy = res["img_x"], res["img_y"]
            label  = res.get("label", "")
            selected = (i == self._sel_idx)
            color = ACC2 if selected else ORG
            ms    = 10   if selected else 6
            mew   = 1.5  if selected else 0.6
            mec   = "white" if selected else BG
            ax.plot(ix, iy, "o", color=color, ms=ms, mec=mec, mew=mew, zorder=5+(selected*3))
            if show_lbl and label:
                ax.text(ix + 6, iy - 6, label,
                        color=color, fontsize=lbl_sz,
                        fontweight="bold" if selected else "normal",
                        zorder=6+(selected*3),
                        path_effects=[pe.Stroke(linewidth=1.5+(selected*0.5),
                                                foreground=BG), pe.Normal()])

        # origin cross
        if self._centre_img:
            cx, cy = self._centre_img
            arm = max(12, view_w * 0.025)
            kw = dict(color=ACC2, lw=2.2, zorder=7,
                      path_effects=[pe.Stroke(linewidth=4, foreground=BG), pe.Normal()])
            ax.plot([cx-arm, cx+arm], [cy, cy], **kw)
            ax.plot([cx, cx], [cy-arm, cy+arm], **kw)
            ax.plot(cx, cy, "o", color=ACC2, ms=9, mec=BG, mew=1.2, zorder=8)
            ax.text(cx+arm*0.7, cy-arm*0.7, "ORIGIN", color=ACC2,
                    fontsize=8, fontweight="bold", zorder=9,
                    path_effects=[pe.Stroke(linewidth=2, foreground=BG), pe.Normal()])

        # +X click indicator
        if self._xdir_img:
            px, py = self._xdir_img
            ax.plot(px, py, "D", color=CYA, ms=8, mec=BG, mew=1.2, zorder=7)
            ax.text(px+8, py-8, "+X click", color=CYA,
                    fontsize=8, fontweight="bold", zorder=8,
                    path_effects=[pe.Stroke(linewidth=2, foreground=BG), pe.Normal()])

        # +Y click indicator (only if user explicitly picked it)
        if self._ydir_img:
            qx, qy = self._ydir_img
            ax.plot(qx, qy, "D", color=YEL, ms=8, mec=BG, mew=1.2, zorder=7)
            ax.text(qx+8, qy-8, "+Y click", color=YEL,
                    fontsize=8, fontweight="bold", zorder=8,
                    path_effects=[pe.Stroke(linewidth=2, foreground=BG), pe.Normal()])

        # CSV-frame axis arrows (drawn from the same basis used for the maths,
        # so flips and the optional +Y pick are reflected here too)
        if self._centre_img and self._xdir_img:
            cx, cy = self._centre_img
            arm = view_w * 0.18
            ux, uy, vx, vy = self._axis_basis()
            for (dx, dy), col, lbl in [((ux, uy), CYA, "+X"), ((vx, vy), YEL, "+Y")]:
                ex = cx + arm*dx
                ey = cy + arm*dy
                ax.plot([cx, ex], [cy, ey], color=BG, lw=5,
                        solid_capstyle="round", zorder=9)
                ax.annotate("", xy=(ex, ey), xytext=(cx, cy), zorder=10,
                    arrowprops=dict(arrowstyle="->, head_width=0.35, head_length=0.45",
                                    color=col, lw=2.5, shrinkA=0, shrinkB=0))
                ax.text(ex+arm*0.07, ey-arm*0.07, lbl, color=col,
                        fontsize=11, fontweight="bold", zorder=11,
                        path_effects=[pe.Stroke(linewidth=3, foreground=BG), pe.Normal()])

    def _redraw(self):
        ax = self._ax; ax.cla(); ax.set_facecolor(BG); ax.axis("off")
        H, W = self._img_h, self._img_w

        if self._mrc_image is not None:
            ax.imshow(self._mrc_image, cmap="gray", origin="upper",
                      vmin=self._bc_lo_var.get(), vmax=self._bc_hi_var.get(),
                      aspect="equal", extent=[-0.5, W-0.5, H-0.5, -0.5], zorder=0)
        else:
            for xi in range(0, W+1, max(1, W//8)):
                ax.axvline(xi, color=BG3, lw=0.4, zorder=0)
            for yi in range(0, H+1, max(1, H//8)):
                ax.axhline(yi, color=BG3, lw=0.4, zorder=0)
            ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)

        ax.set_xlim(self._xlim); ax.set_ylim(self._ylim)
        self._draw_markers(ax, xlim=self._xlim, ylim=self._ylim)
        self._canvas.draw()

    # -- B/C -------------------------------------------------------------------
    def _on_bc_change(self, _=None):
        lo = self._bc_lo_var.get(); hi = self._bc_hi_var.get()
        if lo >= hi: lo = hi-0.01; self._bc_lo_var.set(round(lo, 3))
        self._bc_lo_lbl.configure(text=f"{lo:.2f}")
        self._bc_hi_lbl.configure(text=f"{hi:.2f}")
        self._redraw()

    def _bc_auto(self):
        self._bc_lo_var.set(0.0); self._bc_lo_lbl.configure(text="0.00")
        self._bc_hi_var.set(1.0); self._bc_hi_lbl.configure(text="1.00")
        self._redraw()

    # -- file loading ----------------------------------------------------------
    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Open CSV  (X and Y columns in um)",
            filetypes=[("CSV / TSV", "*.csv *.tsv *.txt"), ("All", "*.*")])
        if not path: return
        try:
            headers, rows, cx, cy, cl = load_csv(path)
        except Exception as e:
            messagebox.showerror("CSV error", str(e)); return
        if not headers:
            messagebox.showerror("CSV error", "No columns found."); return
        if cx is None or cy is None:
            messagebox.showerror("CSV error",
                "Could not find two numeric columns for X and Y.\n"
                "Make sure the CSV has at least two columns with numeric values."); return
        self._csv_rows = rows; self._csv_path = path
        self._col_x = cx; self._col_y = cy; self._col_lbl = cl
        self._update_file_info()
        if self._centre_img and self._xdir_img:
            self._project_csv_to_image()
        self._check_ready(); self._redraw()
        self._sv.set(f"CSV: {len(rows)} rows  |  X='{cx}'  Y='{cy}'"
                     + (f"  label='{cl}'" if cl else ""))

    def _apply_mdoc(self, path, g, pieces):
        self._mdoc_global = g; self._mdoc_pieces = pieces; self._mdoc_path = path
        ps_ang = g.get("PixelSpacing", 10.0)
        if isinstance(ps_ang, (list, tuple)): ps_ang = float(ps_ang[0])
        self._px_nm = float(ps_ang) / 10.0
        # Fit a no-shear similarity map (image pixel -> stage um) from the tiles.
        self._stage_fit = self._fit_pixel_to_stage()
        if self._stage_fit is not None:
            kind = "rotation+flip" if self._stage_fit["reflect"] else "rotation"
            print(f"[converter] image->stage: similarity fit ({kind}), "
                  f"{self._stage_fit['n']} tiles, "
                  f"rotation {self._stage_fit['angle_deg']:+.2f} deg, "
                  f"rmse {self._stage_fit['rmse']:.4f} um, "
                  f"scale {self._stage_fit['scale_nm_px']:.3f} nm/px")
        else:
            print("[converter] image->stage: per-tile interpolation "
                  "(no rotation fit - <2 tiles or no stage spread)")
        self._update_file_info(); self._check_ready()

    def _apply_mrc(self, mrc_path, status_cb=None):
        if not _HAVE_MRCFILE:
            messagebox.showerror("Missing package",
                "mrcfile is not installed.\n\npip install mrcfile"); return False
        img, info_str, full_hw = load_mrc_image(mrc_path, self._mdoc_global,
                                                self._mdoc_pieces, status_cb)
        if img is None:
            messagebox.showerror("MRC error", info_str); return False
        self._mrc_image = img                       # may be downsampled (display)
        self._mrc_path = mrc_path
        # Coordinate system is ALWAYS full resolution, so clicks, the picked
        # centre and all predictions stay correct even when the displayed image
        # is downsampled.  imshow stretches the small array across this extent.
        Hf, Wf = full_hw if full_hw is not None else img.shape
        self._img_h = int(Hf); self._img_w = int(Wf)
        self._xlim = (-0.5, self._img_w-0.5); self._ylim = (self._img_h-0.5, -0.5)
        self._bc_auto()
        self._update_file_info()
        self._sv.set(f"MRC loaded: {info_str}")
        return True

    def _load_mrc_and_mdoc(self):
        path = filedialog.askopenfilename(
            title="Open MRC or mdoc file  (sibling found automatically)",
            filetypes=[("MRC / mdoc", "*.mrc *.rec *.mrcs *.map *.mdoc"),
                       ("All", "*.*")])
        if not path: return

        stem, ext = os.path.splitext(path)
        ext = ext.lower()
        mrc_exts = {".mrc", ".rec", ".mrcs", ".map"}

        if ext in mrc_exts:
            mrc_path = path
            mdoc_path = None
            for suffix in (".mdoc", ".Mdoc", ".MDOC"):
                candidate = stem + suffix
                if os.path.isfile(candidate):
                    mdoc_path = candidate; break
            if mdoc_path is None:
                mdoc_path = filedialog.askopenfilename(
                    title=f"Locate mdoc for  '{os.path.basename(stem)}'",
                    initialdir=os.path.dirname(path),
                    filetypes=[("mdoc", "*.mdoc"), ("All", "*.*")])
                if not mdoc_path: return

        elif ext == ".mdoc":
            mdoc_path = path
            mrc_path = None
            for mext in (".mrc", ".rec", ".mrcs", ".map"):
                candidate = stem + mext
                if os.path.isfile(candidate):
                    mrc_path = candidate; break
            if mrc_path is None:
                mrc_path = filedialog.askopenfilename(
                    title=f"Locate MRC for  '{os.path.basename(stem)}'",
                    initialdir=os.path.dirname(path),
                    filetypes=[("MRC", "*.mrc *.rec *.mrcs *.map"), ("All", "*.*")])
                if not mrc_path: return
        else:
            messagebox.showerror("Unknown file type",
                f"Extension '{ext}' not recognised.\n"
                "Please select a .mrc, .rec, .mrcs, .map, or .mdoc file.")
            return

        try:
            g, pieces, _ = parse_mdoc(mdoc_path)
        except Exception as e:
            messagebox.showerror("mdoc error", str(e)); return
        self._apply_mdoc(mdoc_path, g, pieces)

        def status_cb(msg):
            self._sv.set(msg); self.update_idletasks()

        ok = self._apply_mrc(mrc_path, status_cb=status_cb)
        if ok:
            n = len(pieces)
            self._sv.set(
                f"Loaded  {os.path.basename(mrc_path)}  +  "
                f"{os.path.basename(mdoc_path)}  "
                f"({n} tiles, {self._px_nm:.4f} nm/px)")

    def _load_mdoc(self):
        path = filedialog.askopenfilename(
            title="Open mdoc", filetypes=[("mdoc", "*.mdoc"), ("All", "*.*")])
        if not path: return
        try:
            g, pieces, _ = parse_mdoc(path)
        except Exception as e:
            messagebox.showerror("mdoc error", str(e)); return
        self._apply_mdoc(path, g, pieces)
        self._sv.set(f"mdoc: {len(pieces)} tiles, {self._px_nm:.4f} nm/px")

    def _load_mrc(self):
        path = filedialog.askopenfilename(
            title="Open MRC image",
            filetypes=[("MRC", "*.mrc *.rec *.mrcs *.map"), ("All", "*.*")])
        if not path: return
        def status_cb(msg):
            self._sv.set(msg); self.update_idletasks()
        self._apply_mrc(path, status_cb=status_cb)

    # -- coordinate helpers ----------------------------------------------------
    def _apply_flip_img(self, ix, iy):
        """Mirror an image point about the origin label, per the Flip stage X /
        Flip stage Y toggles.

        Mirroring the predicted point about the label in image space is the
        physical 'grid was flipped between modalities' correction.  Doing it
        here - on the predicted point - means the on-image marker visibly moves
        AND its stage coordinate changes consistently, while the CSV frame and
        its drawn axes (which never call this) stay put.
        """
        if self._centre_img is None:
            return ix, iy
        cx, cy = self._centre_img
        if self._flip_h_var.get(): ix = 2.0*cx - ix
        if self._flip_v_var.get(): iy = 2.0*cy - iy
        return ix, iy

    def _project_csv_to_image(self):
        self._csv_img_pts = []
        if not self._col_x or not self._col_y: return
        for row in self._csv_rows:
            try:
                rx_nm = float(row[self._col_x]) * 1000.0
                ry_nm = float(row[self._col_y]) * 1000.0
            except (ValueError, KeyError): continue
            ix, iy = self._csv_nm_to_img(rx_nm, ry_nm)
            self._csv_img_pts.append(self._apply_flip_img(ix, iy))

    def _csv_nm_to_img(self, csv_x_nm, csv_y_nm):
        if self._centre_img is None or self._px_nm <= 0:
            return (self._img_w/2, self._img_h/2)
        cx_img, cy_img = self._centre_img
        ux, uy, vx, vy = self._axis_basis()
        dx_nm = csv_x_nm*ux + csv_y_nm*vx
        dy_nm = csv_x_nm*uy + csv_y_nm*vy
        ix = cx_img + dx_nm / self._px_nm
        iy = cy_img + dy_nm / self._px_nm
        return (ix, iy)

    def _img_to_csv_um(self, ix, iy):
        """Inverse of _csv_nm_to_img -> CSV (x, y) in um (used for live readout)."""
        if self._centre_img is None or self._px_nm <= 0:
            return (0.0, 0.0)
        cx_img, cy_img = self._centre_img
        ux, uy, vx, vy = self._axis_basis()
        det = ux*vy - vx*uy
        if abs(det) < 1e-12:
            return (0.0, 0.0)
        dx_nm = (ix - cx_img) * self._px_nm
        dy_nm = (iy - cy_img) * self._px_nm
        csv_x_nm = ( vy*dx_nm - vx*dy_nm) / det
        csv_y_nm = (-uy*dx_nm + ux*dy_nm) / det
        return (csv_x_nm / 1000.0, csv_y_nm / 1000.0)

    def _fit_pixel_to_stage(self):
        """Fit a no-shear similarity (rotation + uniform scale + optional flip
        + translation) mapping montage-pixel coords -> stage microns, using
        each tile's pixel centre (PieceCoordinates + tile/2) and StagePosition.

        This recovers the true camera-to-stage rotation/flip/scale straight
        from the mdoc, so it is NOT limited to the 'image-right=+X, image-down=
        -Y, no rotation' assumption of the per-tile fallback.

        Returns a dict describing the transform, or None when it cannot/should
        not be fit (fewer than 2 tiles, or stage positions too clustered, e.g.
        a single-tile or pure image-shift montage) - the caller then falls back
        to per-tile interpolation.
        """
        pieces = self._mdoc_pieces
        if not pieces:
            return None

        img_size = self._mdoc_global.get("ImageSize", [self._img_w, self._img_h])
        if isinstance(img_size, (int, float)):
            tw = th = int(img_size)
        else:
            tw, th = int(img_size[0]), int(img_size[1])

        pts_px, pts_st = [], []
        for p in pieces:
            sp = p.get("StagePosition", None)
            if not isinstance(sp, (list, tuple)) or len(sp) < 2:
                continue
            c = p.get("PieceCoordinates", [0, 0, 0])
            if not isinstance(c, (list, tuple)) or len(c) < 2:
                continue
            pts_px.append((float(c[0]) + tw / 2.0, float(c[1]) + th / 2.0))
            pts_st.append((float(sp[0]), float(sp[1])))

        if len(pts_px) < 2:
            return None

        P = np.asarray(pts_px, dtype=np.float64)
        S = np.asarray(pts_st, dtype=np.float64)
        # Degenerate if stage barely varies (single tile / image-shift montage).
        if float(S.std(axis=0).max()) < 1e-6:
            return None

        n = len(P)
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
            scale_um_px = math.hypot(float(sol[0]), float(sol[1]))
            angle_deg = math.degrees(math.atan2(float(sol[1]), float(sol[0])))
            cand = {"reflect": reflect, "a": float(sol[0]), "b": float(sol[1]),
                    "tx": float(sol[2]), "ty": float(sol[3]),
                    "rmse": rmse, "n": n,
                    "angle_deg": angle_deg,
                    "scale_nm_px": scale_um_px * 1000.0}
            if best is None or rmse < best["rmse"]:
                best = cand
        return best

    def _apply_stage_fit(self, ix, iy):
        f = self._stage_fit
        a, b, tx, ty = f["a"], f["b"], f["tx"], f["ty"]
        if not f["reflect"]:
            return a * ix - b * iy + tx, b * ix + a * iy + ty
        return a * ix + b * iy + tx, b * ix - a * iy + ty

    def _img_to_stage_um(self, ix, iy):
        # Preferred: similarity fit (handles camera-to-stage rotation + flip).
        if self._stage_fit is not None:
            return self._apply_stage_fit(ix, iy)

        # Fallback A: no mdoc at all -> simple centred scaling.
        if not self._mdoc_pieces:
            return ((ix - self._img_w/2)*self._px_nm/1000.0,
                   -(iy - self._img_h/2)*self._px_nm/1000.0)

        # Fallback B: per-tile inverse-distance interpolation (no rotation).
        def tile_origin(p):
            c = p.get("PieceCoordinates", [0,0,0])
            return (float(c[0]), float(c[1])) if isinstance(c,(list,tuple)) else (0.0,0.0)

        img_size = self._mdoc_global.get("ImageSize", [self._img_w, self._img_h])
        if isinstance(img_size,(int,float)): tw = th = int(img_size)
        else:                                tw,th = int(img_size[0]),int(img_size[1])

        containing = [p for p in self._mdoc_pieces
                      if (tile_origin(p)[0] <= ix < tile_origin(p)[0]+tw and
                          tile_origin(p)[1] <= iy < tile_origin(p)[1]+th)]
        candidates = containing if containing else self._mdoc_pieces

        w_sum = sx = sy = 0.0
        for p in candidates:
            tx,ty  = tile_origin(p)
            tc_x = tx+tw/2.0; tc_y = ty+th/2.0
            dist = max(0.01, math.sqrt((ix-tc_x)**2+(iy-tc_y)**2))
            w    = 1.0/dist
            sp   = p.get("StagePosition",[0.0,0.0])
            if not isinstance(sp,(list,tuple)) or len(sp)<2: sp=[0.0,0.0]
            sx_um = float(sp[0]) + (ix-tc_x)*self._px_nm/1000.0
            sy_um = float(sp[1]) - (iy-tc_y)*self._px_nm/1000.0
            w_sum += w; sx += w*sx_um; sy += w*sy_um
        return sx/w_sum, sy/w_sum

    def _predict_stage(self, ix, iy):
        """Predicted ACTUAL stage position (um) for a CSV-projected image point.

        The point is first mirrored about the origin label per the Flip stage X /
        Flip stage Y toggles (see _apply_flip_img), then read into the stage
        frame via _img_to_stage_um (similarity fit when available, else per-tile
        interpolation).  Returns the flipped image point too, so the on-image
        marker is drawn at the same place its stage coordinate refers to.
        """
        fx, fy = self._apply_flip_img(ix, iy)
        sx, sy = self._img_to_stage_um(fx, fy)
        return fx, fy, sx, sy

    # -- calibration steps -----------------------------------------------------
    def _set_step(self, step):
        self._step = step
        self._btn_cen.configure(
            style="Active.TButton" if step==self.PICK_CENTRE else "Step.TButton")
        self._btn_xax.configure(
            style="Active.TButton" if step==self.PICK_XDIR  else "Step.TButton")
        self._btn_yax.configure(
            style="Active.TButton" if step==self.PICK_YDIR  else "Step.TButton")

    def _check_ready(self):
        ok = (self._centre_img is not None and self._xdir_img is not None
              and self._col_x and self._col_y and bool(self._csv_rows))
        self._btn_conv.configure(state="normal" if ok else "disabled")

    def _act_centre(self):
        if not self._csv_rows and self._mrc_image is None:
            messagebox.showwarning("No data","Load a CSV or MRC image first."); return
        self._set_step(self.PICK_CENTRE)
        self._sv.set("Left-click the image point that is the CSV coordinate origin.")

    def _act_xdir(self):
        if self._centre_img is None:
            messagebox.showwarning("No centre","Pick the image centre first."); return
        self._xdir_img = None
        self._set_step(self.PICK_XDIR)
        self._xax_var.set("X-axis:  click a point in +X direction")
        self._sv.set("Left-click any point in the +X direction of your CSV system.")

    def _act_ydir(self):
        if self._centre_img is None:
            messagebox.showwarning("No centre","Pick the image centre first."); return
        if self._xdir_img is None:
            messagebox.showwarning("No +X","Pick the +X direction first."); return
        self._set_step(self.PICK_YDIR)
        self._yax_var.set("Y-axis:  click to choose the +Y side")
        self._sv.set("Left-click on the side where +Y should point "
                     "(it stays perpendicular to +X).")

    def _sync_ydir_label(self):
        side   = "+90 CCW" if self._ydir_sign > 0 else "-90 CW"
        picked = "picked" if self._ydir_img is not None else "default"
        self._yax_var.set(f"Y-axis:  perpendicular, {side} ({picked})")
        self._angy_var.set(f"Y angle: {self._angle_y_deg:.2f} (CCW from image +X)")

    def _clear_ydir(self):
        """Drop the explicit +Y side pick and revert to the default (CCW)."""
        self._ydir_img = None
        self._ydir_sign = 1
        self._angle_y_deg = self._angle_deg + 90.0
        self._sync_ydir_label()
        if self._step == self.PICK_YDIR:
            self._set_step(self.IDLE)
        if self._centre_img and self._xdir_img:
            self._project_csv_to_image()
        if self._results:
            self._convert()
        else:
            self._redraw()
        self._sv.set("+Y reset to default (90 CCW from +X).")

    # -- click / motion --------------------------------------------------------
    def _on_click(self, event):
        if event.inaxes is not self._ax or event.xdata is None: return
        px, py = event.xdata, event.ydata

        if event.button == 3:
            if self._step == self.PICK_XDIR:
                self._xdir_img = None
                self._xax_var.set("X-axis:  not set"); self._ang_var.set("Angle:   -")
                self._redraw()
            elif self._step == self.PICK_YDIR:
                self._clear_ydir()
            elif self._step == self.PICK_CENTRE:
                self._centre_img = None; self._centre_stage = (0.0,0.0)
                self._cen_var.set("Centre:  not set"); self._redraw()
            return
        if event.button != 1: return

        if self._step == self.PICK_CENTRE:
            self._centre_img   = (px, py)
            sx, sy = self._img_to_stage_um(px, py)
            self._centre_stage = (sx, sy)
            self._cen_var.set(f"Centre: img ({px:.0f}, {py:.0f})\n"
                              f"         stage ({sx:.4f}, {sy:.4f}) um")
            self._set_step(self.IDLE); self._check_ready(); self._redraw()
            self._sv.set("Origin set.  Now click 'Pick +X Direction'.")

        elif self._step == self.PICK_XDIR:
            self._xdir_img  = (px, py)
            cx, cy = self._centre_img
            self._angle_deg = math.degrees(math.atan2(-(py-cy), (px-cx)))
            # +Y is always perpendicular to +X; keep it synced, preserving
            # whichever side (CCW/CW) is currently selected for +Y
            self._angle_y_deg = self._angle_deg + 90.0 * self._ydir_sign
            self._sync_ydir_label()
            self._xax_var.set(f"X-axis: centre -> ({px:.0f}, {py:.0f})")
            self._ang_var.set(f"Angle: {self._angle_deg:.2f}  (CCW from image +X)")
            self._set_step(self.IDLE)
            self._project_csv_to_image(); self._check_ready(); self._redraw()
            self._sv.set(f"+X set at {self._angle_deg:.2f}.  "
                         f"Pick +Y (optional) or click 'Convert Coordinates'.")

        elif self._step == self.PICK_YDIR:
            self._ydir_img  = (px, py)
            cx, cy = self._centre_img
            raw = math.degrees(math.atan2(-(py-cy), (px-cx)))
            rel = (raw - self._angle_deg + 180) % 360 - 180       # -180..180
            # snap to the perpendicular on the same side as the click
            self._ydir_sign = 1 if rel >= 0 else -1
            self._angle_y_deg = self._angle_deg + 90.0 * self._ydir_sign
            self._sync_ydir_label()
            self._set_step(self.IDLE)
            self._project_csv_to_image(); self._check_ready(); self._redraw()
            side = "CCW (+90)" if self._ydir_sign > 0 else "CW (-90)"
            self._sv.set(f"+Y locked perpendicular, {side} from +X.  "
                         f"Click 'Convert Coordinates'.")

    def _on_motion(self, event):
        if event.inaxes is not self._ax or event.xdata is None: return
        px, py = event.xdata, event.ydata
        sx, sy = self._img_to_stage_um(px, py)
        if self._centre_img and self._xdir_img:
            csv_x, csv_y = self._img_to_csv_um(px, py)
            self._sv.set(f"Stage ({sx:.4f}, {sy:.4f}) um   "
                         f"CSV ({csv_x:.4f}, {csv_y:.4f}) um   "
                         f"img ({px:.0f}, {py:.0f})")
        else:
            self._sv.set(f"Stage ({sx:.4f}, {sy:.4f}) um   img ({px:.0f}, {py:.0f})")

    # -- tree selection --------------------------------------------------------
    def _on_sel(self, _e):
        sel = self._tree.selection()
        if not sel: self._sel_idx = None; self._redraw(); return
        idx = self._tree.index(sel[0])
        if idx >= len(self._results): return
        self._sel_idx = idx
        px,py = self._results[idx]["img_x"], self._results[idx]["img_y"]
        xl,xr = self._ax.get_xlim(); yl,yr = self._ax.get_ylim()
        hw_x=(xr-xl)/2; hw_y=abs(yr-yl)/2
        self._xlim=(px-hw_x,px+hw_x); self._ylim=(py+hw_y,py-hw_y)
        self._redraw()

    # -- convert ---------------------------------------------------------------
    def _convert(self):
        if not self._col_x or not self._col_y:
            messagebox.showwarning("No columns","No X/Y columns detected in CSV."); return
        self._results = []; skipped = 0
        for i, row in enumerate(self._csv_rows):
            try:
                rx_um = float(row[self._col_x])
                ry_um = float(row[self._col_y])
            except (ValueError, KeyError):
                skipped += 1; continue
            ix, iy = self._csv_nm_to_img(rx_um*1000.0, ry_um*1000.0)
            fx, fy, sx, sy = self._predict_stage(ix, iy)
            label  = str(row.get(self._col_lbl, i+1)).strip() if self._col_lbl else str(i+1)
            self._results.append(dict(label=label,
                csv_x_um=rx_um, csv_y_um=ry_um,
                img_x=fx, img_y=fy,
                stage_x_um=sx, stage_y_um=sy, row=row))
        self._refresh_tree()
        self._btn_exp.configure(state="normal")
        self._btn_png.configure(state="normal")
        self._btn_png_full.configure(state="normal")
        self._redraw()
        msg = f"Converted {len(self._results)} points."
        if skipped: msg += f"  ({skipped} skipped)"
        flips = ("H" if self._flip_h_var.get() else "") + ("V" if self._flip_v_var.get() else "")
        if flips: msg += f"  [flip {flips}]"
        self._sv.set(msg)
        print([r["label"] for r in self._results[:10]])

    def _refresh_tree(self):
        self._tree.delete(*self._tree.get_children())
        for r in self._results:
            self._tree.insert("","end", values=(
                r["label"], f"{r['csv_x_um']:.4f}", f"{r['csv_y_um']:.4f}",
                f"{r['stage_x_um']:+.4f}", f"{r['stage_y_um']:+.4f}"))

    # -- export CSV ------------------------------------------------------------
    def _export(self):
        if not self._results:
            messagebox.showwarning("Empty","Run Convert first."); return
        path = filedialog.asksaveasfilename(title="Save converted CSV",
            defaultextension=".csv", filetypes=[("CSV","*.csv"),("All","*.*")])
        if not path: return
        try:
            cx_img,cy_img = self._centre_img
            csx,csy       = self._centre_stage
            if self._stage_fit is not None:
                map_desc = (("similarity rotation+flip" if self._stage_fit["reflect"]
                             else "similarity rotation")
                            + f" angle={self._stage_fit['angle_deg']:.4f}deg"
                            + f" rmse={self._stage_fit['rmse']:.4f}um")
            else:
                map_desc = "per-tile interpolation (no rotation)"
            with open(path,"w",newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["# source_csv",                    self._csv_path])
                w.writerow(["# source_mdoc",                   self._mdoc_path])
                w.writerow(["# source_mrc",                    self._mrc_path])
                w.writerow(["# csv_x_col",                     self._col_x])
                w.writerow(["# csv_y_col",                     self._col_y])
                w.writerow(["# csv_label_col",                 self._col_lbl or ""])
                w.writerow(["# centre_img_x_px",               f"{cx_img:.3f}"])
                w.writerow(["# centre_img_y_px",               f"{cy_img:.3f}"])
                w.writerow(["# centre_stage_x_um",             f"{csx:.6f}"])
                w.writerow(["# centre_stage_y_um",             f"{csy:.6f}"])
                w.writerow(["# x_axis_angle_deg_CCW_from_img_X", f"{self._angle_deg:.6f}"])
                w.writerow(["# y_axis_angle_deg_CCW_from_img_X", f"{self._angle_y_deg:.6f}"])
                w.writerow(["# y_axis_user_picked",            "yes" if self._ydir_img is not None
                                                               else "no (perpendicular default)"])
                w.writerow(["# flip_stage_x_horizontal",       "yes" if self._flip_h_var.get() else "no"])
                w.writerow(["# flip_stage_y_vertical",         "yes" if self._flip_v_var.get() else "no"])
                w.writerow(["# pixel_spacing_nm_per_px",       f"{self._px_nm:.6f}"])
                w.writerow(["# image_to_stage_map",            map_desc])
                w.writerow(["# all_coordinates_unit",          "um"])
                w.writerow([])
                extra = [k for k in self._results[0]["row"]
                         if k not in (self._col_x, self._col_y)]
                w.writerow(["label","csv_x_um","csv_y_um",
                             "stage_x_um","stage_y_um"] + extra)
                for r in self._results:
                    w.writerow([r["label"],
                                 f"{r['csv_x_um']:.6f}", f"{r['csv_y_um']:.6f}",
                                 f"{r['stage_x_um']:+.6f}", f"{r['stage_y_um']:+.6f}"]
                               + [r["row"].get(k,"") for k in extra])
            messagebox.showinfo("Exported", f"Saved {len(self._results)} rows to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    # -- export PNG: current view ----------------------------------------------
    def _export_png(self):
        """Save exactly what is visible on the canvas (current zoom/pan)."""
        path = filedialog.asksaveasfilename(
            title="Save current view as PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All", "*.*")])
        if not path: return
        try:
            self._fig.savefig(path, dpi=150, bbox_inches="tight",
                              facecolor=BG, edgecolor="none")
            self._sv.set(f"PNG saved: {os.path.basename(path)}")
            messagebox.showinfo("Exported", f"Current view saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("PNG export error", str(e))

    # -- export PNG: full image ------------------------------------------------
    def _export_png_full(self):
        """
        Render the entire image (ignoring current pan/zoom) with all POI
        markers and names, then save as a high-resolution PNG.
        """
        path = filedialog.asksaveasfilename(
            title="Save full image as PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All", "*.*")])
        if not path: return
        try:
            H, W = self._img_h, self._img_w
            # aim for ~2000 px on the long axis at 150 dpi
            long_side = max(H, W)
            fig_in    = max(8.0, long_side / 150.0)
            aspect    = H / W if W > 0 else 1.0

            fig2 = Figure(figsize=(fig_in, fig_in * aspect), facecolor=BG)
            ax2  = fig2.add_subplot(111)
            fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax2.set_facecolor(BG); ax2.axis("off")

            if self._mrc_image is not None:
                ax2.imshow(self._mrc_image, cmap="gray", origin="upper",
                           vmin=self._bc_lo_var.get(), vmax=self._bc_hi_var.get(),
                           aspect="equal", extent=[-0.5, W-0.5, H-0.5, -0.5], zorder=0)
            else:
                for xi in range(0, W+1, max(1, W//8)):
                    ax2.axvline(xi, color=BG3, lw=0.4)
                for yi in range(0, H+1, max(1, H//8)):
                    ax2.axhline(yi, color=BG3, lw=0.4)

            ax2.set_xlim(-0.5, W-0.5)
            ax2.set_ylim(H-0.5, -0.5)

            # reuse shared drawing helper with full-image xlim
            self._draw_markers(ax2, xlim=(-0.5, W-0.5), ylim=(H-0.5, -0.5))

            from matplotlib.backends.backend_agg import FigureCanvasAgg
            FigureCanvasAgg(fig2)
            fig2.savefig(path, dpi=150, bbox_inches="tight",
                         facecolor=BG, edgecolor="none")
            self._sv.set(f"Full-image PNG saved: {os.path.basename(path)}")
            messagebox.showinfo("Exported",
                f"Full image ({W}x{H} px canvas) saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("PNG export error", str(e))

    # -- reset -----------------------------------------------------------------
    def _reset(self):
        self._centre_img=None; self._xdir_img=None; self._ydir_img=None
        self._ydir_sign=1
        self._angle_deg=0.0; self._angle_y_deg=90.0
        self._results=[]; self._sel_idx=None; self._csv_img_pts=[]
        self._centre_stage=(0.0,0.0)
        self._flip_h_var.set(False); self._flip_v_var.set(False)
        self._cen_var.set("Centre:  not set")
        self._xax_var.set("X-axis:  not set"); self._ang_var.set("Angle:   -")
        self._yax_var.set("Y-axis:  perpendicular (default)")
        self._angy_var.set("Y angle: +90 from +X")
        self._btn_conv.configure(state="disabled")
        self._btn_exp.configure(state="disabled")
        self._btn_png.configure(state="disabled")
        self._btn_png_full.configure(state="disabled")
        self._tree.delete(*self._tree.get_children())
        self._set_step(self.IDLE); self._redraw()
        self._sv.set("Calibration reset.")

    # -- info panel ------------------------------------------------------------
    def _update_file_info(self):
        lines = []
        if self._csv_rows:
            lines.append(f"CSV: {len(self._csv_rows)} rows  "
                         f"X='{self._col_x}'  Y='{self._col_y}'"
                         + (f"  lbl='{self._col_lbl}'" if self._col_lbl else ""))
        if self._mdoc_pieces:
            img_size = self._mdoc_global.get("ImageSize",[512,512])
            if isinstance(img_size,(int,float)): iw=ih=int(img_size)
            else: iw,ih=int(img_size[0]),int(img_size[1])
            sps = [p["StagePosition"] for p in self._mdoc_pieces
                   if "StagePosition" in p]
            if sps:
                sp = sps[len(sps)//2]
                s_str = (f"mid stage ({float(sp[0]):.1f}, {float(sp[1]):.1f}) um"
                         if isinstance(sp,(list,tuple)) and len(sp)>=2 else "")
            else:
                s_str = "no StagePosition"
            lines.append(f"mdoc: {len(self._mdoc_pieces)} tiles  "
                         f"{iw}x{ih} px  {self._px_nm:.4f} nm/px  {s_str}")
            if self._stage_fit is not None:
                k = "rot+flip" if self._stage_fit["reflect"] else "rot"
                lines.append(f"img->stage: similarity {k} "
                             f"{self._stage_fit['angle_deg']:+.1f} deg, "
                             f"rmse {self._stage_fit['rmse']:.3f} um")
            else:
                lines.append("img->stage: per-tile interp (no rotation)")
        if self._mrc_image is not None:
            dH,dW = self._mrc_image.shape
            if (dW, dH) == (self._img_w, self._img_h):
                lines.append(f"MRC: assembled {self._img_w}x{self._img_h} px canvas")
            else:
                lines.append(f"MRC: full {self._img_w}x{self._img_h} px  "
                             f"(display {dW}x{dH})")
        self._file_info_var.set("\n".join(lines) if lines else "No files loaded")
        parts = []
        if self._csv_rows:    parts.append(f"CSV ok {len(self._csv_rows)} rows")
        if self._mdoc_pieces: parts.append(f"mdoc ok {len(self._mdoc_pieces)} tiles")
        if self._mrc_image is not None:
            parts.append(f"MRC ok {self._img_w}x{self._img_h}")
        self._info_var.set("  |  ".join(parts) if parts else "No files loaded")


# ==============================================================================
if __name__ == "__main__":
    App().mainloop()