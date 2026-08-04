"""
AutoLamella  -  Milling Position Extractor
==========================================
Pipeline for each lamella:
  0. Filter: lamellae whose defect state is not NONE (FAILURE, REWORK, ...)
     are excluded - shown in the table for reference, never exported.
  1. Read the per-lamella POI (positions[i].poi).  The POI is an offset in
     metres from the image center, and the image center corresponds to the
     MILLING stage position.  Decompose the POI (defined in the tilted image
     plane) into real-space y and z and add it to the stage position:
         cx = stage_x + poi_x              (x is unaffected by tilt)
         cy = stage_y + poi_y * cos(a)
         cz = stage_z + poi_y * sin(a)     where a = stage_tilt
  2. Project point onto z = 0 plane at 0 deg stage tilt:
         x_proj = cx
         y_proj = cy  +  cz * tan(a)
  3. Project reference onto z = 0 plane at 0 deg stage tilt, using its own
     tilt angle (default -23 deg, adjustable in the GUI):
         x_ref_proj = ref_x
         y_ref_proj = ref_y  +  ref_z * tan(ref_tilt)
  4. Subtract projected reference from projected point:
         x_final = x_proj - x_ref_proj
         y_final = y_proj - y_ref_proj
All coordinates in um.  Values in the YAML are in metres, tilt in radians.
Dependencies:
    pip install pyyaml
    tkinter  (stdlib on Windows/macOS; Linux: sudo apt install python3-tk)
"""
import os
import sys
import csv
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
try:
    import yaml
except ImportError:
    sys.exit("Missing dependency:  pip install pyyaml")
# palette
BG   = "#1e1e2e"
BG2  = "#313244"
BG3  = "#45475a"
FG   = "#cdd6f4"
ACC  = "#89b4fa"
ACC2 = "#a6e3a1"
RED  = "#f38ba8"
CYA  = "#89dceb"
YEL  = "#f9e2af"
MAG  = "#cba6f7"
ORG  = "#fab387"
M_TO_UM    = 1_000_000.0
RAD_TO_DEG = 180.0 / math.pi
# ══════════════════════════════════════════════════════════════════════════════
#  Core maths
# ══════════════════════════════════════════════════════════════════════════════
def project_to_zero_plane(x, y, z, stage_tilt_deg):
    """
    Project a 3-D stage position onto the z = 0 detector plane at 0 deg tilt.
        x_proj = x
        y_proj = y  +  z * tan(a)      where a = stage_tilt_deg
    """
    alpha = math.radians(stage_tilt_deg)
    if abs(math.cos(alpha)) < 1e-12:
        return x, float("nan")
    return x, y + z * math.tan(alpha)
DEFAULT_REF_TILT = -23.0
def process(yaml_path, ref_x=0.0, ref_y=0.0, ref_z=0.0,
            ref_tilt=DEFAULT_REF_TILT):
    with open(yaml_path, encoding="utf-8", errors="replace") as fh:
        data = yaml.safe_load(fh)
    rows = []
    for pos in data.get("positions", []):
        # defect state: anything other than NONE (FAILURE, REWORK, ...)
        # marks an errored / unfinished lamella -> excluded from export
        defect   = pos.get("defect") or {}
        dstate   = str(defect.get("state") or "NONE").upper()
        excluded = dstate != "NONE"
        poi = pos.get("poi") or {}
        sp  = (((pos.get("poses") or {})
                .get("MILLING") or {})
               .get("stage_position") or {})
        # stage position (metres -> um); image center == stage position
        sx  = float(sp.get("x") or 0) * M_TO_UM
        sy  = float(sp.get("y") or 0) * M_TO_UM
        sz  = float(sp.get("z") or 0) * M_TO_UM
        t_d = float(sp.get("t") or 0) * RAD_TO_DEG   # stage tilt in deg
        # per-lamella POI: offset from image center, in the tilted image
        # plane (metres -> um)
        px  = float(poi.get("x") or 0) * M_TO_UM
        py  = float(poi.get("y") or 0) * M_TO_UM
        # milling angle field (degrees already, or None) - informational only
        ma  = pos.get("milling_angle")
        alpha = math.radians(t_d)
        # step 1: decompose POI (image plane) into real-space y and z,
        #         then add to the stage position (= image center)
        #   poi_x is along stage X (unaffected by tilt)
        #   poi_y is along the image Y axis, which is tilted at angle a:
        #       real dy = poi_y * cos(a)
        #       real dz = poi_y * sin(a)
        cx = sx + px
        cy = sy + py * math.cos(alpha)
        cz = sz + py * math.sin(alpha)
        # step 2: project point onto z=0 plane at 0 deg tilt
        x_proj, y_proj = project_to_zero_plane(cx, cy, cz, t_d)
        # step 3: project reference onto z=0 plane using its own tilt angle
        x_ref_proj, y_ref_proj = project_to_zero_plane(ref_x, ref_y, ref_z,
                                                       ref_tilt)
        # step 4: subtract projected reference from projected point
        xf = x_proj - x_ref_proj
        yf = y_proj - y_ref_proj
        note = f"excluded - defect state {dstate}" if excluded else ""
        rows.append(dict(
            name          = pos.get("petname", ""),
            number        = pos.get("number", ""),
            # final output
            x_final       = xf,
            y_final       = yf,
            # projected point (before subtracting reference)
            x_proj        = x_proj,
            y_proj        = y_proj,
            # projected reference
            x_ref_proj    = x_ref_proj,
            y_ref_proj    = y_ref_proj,
            # stage + POI decomposed into real space (unprojected)
            combined_x    = cx,
            combined_y    = cy,
            combined_z    = cz,
            # raw inputs
            stage_x       = sx,
            stage_y       = sy,
            stage_z       = sz,
            poi_x         = px,
            poi_y         = py,
            stage_tilt    = t_d,
            mill_ang      = ma,
            ref_tilt      = ref_tilt,
            defect_state  = dstate,
            excluded      = excluded,
            note          = note,
        ))
    return rows
# ══════════════════════════════════════════════════════════════════════════════
#  CSV
# ══════════════════════════════════════════════════════════════════════════════
def _fv(v, nd=6):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v) if v else ""
def save_csv(rows, path, yaml_path="", ref_x=0.0, ref_y=0.0, ref_z=0.0):
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "grid_x_um", "grid_y_um", "stage_tilt_deg"])
        for r in rows:
            if r.get("excluded"):
                continue
            w.writerow([r["name"], _fv(r["x_final"]), _fv(r["y_final"]),
                        _fv(r["stage_tilt"], 3)])
            n += 1
    return n
# ══════════════════════════════════════════════════════════════════════════════
#  Table columns
# ══════════════════════════════════════════════════════════════════════════════
TCOLS = [
    ("name",         "Lamella name",                              140),
    ("defect_state", "Defect state",                               95),
    ("x_final",      "X final (um)",                              120),
    ("y_final",      "Y final (um)",                              120),
    ("stage_tilt",   "Stage tilt a (deg)",                         90),
    ("x_proj",       "X point projected (um)",                    160),
    ("y_proj",       "Y point projected (um)",                    160),
    ("x_ref_proj",   "X ref projected (um)",                      150),
    ("y_ref_proj",   "Y ref projected (um)",                      150),
    ("ref_tilt",     "Ref tilt (deg)",                             90),
    ("combined_x",   "X real = stage+POI (um)",                   150),
    ("combined_y",   "Y real = stage+POI*cos(a) (um)",            190),
    ("combined_z",   "Z real = stage+POI*sin(a) (um)",            190),
    ("stage_x",      "Stage X (um)",                               90),
    ("stage_y",      "Stage Y (um)",                               90),
    ("stage_z",      "Stage Z (um)",                               90),
    ("poi_x",        "POI image X (um)",                          110),
    ("poi_y",        "POI image Y (um)",                          110),
    ("mill_ang",     "Milling angle (deg)",                       110),
    ("note",         "Notes",                                     200),
]
TKEYS = [c[0] for c in TCOLS]
# ══════════════════════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AutoLamella  -  Milling Position Extractor")
        self.configure(bg=BG)
        self.minsize(1200, 700)
        self._rows  = []
        self._yaml  = ""
        self._scol  = "number"
        self._srev  = False
        self._build_styles()
        self._build_ui()
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
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
        s.configure("Mont.TButton",      background=MAG, foreground=BG,
                    font=("Segoe UI", 10, "bold"), padding=5)
        s.map("Mont.TButton", background=[("active", CYA), ("disabled", BG3)])
        s.configure("TLabelframe",       background=BG, relief="groove")
        s.configure("TLabelframe.Label", background=BG, foreground=CYA,
                    font=("Segoe UI", 10, "bold"))
        s.configure("TEntry",            fieldbackground=BG2,
                    foreground=FG, insertcolor=FG, font=("Consolas", 10))
        s.configure("Treeview",          background=BG2, foreground=FG,
                    fieldbackground=BG2, rowheight=22)
        s.configure("Treeview.Heading",  background=BG3, foreground=FG,
                    font=("Segoe UI", 9, "bold"))
        s.map("Treeview", background=[("selected", ACC)])
        s.configure("TSeparator", background=BG3)
    def _build_ui(self):
        PAD = 8
        top = ttk.Frame(self, padding=(PAD, PAD, PAD, 0))
        top.pack(fill="x")
        ttk.Button(top, text="Load YAML", style="Mont.TButton",
                   command=self._load_yaml).pack(side="left", padx=3)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(top, text="Export CSV", style="Accent.TButton",
                   command=self._export).pack(side="left", padx=3)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(top, text="Reset", style="Danger.TButton",
                   command=self._reset).pack(side="left", padx=3)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=6)
        self._info_var = tk.StringVar(value="No file loaded")
        ttk.Label(top, textvariable=self._info_var,
                  style="Sm.TLabel", foreground=BG3).pack(side="left", padx=4)
        # reference
        ref_lf = ttk.LabelFrame(self,
                     text="Reference position  -  defines the (0, 0, 0) origin",
                     padding=(PAD, 6))
        ref_lf.pack(fill="x", padx=PAD, pady=(6, 0))
        ttk.Label(ref_lf,
                  text="Stage coordinates (um) and tilt (deg) of the reference "
                       "point.  Both the point and the reference are projected "
                       "to z = 0 independently before the delta is calculated.",
                  style="Sm.TLabel").grid(row=0, column=0, columnspan=9,
                                          sticky="w", pady=(0, 6))
        self._ref_x = tk.StringVar(value="0.0")
        self._ref_y = tk.StringVar(value="0.0")
        self._ref_z = tk.StringVar(value="0.0")
        self._ref_t = tk.StringVar(value=f"{DEFAULT_REF_TILT:.1f}")
        for col, (lbl, attr) in enumerate([
            ("Ref X (um)",     "_ref_x"),
            ("Ref Y (um)",     "_ref_y"),
            ("Ref Z (um)",     "_ref_z"),
            ("Ref tilt (deg)", "_ref_t"),
        ]):
            ttk.Label(ref_lf, text=lbl, style="Sm.TLabel").grid(
                row=1, column=col*2, sticky="e",
                padx=(0 if col == 0 else 16, 4))
            ttk.Entry(ref_lf, textvariable=getattr(self, attr),
                      width=16).grid(row=1, column=col*2+1, sticky="w")
            getattr(self, attr).trace_add("write", lambda *_: self._recalc())
        ttk.Button(ref_lf, text="Recalculate",
                   command=self._recalc).grid(row=1, column=8, padx=(20, 0))
        # formula
        fml_lf = ttk.LabelFrame(self, text="Transform pipeline", padding=(PAD, 4))
        fml_lf.pack(fill="x", padx=PAD, pady=(6, 0))
        ttk.Label(fml_lf, text=(
            "  Step 0  -  Lamellae with defect state != NONE (FAILURE, REWORK, ...) are\n"
            "             excluded from the CSV export (shown red in the table).\n"
            "\n"
            "  Step 1  -  POI = offset from image center; image center = stage position.\n"
            "             Decompose POI (image plane) into real-space y and z, add to stage:\n"
            "             cx = stage_x + poi_x\n"
            "             cy = stage_y + poi_y * cos(a)       (poi_y along tilted image Y)\n"
            "             cz = stage_z + poi_y * sin(a)       a = stage_tilt\n"
            "\n"
            "  Step 2  -  Project point onto z = 0 plane at 0 deg tilt:\n"
            "             x_proj = cx\n"
            "             y_proj = cy  +  cz * tan(a)\n"
            "\n"
            "  Step 3  -  Project reference onto z = 0 plane using its own tilt angle\n"
            "             (default -23 deg, editable above):\n"
            "             x_ref_proj = ref_x\n"
            "             y_ref_proj = ref_y  +  ref_z * tan(ref_tilt)\n"
            "\n"
            "  Step 4  -  Subtract projected reference from projected point:\n"
            "             x_final = x_proj - x_ref_proj\n"
            "             y_final = y_proj - y_ref_proj"
        ), style="Sm.TLabel", foreground=YEL,
           font=("Consolas", 9), justify="left").pack(anchor="w")
        # table
        tbl_lf = ttk.LabelFrame(self,
                     text="Results  -  click any column header to sort  "
                          "  |  green = exported  |  red = excluded "
                          "(defect state FAILURE / REWORK)",
                     padding=4)
        tbl_lf.pack(fill="both", expand=True, padx=PAD, pady=(6, 0))
        tbl_lf.rowconfigure(0, weight=1)
        tbl_lf.columnconfigure(0, weight=1)
        self._tv = ttk.Treeview(tbl_lf, columns=TKEYS,
                                show="headings", selectmode="browse")
        for key, lbl, w in TCOLS:
            self._tv.heading(key, text=lbl,
                             command=lambda k=key: self._sort(k))
            self._tv.column(key, width=w, anchor="center", minwidth=40)
        self._tv.tag_configure("ok",   foreground=ACC2)
        self._tv.tag_configure("excl", foreground=RED)
        vsb = ttk.Scrollbar(tbl_lf, orient="vertical",   command=self._tv.yview)
        hsb = ttk.Scrollbar(tbl_lf, orient="horizontal", command=self._tv.xview)
        self._tv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tv.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        # status bar
        bot = ttk.Frame(self, padding=(PAD, 2, PAD, PAD))
        bot.pack(fill="x")
        ttk.Separator(bot, orient="horizontal").pack(fill="x", pady=(0, 3))
        self._status = tk.StringVar(value="Load an experiment.yaml to begin.")
        ttk.Label(bot, textvariable=self._status,
                  style="Sm.TLabel", foreground=CYA).pack(side="left")
    def _ref(self):
        def _f(attr, default=0.0):
            try:    return float(getattr(self, attr).get())
            except: return default
        return (_f("_ref_x"), _f("_ref_y"), _f("_ref_z"),
                _f("_ref_t", DEFAULT_REF_TILT))
    def _load_yaml(self):
        path = filedialog.askopenfilename(
            title="Open experiment.yaml",
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")])
        if path:
            self._do_load(path)
    def _do_load(self, path):
        self._status.set(f"Loading {os.path.basename(path)} ...")
        self.update_idletasks()
        try:
            rx, ry, rz, rt = self._ref()
            self._rows = process(path, rx, ry, rz, rt)
            self._yaml = path
        except Exception as e:
            messagebox.showerror("Load error", str(e)); return
        self._populate()
        n_bad = sum(1 for r in self._rows if r["excluded"])
        n_ok  = len(self._rows) - n_bad
        self._info_var.set(
            f"{os.path.basename(path)}  |  {len(self._rows)} lamellae  |  "
            f"{n_ok} ok" + (f"  |  {n_bad} excluded" if n_bad else ""))
        self._status.set(
            f"Loaded {len(self._rows)} positions ({n_bad} excluded).  "
            "Edit the reference coordinates above - the table updates live.")
    def _recalc(self):
        if not self._yaml: return
        try:
            rx, ry, rz, rt = self._ref()
            self._rows = process(self._yaml, rx, ry, rz, rt)
        except Exception as e:
            self._status.set(f"Error: {e}"); return
        self._populate()
        self._status.set(
            f"Recalculated  -  reference ({rx:.3f}, {ry:.3f}, {rz:.3f}) um "
            f"at {rt:.2f} deg")
    def _populate(self):
        def key(r):
            v = r.get(self._scol)
            if v is None: return (1, "")
            try:    return (0, float(v))
            except: return (1, str(v))
        data = sorted(self._rows, key=key, reverse=self._srev)
        self._tv.delete(*self._tv.get_children())
        for r in data:
            tag  = "excl" if r["excluded"] else "ok"
            vals = []
            for k in TKEYS:
                v = r.get(k, "")
                if k == "number":
                    vals.append(str(int(v)) if isinstance(v, (int, float)) else "")
                elif isinstance(v, float):
                    nd = 2 if k in ("stage_tilt", "mill_ang", "ref_tilt") else 4
                    vals.append(_fv(v, nd))
                else:
                    vals.append(str(v) if v else "")
            self._tv.insert("", "end", values=vals, tags=(tag,))
    def _sort(self, col):
        self._srev = (self._scol == col) and not self._srev
        self._scol = col
        self._populate()
        arrow = " v" if self._srev else " ^"
        for k, lbl, _ in TCOLS:
            self._tv.heading(k, text=lbl + (arrow if k == col else ""))
    def _export(self):
        if not self._rows:
            messagebox.showwarning("Nothing to export", "Load a YAML file first.")
            return
        stem = os.path.splitext(os.path.basename(self._yaml))[0]
        path = filedialog.asksaveasfilename(
            title="Save milling positions",
            defaultextension=".csv",
            initialfile=stem + "_milling_positions.csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path: return
        try:
            rx, ry, rz, rt = self._ref()
            n = save_csv(self._rows, path, self._yaml, rx, ry, rz)
            n_bad = len(self._rows) - n
            self._status.set(f"Saved {n} rows ({n_bad} excluded)  ->  {path}")
            messagebox.showinfo("Exported",
                f"Saved {n} rows ({n_bad} excluded) to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))
    def _reset(self):
        self._rows = []; self._yaml = ""
        self._scol = "number"; self._srev = False
        self._ref_x.set("0.0"); self._ref_y.set("0.0"); self._ref_z.set("0.0")
        self._ref_t.set(f"{DEFAULT_REF_TILT:.1f}")
        self._tv.delete(*self._tv.get_children())
        self._info_var.set("No file loaded")
        for k, lbl, _ in TCOLS:
            self._tv.heading(k, text=lbl)
        self._status.set("Reset.  Load an experiment.yaml to begin.")
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        app.after(100, lambda: app._do_load(sys.argv[1]))
    app.mainloop()