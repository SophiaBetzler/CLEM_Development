
import re
import os
import mrcfile
import numpy as np
from pathlib import Path


class MRCReader:

    COORD_KEYS = ("RefinedPieceCoordinates", "AlignedPieceCoordsVS",
                  "AlignedPieceCoords", "PieceCoordinates")


    MONTAGE_FLIP_X = False
    MONTAGE_FLIP_Y = True

    def __init__(self, coord_key, path, refine_alignment=True, section=0):
        self.coord_key = coord_key
        self.output_root = path
        self.section = section
        self.refine_alignment = refine_alignment
        self.mrc_image      = None          # single-image mode; None until loaded
        self.montages       = {}            # {section: assembled array}
        self.section_pieces = {}            # {section: [tile dicts]}
        self.pixel_spacing_um = None
        self._img_hw = None
        self._feather_px = None
        self._global_info = None

    # ------------------------------------------------------------------ #
    # Small helpers for stage coordinate readout
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_coords(piece, coord_key):
        v = piece.get(coord_key)
        if v is None:
            raise KeyError(
                f"Tile ZValue={piece.get('ZValue')} has no '{coord_key}'.")
        return v
    
        
    @staticmethod
    def _px(piece, field):
        c = piece.get(field)
        return (float(c[0]), float(c[1])) if c is not None else None
    
    @staticmethod
    def _stage_xy(piece):
        stage_position = piece.get("StagePosition")
        if isinstance(stage_position, (list, tuple)) and len(stage_position) >= 2:
            return float(stage_position[0]), float(stage_position[1])
        return None
    
    def _piece_stage_z(self, piece):
        stage_position = piece.get("StagePosition")
        if isinstance(stage_position, (list, tuple)) and len(stage_position) >= 3:
            return float(stage_position[2])
        
        for k in ("StageZ", "Z"):
            v = piece.get(k)
            if v is not None:
                return float(v[0] if isinstance(v, (list, tuple)) else v)
            
        global_info = self._global_info
        if global_info:
            for k in ("StageZ", "Z"):
                v = global_info.get(k)
                if v is not None:
                    return float(v[0] if isinstance(v, (list, tuple)) else v)
        return None

    def _section_rotation_angle(self, pieces):
        angles = []
        for piece in pieces:
            angle = piece.get("RotationAngle", 0.0)
            if isinstance(angle, (list, tuple)):
                angle = float(angle[0]) if angle else 0.0
            angles.append(float(angle or 0.0))
        return float(np.median(angles)) if angles else 0.0
    
    def _coerce(self, val_str):
        parts = val_str.split()
        if not parts:
            return val_str
        try:
            nums = [int(p) if re.fullmatch(r"-?\d+", p) else float(p)
                    for p in parts]
            return nums[0] if len(nums) == 1 else nums
        except ValueError:
            return val_str.strip()

    # ------------------------------------------------------------------ #
    # Small helpers for montage display
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_weight_map(h, w, feather_px):
        feather_px = max(1, int(feather_px))

        def ramp(n):
            r = np.ones(n, dtype=np.float32)
            f = min(feather_px, n // 2)
            if f > 0:
                t = np.linspace(0.0, np.pi / 2, f, dtype=np.float32)
                r[:f] = np.sin(t); r[-f:] = np.sin(t)[::-1]
            return r
        return np.outer(ramp(h), ramp(w))

        
    def _normalize_image(self, img):
        img = np.nan_to_num(img.astype(np.float32))
        lo, hi = img.min(), img.max()
        return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    
        
    def _auto_brightness_contrast(self, img, percentiles=(1.0, 99.8), ignore_zeros=True, ignore_whites=True, white_cutoff=0.995):
        img = np.nan_to_num(img.astype(np.float32))
        sample = img[np.isfinite(img)]

        mask = np.ones(sample.shape, dtype=bool)
        if ignore_zeros: mask &= sample > 0
        if ignore_whites: mask &= sample < white_cutoff

        trimmed = sample[mask]
        if trimmed.size: sample = trimmed

        if not sample.size: return np.zeros_like(img)

        lo, hi = np.percentile(sample, percentiles)
        if hi <= lo: lo, hi = sample.min(), sample.max()
        if hi <= lo: return np.zeros_like(img)

        return np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    
    def _flip_for_display(self, arr):
        if self.MONTAGE_FLIP_X: arr = np.fliplr(arr)
        if self.MONTAGE_FLIP_Y: arr = np.flipud(arr)
        return arr


    # ------------------------------------------------------------------ #
    # File / coordinate-field discovery
    # ------------------------------------------------------------------ #

    def _get_site_folder(self, site_id):
        return Path(self.output_root) / site_id

    def _find_latest_montage_mrc(self, site_id):
        folder = self._get_site_folder(site_id)
        matches = [p for p in folder.glob("*montage*.mrc") if p.is_file()]
        if not matches:
            raise FileNotFoundError(f"No montage .mrc found in {folder}")
        return max(matches, key=lambda p: p.stat().st_mtime)
        
    def _find_latest_ome_tiff(self, site_id):
        folder = self._get_site_folder(site_id)

        matches = list(folder.glob("*.ome.tif")) + list(folder.glob("*.ome.tiff"))

        if not matches:
            raise FileNotFoundError(f"No OME-TIFF found in {folder}")

        return max(matches, key=lambda p: p.stat().st_mtime)

    def _find_mdoc_path(self, mrc_filepath):
        directory = os.path.dirname(mrc_filepath)
        stem, ext = os.path.splitext(mrc_filepath)
        candidates = [
                        os.fspath(mrc_filepath) + ".mdoc",                      # foo.mrc.mdoc
                        stem + ".mdoc",                           # foo.mdoc
                        stem + ext.replace(".", "_") + ".mdoc",   # foo_mrc.mdoc
                    ]
        for c in candidates:
            if os.path.isfile(c) and os.path.getsize(c) > 0: return c
        base = os.path.basename(stem)
        for fn in os.listdir(directory):
            full = os.path.join(directory, fn)
            if (fn.lower().endswith(".mdoc") and fn.startswith(base) and os.path.getsize(full) > 0):
                return full
        return None

    def _validate_coord_key(self, pieces):
        if self.coord_key not in self.COORD_KEYS:
            raise ValueError(
                f"coord_key must be one of {self.COORD_KEYS}, got {self.coord_key!r}")
        if not pieces:
            raise ValueError(
                "No tiles were parsed from the mdoc (0 [ZValue] blocks). "
                "The montage cannot be built - check the mdoc contents.")
        missing = [p.get("ZValue") for p in pieces
                   if p.get(self.coord_key) is None]
        if missing:
            shown = missing[:10]
            more = f" ... (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            raise KeyError(
                f"The field '{self.coord_key}' is not present for {len(missing)} tile(s) (ZValues: {shown}{more}). "
                f"Available, fully-populated fields: {self._available_coord_keys(pieces)}")

    def _available_coord_keys(self, pieces):
        return [k for k in self.COORD_KEYS if all(p.get(k) is not None for p in pieces)]

    def _build_section_pieces(self, pieces):
        section_map = {}
        for piece in pieces:
            pc = self._get_coords(piece, self.coord_key)
            sec = int(pc[2]) if isinstance(pc, (list, tuple)) and len(pc) >= 3 else 0
            section_map.setdefault(sec, []).append(piece)
        if not section_map:
            raise ValueError("No [ZValue] tiles found in the mdoc.")
        self.section_pieces = section_map
        return section_map

    def _display_key(self, mrc_filepath, ensure=True):
        if self.refine_alignment:
            if ensure and all(p.get("RefinedPieceCoordinates") is None
                              for p in self.section_pieces[self.section]):
                self.refine_tile_alignment(mrc_filepath=mrc_filepath)
            return "RefinedPieceCoordinates"
        return self.coord_key
    
    def identify_montage_file(self, site_id):
        if site_id is not None:
            matches = list(Path(os.path.join(self.output_root, site_id)).glob("*montage*.mrc"))
        else:
            matches = list(Path(self.output_root).glob("*montage*.mrc"))
        if not matches:
            raise FileNotFoundError(f"No montage .mrc found in {self.output_root}")
        montage_filename = str(max(matches, key=lambda p: p.stat().st_mtime))
        return montage_filename

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #

    
    def load_latest_from_site(self, site_id):

        mrc_path = self._find_latest_montage_mrc(site_id)
        ome_path = self._find_latest_ome_tiff(site_id)

        mrc_data = self.load_mrc_montage_data(mrc_path)
        tiff_data = self.load_ome_tiff_data(ome_path)

        data = {
            "site_id": site_id,
            "folder": self._get_site_folder(site_id),
            **mrc_data,
            **tiff_data,
        }

        self.current_data = data
        return data


    def load_mrc_single(self, mrc_filepath=None):
        if mrc_filepath is None:
            raise ValueError("No MRC file path provided.")
        with mrcfile.open(mrc_filepath, mode="r", permissive=True) as mrc:
            data = mrc.data.copy()
            voxel = mrc.voxel_size
            info = f"shape={data.shape}  voxel={voxel}"
        if data.ndim == 3:
            mid = data.shape[0] // 2
            data = data[mid]
            info += f"  (z={mid})"
        elif data.ndim != 2:
            raise ValueError(f"Unsupported MRC ndim={data.ndim}")
        self.mrc_image = self._normalize_image(data)
        return self.mrc_image, info

    def load_mrc_montage(self, mrc_filepath):
        mdoc_path = self._find_mdoc_path(mrc_filepath=mrc_filepath)
        if mdoc_path is None:
            raise FileNotFoundError(
                f"No .mdoc found next to {os.path.basename(mrc_filepath)}.")
        print(f"[INFO] Using mdoc: {mdoc_path}")

        global_info, pieces, mont_sections = self.parse_mdoc(mdoc_path)
        self._global_info = global_info
        self._validate_coord_key(pieces)
        print(f"[INFO] Using coordinate field: {self.coord_key}")

        ps_ang = global_info.get("PixelSpacing", 10000.0)
        if isinstance(ps_ang, (list, tuple)):
            ps_ang = float(ps_ang[0])
        self.pixel_spacing_um = float(ps_ang) / 10000.0

        img_size = global_info.get("ImageSize", [4096, 4096])
        if isinstance(img_size, (int, float)):
            img_w = img_h = int(img_size)
        else:
            img_w, img_h = int(img_size[0]), int(img_size[1])

        ps = global_info.get("PieceSpacing", [img_w - 410, img_h - 410])
        if isinstance(ps, (int, float)):
            ps_x = ps_y = int(ps)
        else:
            ps_x, ps_y = int(ps[0]), int(ps[1])
        feather_px = max(1, min(img_w - ps_x, img_h - ps_y))

        self._img_hw = (img_h, img_w)
        self._feather_px = feather_px
        self._build_section_pieces(pieces)

        self.montages = {}
        saved_section = self.section
        for sec in sorted(self.section_pieces):
            self.section = sec
            print(f"[INFO] Assembling section {sec}: "
                  f"{len(self.section_pieces[sec])} tiles")
            montage, min_x, min_y = self._assemble_montage(mrc_filepath, img_h, img_w, feather_px, self.coord_key)
            self.montages[sec] = montage
        self.section = saved_section

        print(f"[INFO] Built {len(self.montages)} montage(s) at "
              f"{self.pixel_spacing_um:.4f} um/px")
        return self.montages
    
    def load_mrc_montage_data(self, mrc_path):
        montages = self.load_mrc_montage(mrc_path)

        return {
            "mrc_path": mrc_path,
            "montages": montages,
            "section_pieces": self.section_pieces,
            "img_hw": self._img_hw,
            "feather_px": self._feather_px,
            "pixel_spacing_um": self.pixel_spacing_um,
        }


    def load_ome_tiff(self, ome_path):
        import tifffile

        ome_path = os.fspath(ome_path)

        with tifffile.TiffFile(ome_path) as tf:
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
            data = data[np.newaxis]
            axes = "Y" + axes

        for dim in ["C", "Z"]:
            if dim not in axes:
                data = data[np.newaxis]
                axes = dim + axes

        order = [axes.index(dim) for dim in "CZYX"]
        data = np.transpose(data, order)

        normed = np.zeros(data.shape, dtype=np.float32)
        for c in range(data.shape[0]):
            normed[c] = self._normalize_image(data[c])

        return normed, info
    
    def load_ome_tiff_data(self, ome_path):
        tiff_stack, tiff_info = self.load_ome_tiff(ome_path)

        return {
            "ome_path": ome_path,
            "tiff_stack": tiff_stack,
            "tiff_info": tiff_info,
        }

    def parse_mdoc(self, mdoc_filepath):
        mdoc_path = mdoc_filepath
        global_info, pieces, mont_sections = {}, [], []
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
                    current[key.strip()] = self._coerce(val.strip())

        for piece in pieces:
            for key in self.COORD_KEYS:
                piece.setdefault(key, None)

        print(f"[INFO] Parsed {len(pieces)} tiles, {len(mont_sections)} mont section(s).")
        return global_info, pieces, mont_sections

    # ------------------------------------------------------------------ #
    # Montage assembly
    # ------------------------------------------------------------------ #

    def _assemble_montage(self, mdoc_filepath, img_h, img_w, feather_px, key):
        pieces = self.section_pieces[self.section]

        coords = [(p, float(self._get_coords(p, key)[0]), float(self._get_coords(p, key)[1])) for p in pieces]

        min_x = min(x for _, x, _ in coords)
        min_y = min(y for _, _, y in coords)
        placements = [(p, int(round(x - min_x)), int(round(y - min_y)))for p, x, y in coords]

        mont_w = max(dx + img_w for _, dx, _ in placements)
        mont_h = max(dy + img_h for _, _, dy in placements)

        canvas = np.zeros((mont_h, mont_w), dtype=np.float64)
        weights = np.zeros((mont_h, mont_w), dtype=np.float64)

        wmap = self._cosine_weight_map(img_h, img_w, feather_px).astype(np.float64)

        z_indices = [p["ZValue"] for p in pieces]

        with mrcfile.open(mdoc_filepath, mode="r", permissive=True) as mrc:
            data = mrc.data
            if data is None:
                raise ValueError("MRC contains no image data.")
            if data.ndim == 2:
                return self._normalize_image(data.astype(np.float32))
            if data.ndim != 3:
                raise ValueError(f"Unsupported MRC shape {data.shape}")
            n_frames = data.shape[0]
            tiles = {z: data[z].astype(np.float64)
                     for z in z_indices if z < n_frames}

        for piece, x, y in coords:
            z = piece["ZValue"]
            if z not in tiles:
                continue
            dx = int(round(x - min_x)); dy = int(round(y - min_y))
            canvas[dy:dy + img_h, dx:dx + img_w] += tiles[z] * wmap
            weights[dy:dy + img_h, dx:dx + img_w] += wmap

        valid = weights > 0
        canvas[valid] /= weights[valid]
        return (self._normalize_image(canvas.astype(np.float32)), float(min_x), float(min_y))

    # ------------------------------------------------------------------ #
    # Determine stage rotation vs montage
    # ------------------------------------------------------------------ #

    def _fit_pixel_to_stage_rotation_matrix(self, pieces):
        pts = [(p["px"], p["stage"]) for p in pieces if p.get("px") is not None and p.get("stage") is not None]

        if len(pts) <3:
            return None, {"n": len(pts), "message": "Not enough points to fit rotation."}
        
        P = np.array([[p[0], p[1]] for p, _ in pts], float)
        S = np.array([[s[0], s[1]] for _, s in pts], float)
        A = np.hstack([P, np.ones((len(P), 1), float)])
        sol, *_ = np.linalg.lstsq(A, S, rcond=None)
        M, t = sol[:2].T, sol[2]

        residuals = np.hypot(*(P @ M.T + t - S).T)
        info = {"n": len(pts),
                "det": float(np.linalg.det(M)),
                "angle_deg": float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
                "rms_um": float(np.sqrt(np.mean(residuals ** 2))),
                "max_um": float(residuals.max()),}
        
        return M, info
        


    # ------------------------------------------------------------------ #
    # Refinement
    # ------------------------------------------------------------------ #
    def refine_tile_alignment(self, mrc_filepath):
        from skimage.registration import phase_cross_correlation as _pcc

        OUT_KEY = "RefinedPieceCoordinates"
        pieces = self.section_pieces[self.section]
        n = len(pieces)
        if n < 2:
            print("[refine] only one tile - nothing to refine.")
            return pieces

        img_h, img_w = self._img_hw
        overlap = self._feather_px
        ps_x, ps_y = img_w - overlap, img_h - overlap
        max_shift = overlap

        def cur_pos(p):
            c = self._get_coords(p, self.coord_key)
            return float(c[0]), float(c[1])

        grid = {}
        for p in pieces:
            pc = p.get("PieceCoordinates")
            if pc is None:
                pc = self._get_coords(p, self.coord_key)
            gx = int(round(float(pc[0]) / ps_x)) if ps_x > 0 else 0
            gy = int(round(float(pc[1]) / ps_y)) if ps_y > 0 else 0
            grid[(gx, gy)] = p

        with mrcfile.open(mrc_filepath, mode="r", permissive=True) as mrc:
            if mrc.data is None or mrc.data.ndim != 3:
                print("[refine] MRC is a single 2-D image - cannot refine.")
                return pieces
            n_frames = mrc.data.shape[0]
            tiles = {p["ZValue"]: mrc.data[p["ZValue"]].astype(np.float32) for p in pieces if p["ZValue"] < n_frames}

        z_list = [p["ZValue"] for p in pieces]
        z2idx = {z: i for i, z in enumerate(z_list)}

        def _norm(a):
            a = a.astype(np.float32); s = a.std()
            return (a - a.mean()) / s if s > 0 else a - a.mean()

        A_rows, b_rows, n_pairs = [], [], 0
        for (gx, gy), p_ref in sorted(grid.items()):
            zr = p_ref["ZValue"]
            if zr not in tiles:
                continue
            for direction, (dgx, dgy) in [("R", (1, 0)), ("D", (0, 1))]:
                nbr = (gx + dgx, gy + dgy)
                if nbr not in grid:
                    continue
                zm = grid[nbr]["ZValue"]
                if zm not in tiles:
                    continue
                n_pairs += 1
                if direction == "R":
                    ref_strip, mov_strip = tiles[zr][:, -overlap:], tiles[zm][:, :overlap]
                else:
                    ref_strip, mov_strip = tiles[zr][-overlap:, :], tiles[zm][:overlap, :]
                try:
                    raw = _pcc(_norm(ref_strip), _norm(mov_strip), upsample_factor=10, normalization='phase')
                    shift = raw[0] if isinstance(raw, tuple) else raw
                    dy, dx = float(shift[0]), float(shift[1])
                except Exception:
                    dy, dx = 0.0, 0.0

                if abs(dy) > max_shift: dy = 0.0
                if abs(dx) > max_shift: dx = 0.0
                i_r, i_m = z2idx[zr], z2idx[zm]

                def add(var_j, var_i, rhs):
                    row = np.zeros(2 * n); row[var_j] = 1.0; row[var_i] = -1.0
                    A_rows.append(row); b_rows.append(rhs)

                if direction == "R":
                    add(2 * i_m, 2 * i_r, ps_x + dx); add(2 * i_m + 1, 2 * i_r + 1, dy)
                else:
                    add(2 * i_m, 2 * i_r, dx);        add(2 * i_m + 1, 2 * i_r + 1, ps_y + dy)

        if not A_rows:
            print("[refine] no adjacent tile pairs measured.")
            return pieces

        x0, y0 = cur_pos(pieces[0])
        aw = 1e4
        row = np.zeros(2 * n); row[0] = aw; A_rows.append(row); b_rows.append(x0 * aw)
        row = np.zeros(2 * n); row[1] = aw; A_rows.append(row); b_rows.append(y0 * aw)

        A = np.array(A_rows, dtype=np.float64)
        b = np.array(b_rows, dtype=np.float64)
        result, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)

        for i, p in enumerate(pieces):
            src = self._get_coords(p, self.coord_key)
            z_val = src[2] if isinstance(src, (list, tuple)) and len(src) >= 3 else 0
            p[OUT_KEY] = [float(result[2 * i]), float(result[2 * i + 1]), z_val]

        self.montages.pop(self.section, None)

        deltas = []
        for i, p in enumerate(pieces):
            src = self._get_coords(p, self.coord_key)
            dx = float(result[2 * i])     - float(src[0])
            dy = float(result[2 * i + 1]) - float(src[1])
            deltas.append((dx * dx + dy * dy) ** 0.5)
        deltas = np.asarray(deltas)
        residual = float(np.sqrt(np.mean((A @ result - b) ** 2)))
        full_rank = (rank == 2 * n)

        print("\n========== Refinement summary ==========")
        print(f"  section            : {self.section}")
        print(f"  tiles refined      : {n}")
        print(f"  correlation pairs  : {n_pairs}")
        print(f"  system rank        : {rank} / {2 * n}  "
              f"({'full' if full_rank else 'RANK-DEFICIENT'})")
        print(f"  solve residual     : {residual:.2f} px "
              f"({residual * self.pixel_spacing_um:.4f} um)")
        print(f"  tile shift from {self.coord_key}:")
        print(f"      mean / median  : {deltas.mean():.1f} / "
              f"{np.median(deltas):.1f} px")
        print(f"      min / max      : {deltas.min():.1f} / {deltas.max():.1f} px")
        print(f"  source -> output   : '{self.coord_key}' -> '{OUT_KEY}'")
        print("========================================\n")
        return pieces
    


    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #
    def show(self, mrc_filepath, contrast_percentiles=(1.0, 99.0)):
        import matplotlib.pyplot as plt
        sec = self.section

        if self.section_pieces and sec in self.section_pieces:
            key = self._display_key(mrc_filepath=mrc_filepath)                 # refines if needed
            img = self._assemble_montage(mrc_filepath, *self._img_hw, self._feather_px, key)[0]

        elif self.mrc_image is not None:
            img, key, tile = self.mrc_image, None, None
        else:
            raise ValueError("Nothing loaded - call load_mrc_montage() or load_mrc_single() first.")

        max_px = 2000
        h, w = img.shape[:2]
        ds = max(1, max(h, w) // max_px)

        disp_img = self._auto_brightness_contrast(img, contrast_percentiles)
        disp_img = self._flip_for_display(disp_img)
        disp = disp_img[::ds, ::ds] if ds > 1 else disp_img

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(disp, cmap="gray", origin="upper", aspect="equal", vmin=0.0, vmax=1.0, extent=[-0.5, w - 0.5, h - 0.5, -0.5])
        ax.set_title(key or "single image")
        ax.axis("off")
        plt.tight_layout()
        plt.show()


    # ------------------------------------------------------------------ #
    # Create Dictionary Output Summarizing aligned Montage
    # ------------------------------------------------------------------ #

    def build_montage_summary(self, mrc_filepath):
        if self.refine_alignment:
            alignment = "fine"
            key = "RefinedPieceCoordinates"
        else:
            alignment = self.coord_key
            key = self.coord_key

        pieces = self.section_pieces[self.section]

        if self.refine_alignment and all(p.get("RefinedPieceCoordinates") is None
                                         for p in pieces):
            self.refine_tile_alignment(mrc_filepath=mrc_filepath)

        pix_um   = self.pixel_spacing_um
        theta       = np.deg2rad(self._section_rotation_angle(pieces))

        img, min_x, min_y = self._assemble_montage(mrc_filepath,*self._img_hw, self._feather_px, key)
                
        tiles = []
        for p in pieces:
            tiles.append({
                        "z": p.get("ZValue"), 
                        "stage_z": self._piece_stage_z(p),
                        "px": self._px(p, key),
                        "stage": self._stage_xy(p),
                        })
        M, fit = self._fit_pixel_to_stage_rotation_matrix(tiles)
        if M is not None:
            print(f"[fit] pixel -> stage n={fit['n']} angle={fit['angle_deg']:.2f} deg"
                  f"det={fit['det']:+.3e} ({'flip' if fit['det'] < 0 else 'no flip'}) "
                  f"rms={fit['rms_um']:.4f} um max={fit['max_um']:.4f} um")
        else:
            print(f"[fit] pixel -> stage n={fit['n']}  {fit['message']}")

        return {"image": img, "min_x": min_x, "min_y": min_y,
                "pixel_spacing_um": pix_um, "rotation_deg": float(np.rad2deg(theta)),
                "img_hw": self._img_hw, "section": self.section,
                "alignment": alignment, "tiles": tiles,
                "stage_matrix": M, "stage_fit": fit,
                "path": self.tem.path,
                "position": Path(mrc_filepath).parent.name }

    def run_montage_loader_and_create_summary(self, mrc_filepath):
        if not self.section_pieces:
            self.load_mrc_montage(mrc_filepath=mrc_filepath) 
        self.show(mrc_filepath=mrc_filepath, contrast_percentiles=(1.0, 99.))
        return self.build_montage_summary(mrc_filepath = mrc_filepath)
