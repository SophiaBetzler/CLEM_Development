import re
import os
import mrcfile
import numpy as np


class MRCReader:

    COORD_KEYS = ("RefinedPieceCoordinates", "AlignedPieceCoordsVS",
                  "AlignedPieceCoords", "PieceCoordinates")

    PIECE_FIELDS = {
        "stage_position":               "StagePosition",
        "piece_coordinates":            "PieceCoordinates",
        "aligned_piece_coordinates":    "AlignedPieceCoords",
        "aligned_piece_coordinates_vs": "AlignedPieceCoordsVS",
        "refined_piece_coordinates":    "RefinedPieceCoordinates",
    }

    def __init__(self, path, coord_key, refine_alignment=False, section=0):
        self.path = path
        self.coord_key = coord_key
        self.section = section
        self.refine_alignment = refine_alignment
        self.mrc_image      = None          # single-image mode; None until loaded
        self.montages       = {}            # {section: assembled array}
        self.section_pieces = {}            # {section: [tile dicts]}

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_coords(piece, coord_key):
        v = piece.get(coord_key)
        if v is None:
            raise KeyError(
                f"Tile ZValue={piece.get('ZValue')} has no '{coord_key}'.")
        return v

    @staticmethod
    def _cosine_weight_map(h, w, feather_px):
        """Per-tile blend weights: 1.0 in the centre, sine-tapered to 0 over
        feather_px at every edge, so overlapping tiles blend without seams."""
        feather_px = max(1, int(feather_px))

        def ramp(n):
            r = np.ones(n, dtype=np.float32)
            f = min(feather_px, n // 2)
            if f > 0:
                t = np.linspace(0.0, np.pi / 2, f, dtype=np.float32)
                r[:f] = np.sin(t); r[-f:] = np.sin(t)[::-1]
            return r
        return np.outer(ramp(h), ramp(w))

    def _coerce(self, val_str):
        "Transform mdoc value strings into ints / floats / lists / strings."
        parts = val_str.split()
        if not parts:
            return val_str
        try:
            nums = [int(p) if re.fullmatch(r"-?\d+", p) else float(p)
                    for p in parts]
            return nums[0] if len(nums) == 1 else nums
        except ValueError:
            return val_str.strip()

    def _normalize_image(self, img):
        img = np.nan_to_num(img.astype(np.float32))
        lo, hi = img.min(), img.max()
        return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)

    # ------------------------------------------------------------------ #
    # File / coordinate-field discovery
    # ------------------------------------------------------------------ #
    def _find_mdoc_path(self):
        directory = os.path.dirname(self.path)
        stem, ext = os.path.splitext(self.path)
        candidates = [
            self.path + ".mdoc",                      # foo.mrc.mdoc
            stem + ".mdoc",                           # foo.mdoc
            stem + ext.replace(".", "_") + ".mdoc",   # foo_mrc.mdoc
        ]
        for c in candidates:
            if os.path.isfile(c) and os.path.getsize(c) > 0:
                return c
        base = os.path.basename(stem)
        for fn in os.listdir(directory):
            full = os.path.join(directory, fn)
            if (fn.lower().endswith(".mdoc") and fn.startswith(base)
                    and os.path.getsize(full) > 0):
                return full
        return None

    def _validate_coord_key(self, pieces):
        if self.coord_key not in self.COORD_KEYS:
            raise ValueError(
                f"coord_key must be one of {self.COORD_KEYS}, got "
                f"{self.coord_key!r}")
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
                f"The field '{self.coord_key}' is not present for "
                f"{len(missing)} tile(s) (ZValues: {shown}{more}). "
                f"Available, fully-populated fields: "
                f"{self._available_coord_keys(pieces)}")

    def _available_coord_keys(self, pieces):
        return [k for k in self.COORD_KEYS
                if all(p.get(k) is not None for p in pieces)]

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

    # ------------------------------------------------------------------ #
    # Stage-position helpers
    # ------------------------------------------------------------------ #
    def _tile_center_stage_position(self, section):
        out = {}
        for p in self.section_pieces[section]:
            sp = p.get("StagePosition")
            if isinstance(sp, (list, tuple)) and len(sp) >= 2:
                out[p.get("ZValue")] = (float(sp[0]), float(sp[1]))
        return out

    def _get_stage_positions(self, rotated=True):
        field = "ShiftedRot" if rotated else "ShiftedPlain"
        out = {}
        for tile in self.section_pieces[self.section]:
            esp = tile.get("estimated_stage_position")
            if esp is None:
                raise RuntimeError(
                    "No estimated positions - call tile_alignment_shift() first.")
            out[tile.get("ZValue")] = esp[field]
        return out

    # ------------------------------------------------------------------ #
    # Montage assembly
    # ------------------------------------------------------------------ #
    def _assemble_one_montage(self, img_h, img_w, feather_px, key):
        pieces = self.section_pieces[self.section]

        mont_w = mont_h = 0
        for piece in pieces:
            c = self._get_coords(piece, key)
            mont_w = max(mont_w, int(round(float(c[0]))) + img_w)
            mont_h = max(mont_h, int(round(float(c[1]))) + img_h)

        z_indices = [piece["ZValue"] for piece in pieces]
        with mrcfile.open(self.path, mode="r", permissive=True) as mrc:
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

        wmap = self._cosine_weight_map(img_h, img_w, feather_px).astype(np.float64)
        canvas = np.zeros((mont_h, mont_w), dtype=np.float64)
        weights = np.zeros((mont_h, mont_w), dtype=np.float64)

        for piece in pieces:
            z_idx = piece["ZValue"]
            if z_idx not in tiles:
                continue
            coords = self._get_coords(piece, key)
            cx = int(round(float(coords[0])))
            cy = int(round(float(coords[1])))
            tile = tiles[z_idx]
            sy0 = max(0, -cy);              sx0 = max(0, -cx)
            sy1 = min(img_h, mont_h - cy);  sx1 = min(img_w, mont_w - cx)
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            dy0 = cy + sy0; dx0 = cx + sx0
            dy1 = dy0 + (sy1 - sy0); dx1 = dx0 + (sx1 - sx0)
            canvas [dy0:dy1, dx0:dx1] += tile[sy0:sy1, sx0:sx1] * wmap[sy0:sy1, sx0:sx1]
            weights[dy0:dy1, dx0:dx1] += wmap[sy0:sy1, sx0:sx1]

        valid = weights > 0
        canvas[valid] /= weights[valid]
        return self._normalize_image(canvas.astype(np.float32))

    def _display_key(self, ensure=True):
        if self.refine_alignment:
            if ensure and all(p.get("RefinedPieceCoordinates") is None
                              for p in self.section_pieces[self.section]):
                self.refine_tile_alignment()
            return "RefinedPieceCoordinates"
        return self.coord_key

    # ------------------------------------------------------------------ #
    # Per-tile alignment shift -> estimated stage position
    # ------------------------------------------------------------------ #
    def tile_alignment_shift(self):
        key = self._display_key()
        pix_um = self.pixel_spacing_um
        stage = self._tile_center_stage_position(self.section)

        for tile in self.section_pieces[self.section]:
            z = tile.get("ZValue")
            aligned = tile.get(key)
            nominal = tile.get("PieceCoordinates")
            stage_position = stage.get(z)

            if aligned is None or nominal is None or stage_position is None:
                tile["estimated_stage_position"] = {
                    "ShiftPx": None, "ShiftUm": None,
                    "ShiftedPlain": stage_position, "ShiftedRot": stage_position,
                }
                print(f"[WARNING] tile z={z}: no shift applied "
                      f"(missing {key}, PieceCoordinates, or StagePosition).")
                continue

            dx_px = float(aligned[0]) - float(nominal[0])
            dy_px = float(aligned[1]) - float(nominal[1])
            dx_um = dx_px * pix_um
            dy_um = dy_px * pix_um

            shifted_plain = (stage_position[0] + dx_um, stage_position[1] + dy_um)

            r = tile.get("RotationAngle", 0.0)
            th = np.deg2rad(float(r[0]) if isinstance(r, (list, tuple)) else float(r))
            cos, sin = np.cos(th), np.sin(th)
            shifted_rot = (stage_position[0] + (cos * dx_um - sin * dy_um),
                           stage_position[1] + (sin * dx_um + cos * dy_um))

            tile["estimated_stage_position"] = {
                "ShiftPx":      (dx_px, dy_px),
                "ShiftUm":      (dx_um, dy_um),
                "ShiftedPlain": shifted_plain,
                "ShiftedRot":   shifted_rot,
            }
        return self.section_pieces[self.section]

    # ------------------------------------------------------------------ #
    # mdoc parsing
    # ------------------------------------------------------------------ #
    def parse_mdoc(self, mdoc_path=None):
        mdoc_path = mdoc_path or self.path
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

        # Guarantee all coordinate variants exist (None when absent).
        for piece in pieces:
            for key in self.COORD_KEYS:
                piece.setdefault(key, None)

        print(f"[INFO] Parsed {len(pieces)} tiles, "
              f"{len(mont_sections)} mont section(s).")
        return global_info, pieces, mont_sections

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #
    def load_mrc_single(self):
        """Read the MRC with no mdoc / physical coordinates."""
        with mrcfile.open(self.path, mode="r", permissive=True) as mrc:
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

    def load_mrc_montage(self):
        mdoc_path = self._find_mdoc_path()
        if mdoc_path is None:
            raise FileNotFoundError(
                f"No .mdoc found next to {os.path.basename(self.path)}.")
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

        # Geometry / state needed by assembly + downstream methods.
        self._img_hw = (img_h, img_w)
        self._feather_px = feather_px
        self._build_section_pieces(pieces)

        # Build every section's montage from the loaded coordinate field.
        self.montages = {}
        saved_section = self.section
        for sec in sorted(self.section_pieces):
            self.section = sec
            print(f"[INFO] Assembling section {sec}: "
                  f"{len(self.section_pieces[sec])} tiles")
            self.montages[sec] = self._assemble_one_montage(
                img_h, img_w, feather_px, self.coord_key)
        self.section = saved_section

        print(f"[INFO] Built {len(self.montages)} montage(s) at "
              f"{self.pixel_spacing_um:.4f} um/px")
        return self.montages

    # ------------------------------------------------------------------ #
    # Refinement
    # ------------------------------------------------------------------ #
    def refine_tile_alignment(self):
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

        # Neighbour grid from NOMINAL coords (stable topology).
        grid = {}
        for p in pieces:
            pc = p.get("PieceCoordinates")
            if pc is None:
                pc = self._get_coords(p, self.coord_key)
            gx = int(round(float(pc[0]) / ps_x)) if ps_x > 0 else 0
            gy = int(round(float(pc[1]) / ps_y)) if ps_y > 0 else 0
            grid[(gx, gy)] = p

        with mrcfile.open(self.path, mode="r", permissive=True) as mrc:
            if mrc.data is None or mrc.data.ndim != 3:
                print("[refine] MRC is a single 2-D image - cannot refine.")
                return pieces
            n_frames = mrc.data.shape[0]
            tiles = {p["ZValue"]: mrc.data[p["ZValue"]].astype(np.float32)
                     for p in pieces if p["ZValue"] < n_frames}

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
                    raw = _pcc(_norm(ref_strip), _norm(mov_strip),
                               upsample_factor=10, normalization='phase')
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
    def show(self, annotate=True, rotated=True):
        import matplotlib.pyplot as plt
        sec = self.section

        if self.section_pieces and sec in self.section_pieces:
            key = self._display_key()                 # refines if needed
            if annotate and any("estimated_stage_position" not in p
                                for p in self.section_pieces[sec]):
                self.tile_alignment_shift()
            img = self._assemble_one_montage(*self._img_hw, self._feather_px, key)
            tiles = self.section_pieces[sec] if annotate else None
        elif self.mrc_image is not None:
            img, key, tiles = self.mrc_image, None, None
        else:
            raise ValueError("Nothing loaded - call load_mrc_montage() or "
                             "load_mrc_single() first.")

        max_px = 2000
        h, w = img.shape[:2]
        ds = max(1, max(h, w) // max_px)
        disp = img[::ds, ::ds] if ds > 1 else img

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(disp, cmap="gray", origin="upper", aspect="equal",
                  vmin=0.0, vmax=1.0, extent=[-0.5, w - 0.5, h - 0.5, -0.5])
        ax.set_title(key or "single image")

        if tiles is not None:
            img_h, img_w = self._img_hw
            for p in tiles:
                c  = self._get_coords(p, key)            # tile origin (px) for THIS key
                cx = float(c[0]) + img_w / 2.0           # -> tile centre (px)
                cy = float(c[1]) + img_h / 2.0
                esp = p.get("estimated_stage_position")
                plain = esp.get("ShiftedPlain") if esp else None
                rot   = esp.get("ShiftedRot")   if esp else None

                # Offset the two labels vertically so they stack, not overlap.
                # ~4% of a tile height reads well at montage scale.
                dy = img_h * 0.04

                if plain is not None:
                    ax.text(cx, cy - dy, f"{plain[0]:.1f}, {plain[1]:.1f}",
                            ha="center", va="bottom", fontsize=7, color="yellow",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="black", ec="none", alpha=0.5))
                if rot is not None:
                    ax.text(cx, cy + dy, f"{rot[0]:.1f}, {rot[1]:.1f}",
                            ha="center", va="top", fontsize=7, color="red",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="black", ec="none", alpha=0.5))
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def create_montage(self, single=False):
        if single:
            self.load_mrc_single()
            self.show()
        else:
            self.load_mrc_montage()
            if self.refine_alignment:
                self.refine_tile_alignment()
            self.tile_alignment_shift()
            self.show()
        return self.section_pieces

    # ------------------------------------------------------------------ #
    # Picker
    # ------------------------------------------------------------------ #
    import re
import os
import mrcfile
import numpy as np


class MRCReader:

    COORD_KEYS = ("RefinedPieceCoordinates", "AlignedPieceCoordsVS",
                  "AlignedPieceCoords", "PieceCoordinates")

    PIECE_FIELDS = {
        "stage_position":               "StagePosition",
        "piece_coordinates":            "PieceCoordinates",
        "aligned_piece_coordinates":    "AlignedPieceCoords",
        "aligned_piece_coordinates_vs": "AlignedPieceCoordsVS",
        "refined_piece_coordinates":    "RefinedPieceCoordinates",
    }

    def __init__(self, path, coord_key, refine_alignment=False, section=0):
        self.path = path
        self.coord_key = coord_key
        self.section = section
        self.refine_alignment = refine_alignment
        self.mrc_image      = None          # single-image mode; None until loaded
        self.montages       = {}            # {section: assembled array}
        self.section_pieces = {}            # {section: [tile dicts]}

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_coords(piece, coord_key):
        v = piece.get(coord_key)
        if v is None:
            raise KeyError(
                f"Tile ZValue={piece.get('ZValue')} has no '{coord_key}'.")
        return v

    @staticmethod
    def _cosine_weight_map(h, w, feather_px):
        """Per-tile blend weights: 1.0 in the centre, sine-tapered to 0 over
        feather_px at every edge, so overlapping tiles blend without seams."""
        feather_px = max(1, int(feather_px))

        def ramp(n):
            r = np.ones(n, dtype=np.float32)
            f = min(feather_px, n // 2)
            if f > 0:
                t = np.linspace(0.0, np.pi / 2, f, dtype=np.float32)
                r[:f] = np.sin(t); r[-f:] = np.sin(t)[::-1]
            return r
        return np.outer(ramp(h), ramp(w))

    def _coerce(self, val_str):
        "Transform mdoc value strings into ints / floats / lists / strings."
        parts = val_str.split()
        if not parts:
            return val_str
        try:
            nums = [int(p) if re.fullmatch(r"-?\d+", p) else float(p)
                    for p in parts]
            return nums[0] if len(nums) == 1 else nums
        except ValueError:
            return val_str.strip()

    def _normalize_image(self, img):
        img = np.nan_to_num(img.astype(np.float32))
        lo, hi = img.min(), img.max()
        return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)

    # ------------------------------------------------------------------ #
    # File / coordinate-field discovery
    # ------------------------------------------------------------------ #
    def _find_mdoc_path(self):
        directory = os.path.dirname(self.path)
        stem, ext = os.path.splitext(self.path)
        candidates = [
            self.path + ".mdoc",                      # foo.mrc.mdoc
            stem + ".mdoc",                           # foo.mdoc
            stem + ext.replace(".", "_") + ".mdoc",   # foo_mrc.mdoc
        ]
        for c in candidates:
            if os.path.isfile(c) and os.path.getsize(c) > 0:
                return c
        base = os.path.basename(stem)
        for fn in os.listdir(directory):
            full = os.path.join(directory, fn)
            if (fn.lower().endswith(".mdoc") and fn.startswith(base)
                    and os.path.getsize(full) > 0):
                return full
        return None

    def _validate_coord_key(self, pieces):
        if self.coord_key not in self.COORD_KEYS:
            raise ValueError(
                f"coord_key must be one of {self.COORD_KEYS}, got "
                f"{self.coord_key!r}")
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
                f"The field '{self.coord_key}' is not present for "
                f"{len(missing)} tile(s) (ZValues: {shown}{more}). "
                f"Available, fully-populated fields: "
                f"{self._available_coord_keys(pieces)}")

    def _available_coord_keys(self, pieces):
        return [k for k in self.COORD_KEYS
                if all(p.get(k) is not None for p in pieces)]

    def _build_section_pieces(self, pieces):
        """Group tiles into {section: [tiles]} by the 3rd element of the chosen
        coordinate field, and store as self.section_pieces."""
        section_map = {}
        for piece in pieces:
            pc = self._get_coords(piece, self.coord_key)
            sec = int(pc[2]) if isinstance(pc, (list, tuple)) and len(pc) >= 3 else 0
            section_map.setdefault(sec, []).append(piece)
        if not section_map:
            raise ValueError("No [ZValue] tiles found in the mdoc.")
        self.section_pieces = section_map
        return section_map

    # ------------------------------------------------------------------ #
    # Stage-position helpers
    # ------------------------------------------------------------------ #
    def _tile_center_stage_position(self, section):
        """{ZValue: (sx, sy)} recorded StagePosition per tile."""
        out = {}
        for p in self.section_pieces[section]:
            sp = p.get("StagePosition")
            if isinstance(sp, (list, tuple)) and len(sp) >= 2:
                out[p.get("ZValue")] = (float(sp[0]), float(sp[1]))
        return out

    def _get_stage_positions(self, rotated=True):
        """Read out the per-tile estimated stage position computed by
        tile_alignment_shift().  Returns {ZValue: (sx, sy)}."""
        field = "ShiftedRot" if rotated else "ShiftedPlain"
        out = {}
        for tile in self.section_pieces[self.section]:
            esp = tile.get("estimated_stage_position")
            if esp is None:
                raise RuntimeError(
                    "No estimated positions - call tile_alignment_shift() first.")
            out[tile.get("ZValue")] = esp[field]
        return out

    # ------------------------------------------------------------------ #
    # Montage assembly
    # ------------------------------------------------------------------ #
    def _assemble_one_montage(self, img_h, img_w, feather_px, key):
        """Assemble self.section from the given coordinate field `key`.
        Canvas extent is computed from that field's tile positions."""
        pieces = self.section_pieces[self.section]

        mont_w = mont_h = 0
        for piece in pieces:
            c = self._get_coords(piece, key)
            mont_w = max(mont_w, int(round(float(c[0]))) + img_w)
            mont_h = max(mont_h, int(round(float(c[1]))) + img_h)

        z_indices = [piece["ZValue"] for piece in pieces]
        with mrcfile.open(self.path, mode="r", permissive=True) as mrc:
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

        wmap = self._cosine_weight_map(img_h, img_w, feather_px).astype(np.float64)
        canvas = np.zeros((mont_h, mont_w), dtype=np.float64)
        weights = np.zeros((mont_h, mont_w), dtype=np.float64)

        for piece in pieces:
            z_idx = piece["ZValue"]
            if z_idx not in tiles:
                continue
            coords = self._get_coords(piece, key)
            cx = int(round(float(coords[0])))
            cy = int(round(float(coords[1])))
            tile = tiles[z_idx]
            sy0 = max(0, -cy);              sx0 = max(0, -cx)
            sy1 = min(img_h, mont_h - cy);  sx1 = min(img_w, mont_w - cx)
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            dy0 = cy + sy0; dx0 = cx + sx0
            dy1 = dy0 + (sy1 - sy0); dx1 = dx0 + (sx1 - sx0)
            canvas [dy0:dy1, dx0:dx1] += tile[sy0:sy1, sx0:sx1] * wmap[sy0:sy1, sx0:sx1]
            weights[dy0:dy1, dx0:dx1] += wmap[sy0:sy1, sx0:sx1]

        valid = weights > 0
        canvas[valid] /= weights[valid]
        return self._normalize_image(canvas.astype(np.float32))

    def _display_key(self, ensure=True):
        """The coordinate field that should be displayed/picked on:
        RefinedPieceCoordinates when refine_alignment is on (refining first if
        needed), else self.coord_key."""
        if self.refine_alignment:
            if ensure and all(p.get("RefinedPieceCoordinates") is None
                              for p in self.section_pieces[self.section]):
                self.refine_tile_alignment()
            return "RefinedPieceCoordinates"
        return self.coord_key
    
    def _tile_stage_z(self, piece):
        sp = piece.get("StagePosition")
        if isinstance(sp, (list, tuple)) and len(sp) >= 3:
            return float(sp[2])

        for k in ("StageZ", "Z"):
            v = piece.get(k)
            if v is not None:
                return float(v[0] if isinstance(v, (list, tuple)) else v)

        gi = getattr(self, "_global_info", None)
        if gi:
            for k in ("StageZ", "Z"):
                v = gi.get(k)
                if v is not None:
                    return float(v[0] if isinstance(v, (list, tuple)) else v)
        return None

    # ------------------------------------------------------------------ #
    # Per-tile alignment shift -> estimated stage position
    # ------------------------------------------------------------------ #
    def tile_alignment_shift(self):
        """Per tile, compute the alignment shift (key - PieceCoordinates) in
        px and um, and the resulting estimated stage position both without and
        with the RotationAngle applied.  Stored on each tile under
        'estimated_stage_position'."""
        key = self._display_key()
        pix_um = self.pixel_spacing_um
        stage = self._tile_center_stage_position(self.section)

        for tile in self.section_pieces[self.section]:
            z = tile.get("ZValue")
            aligned = tile.get(key)
            nominal = tile.get("PieceCoordinates")
            stage_position = stage.get(z)

            if aligned is None or nominal is None or stage_position is None:
                tile["estimated_stage_position"] = {
                    "ShiftPx": None, "ShiftUm": None,
                    "ShiftedPlain": stage_position, "ShiftedRot": stage_position,
                }
                print(f"[WARNING] tile z={z}: no shift applied "
                      f"(missing {key}, PieceCoordinates, or StagePosition).")
                continue

            dx_px = float(aligned[0]) - float(nominal[0])
            dy_px = float(aligned[1]) - float(nominal[1])
            dx_um = dx_px * pix_um
            dy_um = dy_px * pix_um

            shifted_plain = (stage_position[0] + dx_um, stage_position[1] + dy_um)

            r = tile.get("RotationAngle", 0.0)
            th = np.deg2rad(float(r[0]) if isinstance(r, (list, tuple)) else float(r))
            cos, sin = np.cos(th), np.sin(th)
            shifted_rot = (stage_position[0] + (cos * dx_um - sin * dy_um),
                           stage_position[1] + (sin * dx_um + cos * dy_um))

            tile["estimated_stage_position"] = {
                "ShiftPx":      (dx_px, dy_px),
                "ShiftUm":      (dx_um, dy_um),
                "ShiftedPlain": shifted_plain,
                "ShiftedRot":   shifted_rot,
            }
        return self.section_pieces[self.section]

    # ------------------------------------------------------------------ #
    # mdoc parsing
    # ------------------------------------------------------------------ #
    def parse_mdoc(self, mdoc_path=None):
        mdoc_path = mdoc_path or self.path
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

        # Guarantee all coordinate variants exist (None when absent).
        for piece in pieces:
            for key in self.COORD_KEYS:
                piece.setdefault(key, None)

        print(f"[INFO] Parsed {len(pieces)} tiles, "
              f"{len(mont_sections)} mont section(s).")
        return global_info, pieces, mont_sections

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #
    def load_mrc_single(self):
        """Read the MRC with no mdoc / physical coordinates."""
        with mrcfile.open(self.path, mode="r", permissive=True) as mrc:
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

    def load_mrc_montage(self):
        mdoc_path = self._find_mdoc_path()
        if mdoc_path is None:
            raise FileNotFoundError(
                f"No .mdoc found next to {os.path.basename(self.path)}.")
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

        # Geometry / state needed by assembly + downstream methods.
        self._img_hw = (img_h, img_w)
        self._feather_px = feather_px
        self._build_section_pieces(pieces)

        # Build every section's montage from the loaded coordinate field.
        self.montages = {}
        saved_section = self.section
        for sec in sorted(self.section_pieces):
            self.section = sec
            print(f"[INFO] Assembling section {sec}: "
                  f"{len(self.section_pieces[sec])} tiles")
            self.montages[sec] = self._assemble_one_montage(
                img_h, img_w, feather_px, self.coord_key)
        self.section = saved_section

        print(f"[INFO] Built {len(self.montages)} montage(s) at "
              f"{self.pixel_spacing_um:.4f} um/px")
        return self.montages

    # ------------------------------------------------------------------ #
    # Refinement
    # ------------------------------------------------------------------ #
    def refine_tile_alignment(self):
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

        # Neighbour grid from NOMINAL coords (stable topology).
        grid = {}
        for p in pieces:
            pc = p.get("PieceCoordinates")
            if pc is None:
                pc = self._get_coords(p, self.coord_key)
            gx = int(round(float(pc[0]) / ps_x)) if ps_x > 0 else 0
            gy = int(round(float(pc[1]) / ps_y)) if ps_y > 0 else 0
            grid[(gx, gy)] = p

        with mrcfile.open(self.path, mode="r", permissive=True) as mrc:
            if mrc.data is None or mrc.data.ndim != 3:
                print("[refine] MRC is a single 2-D image - cannot refine.")
                return pieces
            n_frames = mrc.data.shape[0]
            tiles = {p["ZValue"]: mrc.data[p["ZValue"]].astype(np.float32)
                     for p in pieces if p["ZValue"] < n_frames}

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
                    raw = _pcc(_norm(ref_strip), _norm(mov_strip),
                               upsample_factor=10, normalization='phase')
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
    def show(self, annotate=True, rotated=True):
        import matplotlib.pyplot as plt
        sec = self.section

        if self.section_pieces and sec in self.section_pieces:
            key = self._display_key()                 # refines if needed
            if annotate and any("estimated_stage_position" not in p
                                for p in self.section_pieces[sec]):
                self.tile_alignment_shift()
            img = self._assemble_one_montage(*self._img_hw, self._feather_px, key)
            tiles = self.section_pieces[sec] if annotate else None
        elif self.mrc_image is not None:
            img, key, tiles = self.mrc_image, None, None
        else:
            raise ValueError("Nothing loaded - call load_mrc_montage() or "
                             "load_mrc_single() first.")

        max_px = 2000
        h, w = img.shape[:2]
        ds = max(1, max(h, w) // max_px)
        disp = img[::ds, ::ds] if ds > 1 else img

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(disp, cmap="gray", origin="upper", aspect="equal",
                  vmin=0.0, vmax=1.0, extent=[-0.5, w - 0.5, h - 0.5, -0.5])
        ax.set_title(key or "single image")

        if tiles is not None:
            img_h, img_w = self._img_hw
            field = "ShiftedRot" if rotated else "ShiftedPlain"
            for p in tiles:
                c = self._get_coords(p, key)             # tile origin (px) for THIS key
                cx = float(c[0]) + img_w / 2.0           # -> tile centre (px), label pos
                cy = float(c[1]) + img_h / 2.0
                esp = p.get("estimated_stage_position")
                val = esp[field] if esp and esp.get(field) is not None else None
                label = f"{val[0]:.1f}, {val[1]:.1f}" if val is not None else "n/a"
                ax.text(cx, cy, label, ha="center", va="center", fontsize=7,
                        color="yellow", bbox=dict(boxstyle="round,pad=0.2",
                                                  fc="black", ec="none", alpha=0.5))
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def create_montage(self, single=False):
        if single:
            self.load_mrc_single()
            self.show()
        else:
            self.load_mrc_montage()
            if self.refine_alignment:
                self.refine_tile_alignment()
            self.tile_alignment_shift()
            self.show()

    # ------------------------------------------------------------------ #
    # Picker
    # ------------------------------------------------------------------ #
    def pick_stage_positions(self, max_px=2000):
        import matplotlib.pyplot as plt

        sec = self.section
        pieces = self.section_pieces[sec]
        key = self._display_key()

        if any("estimated_stage_position" not in p for p in pieces):
            self.tile_alignment_shift()

        img = self._assemble_one_montage(*self._img_hw, self._feather_px, key)
        h, w = img.shape[:2]
        ds = max(1, max(h, w) // max_px)
        disp = img[::ds, ::ds] if ds > 1 else img
        img_h, img_w = self._img_hw
        pix_um = self.pixel_spacing_um

        tiles = []
        for p in pieces:
            c = self._get_coords(p, key)
            cx = float(c[0]) + img_w / 2.0
            cy = float(c[1]) + img_h / 2.0

            esp = p.get("estimated_stage_position")
            sp = p.get("StagePosition")
            sp = ((float(sp[0]), float(sp[1]))
                if isinstance(sp, (list, tuple)) and len(sp) >= 2 else None)

            anchor_plain = (esp.get("ShiftedPlain") if esp else None) or sp
            anchor_rot   = (esp.get("ShiftedRot")   if esp else None) or sp
            if anchor_plain is None or anchor_rot is None:
                continue

            r = p.get("RotationAngle", 0.0)
            th = np.deg2rad(float(r[0]) if isinstance(r, (list, tuple)) else float(r))

            stage_z = self._tile_stage_z(p)

            tiles.append({"z": p.get("ZValue"), "stage_z": stage_z,
                        "cx": cx, "cy": cy,
                        "ax_p": anchor_plain[0], "ay_p": anchor_plain[1],
                        "ax_r": anchor_rot[0],   "ay_r": anchor_rot[1],
                        "cos": np.cos(th), "sin": np.sin(th)})

        def tile_for(px, py):
            inside = [t for t in tiles
                    if t["cx"] - img_w / 2 <= px < t["cx"] + img_w / 2
                    and t["cy"] - img_h / 2 <= py < t["cy"] + img_h / 2]
            pool = inside if inside else tiles
            return min(pool, key=lambda t: (px - t["cx"]) ** 2 + (py - t["cy"]) ** 2)

        picks = []
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(disp, cmap="gray", origin="upper", aspect="equal",
                vmin=0.0, vmax=1.0, extent=[-0.5, w - 0.5, h - 0.5, -0.5])
        ax.set_title("Left-click to pick stage positions; close window when done")
        ax.axis("off")

        def on_click(event):
            if event.button != 1 or event.inaxes is not ax or event.xdata is None:
                return
            px, py = event.xdata, event.ydata
            t = tile_for(px, py)

            dx_um = (px - t["cx"]) * pix_um
            dy_um = (py - t["cy"]) * pix_um

            sx_p = t["ax_p"] + dx_um
            sy_p = t["ay_p"] + dy_um
            sx_r = t["ax_r"] + (t["cos"] * dx_um - t["sin"] * dy_um)
            sy_r = t["ay_r"] + (t["sin"] * dx_um + t["cos"] * dy_um)

            picks.append({"px": px, "py": py, "z": t["z"],
                        "stage_z": t["stage_z"],
                        "plain": (sx_p, sy_p), "rot": (sx_r, sy_r)})
            print(f"point {len(picks)}: tile z={t['z']}  "
                f"pixel=({px:.0f}, {py:.0f})  "
                f"plain=({sx_p:.3f}, {sy_p:.3f}) um  "
                f"rot=({sx_r:.3f}, {sy_r:.3f}) um  "
                f"stageZ={t['stage_z']}")

            ax.plot(px, py, "+", color="cyan", markersize=12, markeredgewidth=1.5)
            ax.text(px + 8, py - 8, str(len(picks)), color="cyan",
                    fontsize=9, fontweight="bold")
            ax.annotate(f"P {sx_p:.1f}, {sy_p:.1f}", (px, py),
                        xytext=(px + 8, py + 16), color="yellow", fontsize=7)
            ax.annotate(f"R {sx_r:.1f}, {sy_r:.1f}", (px, py),
                        xytext=(px + 8, py + 30), color="red", fontsize=7)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", on_click)
        plt.tight_layout()
        plt.show()
        return picks

if __name__ == "__main__":
    mrc = MRCReader(
        path='/Users/sophia.betzler/Desktop/12-chief-dog_montage_20260616-07-47-11.mrc',
        coord_key="PieceCoordinates", refine_alignment=False, section=0)
    tile_summary = mrc.create_montage()
    print(tile_summary)
    picks = mrc.pick_stage_positions()