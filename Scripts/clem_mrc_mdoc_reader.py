from email.mime import base
import re
import os
import mrcfile
import tifffile
import numpy as np
from pathlib import Path
from clem_dataclasses import *

class MRCReader:

    COORD_KEYS = ("RefinedPieceCoordinates", "AlignedPieceCoordsVS",
                  "AlignedPieceCoords", "PieceCoordinates")


    MONTAGE_FLIP_X = False
    MONTAGE_FLIP_Y = True

    def __init__(self, coord_key, section=0):
        self.coord_key = coord_key
        self.section = section

    # ------------------------------------------------------------------ #
    # Small helpers for stage coordinate readout
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_scalar(value):
        """Best-effort float from an mdoc value that may be scalar or list."""
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            value = value[0] if len(value) else None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _available_coord_keys(self, pieces):
        """Coordinate fields present for every piece (used in error text)."""
        return [k for k in self.COORD_KEYS
                if pieces and all(p.get(k) is not None for p in pieces)]

    @staticmethod
    def _get_coords(piece, coord_key):
        v = piece.get(coord_key)
        if v is None:
            raise KeyError(
                f"Tile ZValue={piece.get('ZValue')} has no '{coord_key}'.")
        return v
    
    @staticmethod
    def _piece_xy_position(piece, field):
        c = piece.get(field)
        if c is None:
            return None
        return {"piece_x_px": float(c[0]), "piece_y_px": float(c[1])}
    
    @staticmethod
    def _piece_stage_xy_position(piece):
        stage_position = piece.get("StagePosition")
        if isinstance(stage_position, (list, tuple)) and len(stage_position) >= 2:
            return {"stage_x_um": float(stage_position[0]), "stage_y_um": float(stage_position[1])} 
        raise ValueError("No stage coordinates available for the montage piece.")
    
    @staticmethod
    def _piece_stage_z_position(piece, global_stage_z_um=None):
        stage_position = piece.get("StagePosition")

        if isinstance(stage_position, (list, tuple)) and len(stage_position) >= 3:
            return float(stage_position[2])

        for key in ("StageZ", "Z"):
            value = piece.get(key)
            if value is not None:
                if isinstance(value, (list, tuple)):
                    value = value[0]
                return float(value)

        return global_stage_z_um

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
    
    @staticmethod
    def _flip_for_display(arr, flip_x=False, flip_y=False):
        if flip_x:
            arr = np.fliplr(arr)
        if flip_y:
            arr = np.flipud(arr)
        return arr

    # ------------------------------------------------------------------ #
    # File / coordinate-field discovery
    # ------------------------------------------------------------------ #
    def _find_latest_mrc_dataclass(self, site_data):
        candidates = [m for m in site_data.mrcs.values() if m is not None]
        if not candidates:
            return None
        def _recency(m):
            ts = getattr(m, "timestamp", None)
            if ts:
                return (1, ts)
            try:
                return (0, os.path.getmtime(m.mrc_path))
            except (OSError, TypeError):
                return (0, 0.0)
        return max(candidates, key=_recency)

    @staticmethod
    def _site_folder(site_data):
        """Site folder as a Path.  Accepts a SiteDataSummary (whose .path is a
        str), a Path, or a plain string, so the finders below work no matter
        which of the three a caller hands them."""
        folder = getattr(site_data, "path", site_data)
        if folder is None:
            raise ValueError("No site folder: site_data.path is not set.")
        folder = Path(os.fspath(folder))
        if not folder.is_dir():
            raise NotADirectoryError(f"Site folder does not exist: {folder}")
        return folder

    def _find_latest_montage_mrc(self, site_data):
        folder = self._site_folder(site_data)
        matches = [p for p in folder.glob("*montage*.mrc") if p.is_file()]
        if not matches:                          # fall back to any .mrc / .rec
            matches = [p for p in (*folder.glob("*.mrc"), *folder.glob("*.rec"))
                       if p.is_file()]
        if not matches:
            raise FileNotFoundError(f"No montage .mrc found in {folder}")
        return max(matches, key=lambda p: p.stat().st_mtime)

    def _find_latest_ome_tiff(self, site_data):
        folder = self._site_folder(site_data)
        matches = [p for p in (*folder.glob("*.ome.tif"), *folder.glob("*.ome.tiff"))
                   if p.is_file()]
        if not matches:
            raise FileNotFoundError(f"No OME-TIFF found in {folder}")
        return max(matches, key=lambda p: p.stat().st_mtime)

    def _find_latest_czi(self, site_data):
        folder = self._site_folder(site_data)
        matches = [p for p in folder.glob("*.czi") if p.is_file()]
        return max(matches, key=lambda p: p.stat().st_mtime) if matches else None

    def _fit_latest_transfer(self, site_data):
        folders = [Path(site_data.path).parent / "transforms", Path(site_data.path) / "transforms"]
        patterns = ("transform_*.haml", "transform_*.yml", "transform_*.csv", "*.txt")
        matches = []
        for folder in folders:
            if folder.is_dir():
                for pat in patterns:
                    matches += [p for p in folder.glob(pat) if p.is_file()]
        if not matches:
            raise FileNotFoundError(f"No transform found.")
        return max(matches, key=lambda p:p.stat().st_mtime)

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
                "No edited pieces were parsed from the mdoc (0 [ZValue] blocks). "
                "The montage cannot be built - check the mdoc contents.")
        missing = [p.get("ZValue") for p in pieces
                   if p.get(self.coord_key) is None]
        if missing:
            shown = missing[:10]
            more = f" ... (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            raise KeyError(
                f"The field '{self.coord_key}' is not present for {len(missing)} tile(s) (ZValues: {shown}{more}). "
                f"Available, fully-populated fields: {self._available_coord_keys(pieces)}")

    def _build_section_pieces(self, pieces):
        section_map = {}
        for piece in pieces:
            pc = self._get_coords(piece, self.coord_key)
            sec = int(pc[2]) if isinstance(pc, (list, tuple)) and len(pc) >= 3 else 0
            section_map.setdefault(sec, []).append(piece)
        if not section_map:
            raise ValueError("No [ZValue] edited pieces found in the mdoc.")
        self.section_pieces = section_map
        return section_map

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #

    # NOTE: create_site_data_class() was removed -- it was never called
    # (ExecutiveControls.run_acquire_position_montages builds SiteDataSummary
    # directly) and referenced a _get_site_folder() method that does not exist.
    # The _find_latest_* helpers above are the reusable part; call them with a
    # SiteDataSummary, a Path, or a folder string.

    def load_mrc_into_data_class(self, site_data, mrc_path):
        mrc_montage = site_data.populate_mrc(self, mrc_path)   # stores into site_data.mrcs[label]
        site_data.registration = None
        return mrc_montage

    def load_tiff_into_data_class(self, site_data, ome_path):
        site_data.populate_tiff(self, ome_path)         
        site_data.registration = None
        return site_data

    def load_mrc_montage_data(self, mrc_path):
        return self.build_montage_summary(mrc_path)

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
        c, z, y, x = tiff_stack.shape
        return TiffSummary(
                ome_path=os.path.abspath(os.fspath(ome_path)),
                stack_czyx=tiff_stack, num_channels=c, num_z_slices=z, stack_height=y,
                stack_width=x,info=tiff_info,)
    
    @staticmethod
    def read_tiff_pixel_spacing_um(path):
        """XY pixel spacing (um/px) for an OME-TIFF or ImageJ/plain TIFF,
        or None if it can't be determined."""
        import re
        _TO_UM = {
            "m": 1e6, "meter": 1e6, "metre": 1e6,
            "cm": 1e4, "centimeter": 1e4, "centimetre": 1e4,
            "mm": 1e3, "millimeter": 1e3, "millimetre": 1e3,
            "um": 1.0, "\u00b5m": 1.0, "micron": 1.0, "microns": 1.0,
            "micrometer": 1.0, "micrometre": 1.0,
            "nm": 1e-3, "nanometer": 1e-3, "nanometre": 1e-3,
            "inch": 25400.0, "in": 25400.0,
        }

        def to_um(value, unit):
            if value is None or unit is None:
                return None
            factor = _TO_UM.get(str(unit).strip().lower())
            return float(value) * factor if factor is not None else None

        with tifffile.TiffFile(path) as tf:
            ome = tf.ome_metadata
            if ome:
                mx = re.search(r'PhysicalSizeX\s*=\s*"([^"]+)"', ome)
                mu = re.search(r'PhysicalSizeXUnit\s*=\s*"([^"]+)"', ome)
                if mx:
                    unit = mu.group(1) if mu else "um"
                    result = to_um(float(mx.group(1)), unit)
                    if result:
                        return result

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
                    size_in_unit = 1.0 / px_per_unit
                    if unit is None:
                        ru = page.tags.get("ResolutionUnit")
                        ru_val = int(ru.value) if ru is not None else 1
                        unit = {2: "inch", 3: "cm"}.get(ru_val)
                    result = to_um(size_in_unit, unit)
                    if result:
                        return result
        return None
    
    def load_czi(self, czi_path):
        czi_path = os.fspath(czi_path)
        arr, axes = None, None
        try:
            from aicspylibczi import CziFile
            czi = CziFile(czi_path)
            arr, shp = czi.read_image()
            axes = "".join(d for d, _ in shp).upper()
        except Exception:
            arr = None
        if arr is None:
            from czifile import CziFile as _GohlkeCzi   # ImportError if neither present
            with _GohlkeCzi(czi_path) as czi:
                arr = np.asarray(czi.asarray())
                axes = "".join(czi.axes).upper()

        arr = np.asarray(arr)
        for ax in list(axes):                       # collapse anything but C,Z,Y,X
            if ax not in "CZYX":
                idx = axes.index(ax)
                take = arr.shape[idx] // 2 if arr.shape[idx] > 1 else 0
                arr = arr.take(take, axis=idx)
                axes = axes.replace(ax, "", 1)
        for ax in ("C", "Z"):
            if ax not in axes:
                arr = arr[np.newaxis]; axes = ax + axes
        arr = np.transpose(arr, [axes.index(a) for a in "CZYX"])

        normed = np.zeros(arr.shape, dtype=np.float32)
        for c in range(arr.shape[0]):
            normed[c] = self._normalize_image(arr[c])
        info = f"czi shape={tuple(arr.shape)}  ({os.path.basename(czi_path)})"
        return normed, info

    @staticmethod
    def read_czi_pixel_spacing_um(path):
        import re
        with open(os.fspath(path), "rb") as fh:
            raw = fh.read()
        s = raw.find(b"<ImageDocument")
        e = raw.find(b"</ImageDocument>")
        xml = (raw[s:e] if (s != -1 and e != -1) else raw).decode("utf-8", "replace")
        m = re.search(r'<Distance Id="X">\s*<Value>([^<]+)</Value>', xml)
        return float(m.group(1)) * 1e6 if m else None   # metres -> micrometres
    
    def load_czi_into_data_class(self, site_data, czi_path):
        site_data.populate_czi(self, czi_path)
        return site_data


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

        print(f"[INFO] Parsed {len(pieces)} edited pieces, {len(mont_sections)} mont section(s).")
        return global_info, pieces, mont_sections

    # ------------------------------------------------------------------ #
    # Cropping tools
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce_scalar(value, name):
        if value is None:
            raise ValueError(f"{name} must not be None")
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != 1:
                raise ValueError(f"{name} must be a scalar-like value")
            value = value[0]
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric, got {type(value).__name__}") from exc


    def _crop_centered_at_pixel_coord(self, full, px, py, pixel_spacing_um, fov_um, fill=0.0):
        cw = self._fov_in_px(pixel_spacing_um, fov_um)
        H, W = full.shape
        px = self._coerce_scalar(px, "px")
        py = self._coerce_scalar(py, "py")
        half = cw // 2
        x0, y0 = int(round(px)) - half, int(round(py)) - half
        out = np.full((cw, cw), fill, dtype=np.float32)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W, x0 + cw), min(H, y0 + cw)
        if sx1 > sx0 and sy1 > sy0:
            out[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = full[sy0:sy1, sx0:sx1]

        return out

    def _fov_in_px(self, pixel_spacing_um, fov_um):
        fov_um = self._coerce_scalar(fov_um, "fov_um")
        spacing_um = self._coerce_scalar(pixel_spacing_um, "pixel_spacing_um")
        if spacing_um <= 0:
            raise ValueError("pixel_spacing_um must be positive")
        cw = max(2, int(round(fov_um / spacing_um)))
        return cw
        
    def write_mrc_crops(self, mrc_dataclass, fov_um, pixel_spacing_um, output_root=None,
                    label='crop', skip_pick_id=None):

        def correct_image(crop):
            sat = crop >= 1.0
            valid = crop[~sat]
            crop[sat] = float(valid.mean()) if valid.size else 0.0
            return crop

        if output_root is None:
            output_root = os.path.join(os.path.dirname(mrc_dataclass.mrc_path), "picks", "crop")
        os.makedirs(os.path.dirname(output_root) or ".", exist_ok=True)

        written = []
        for pick in mrc_dataclass.picks:
            if skip_pick_id is not None and pick.pick_id == skip_pick_id:
                written.append(None); continue
            H, W = mrc_dataclass.image.shape[:2]
            px, py = pick.image_coord_x, pick.image_coord_y
            if mrc_dataclass.flip_x: px = W - 1 - px
            if mrc_dataclass.flip_y: py = H - 1 - py
            crop = self._crop_centered_at_pixel_coord(mrc_dataclass.image, px, py, pixel_spacing_um, fov_um)
            crop = correct_image(crop)
            out = f"{output_root}_{label}_{pick.pick_id}.mrc"
            with mrcfile.new(out, overwrite=True) as mrc:
                mrc.set_data(crop.astype(np.float32))
                mrc.voxel_size = pixel_spacing_um * 10000
                mrc.update_header_from_data()
            written.append(out)
        return written

    def write_multichannel_crops(self, mrc_dataclass, warp_slice, n_channels, n_z,
                                fov_um, output_root, pixel_spacing_um, prefix=""):
        cw = self._fov_in_px(pixel_spacing_um, fov_um)
        picks = mrc_dataclass.picks
        stacks = [np.zeros((n_z, 1 + n_channels, cw, cw), np.float32) for _ in picks]
        def to_true(px, py):
            H, W = mrc_dataclass.image.shape[:2]
            if mrc_dataclass.flip_x: px = W - 1 - px
            if mrc_dataclass.flip_y: py = H - 1 - py
            return px, py
        for pi, pick in enumerate(picks):                 # channel 0 = TEM montage
            px, py = to_true(pick.image_coord_x, pick.image_coord_y)
            crop = self._crop_centered_at_pixel_coord(mrc_dataclass.image, px, py, pixel_spacing_um, fov_um)
            for z in range(n_z):
                stacks[pi][z, 0] = crop
        for c in range(n_channels):                       # channels = warped FL
            for z in range(n_z):
                full = warp_slice(c, z)
                for pi, pick in enumerate(picks):
                    px, py = to_true(pick.image_coord_x, pick.image_coord_y)
                    stacks[pi][z, c + 1] = self._crop_centered_at_pixel_coord(full, px, py, pixel_spacing_um, fov_um)
        res = (1.0 / pixel_spacing_um) if pixel_spacing_um > 0 else 1.0
        labels = (["TEM"] + [f"Ch{c}" for c in range(n_channels)]) * n_z
        os.makedirs(os.path.dirname(output_root) or ".", exist_ok=True)

        written = []
        for pi, pick in enumerate(picks):
            out = f"{output_root}{prefix}_{pick.pick_id}.tif"
            tifffile.imwrite(out, stacks[pi], imagej=True, resolution=(res, res),
                            metadata={"axes": "ZCYX", "unit": "um", "Labels": labels})
            written.append(os.path.basename(out))
        return written


    def write_fov_crops(self, site_data, warp_slice, n_channels, n_z,
                        fov_um, output_root):
        tif = self.write_multichannel_crops(
            site_data, warp_slice, n_channels, n_z, fov_um, output_root)
        mrc_paths = self.write_mrc_crops(site_data, fov_um, output_root)
        for i, pick in enumerate(site_data.picks):
            if i < len(mrc_paths) and mrc_paths[i] is not None:
                pick.view_crop_path = mrc_paths[i]
        return {"tif": tif, "mrc": mrc_paths}
    # ------------------------------------------------------------------ #
    # Montage assembly
    # ------------------------------------------------------------------ #

    def _assemble_montage(self, mdoc_filepath, img_h, img_w, feather_px, key, pieces=None, status_cb=None):
        pieces = pieces or self.section_pieces[self.section]

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
                return self._normalize_image(data.astype(np.float32)), 0.0, 0.0
            if data.ndim != 3:
                raise ValueError(f"Unsupported MRC shape {data.shape}")
            n_frames = data.shape[0]
            pieces_edited = {z: data[z].astype(np.float64)
                     for z in z_indices if z < n_frames}

        for piece, x, y in coords:
            z = piece["ZValue"]
            if z not in pieces_edited:
                continue
            dx = int(round(x - min_x)); dy = int(round(y - min_y))
            canvas[dy:dy + img_h, dx:dx + img_w] += pieces_edited[z] * wmap
            weights[dy:dy + img_h, dx:dx + img_w] += wmap

        valid = weights > 0
        canvas[valid] /= weights[valid]
        return (self._normalize_image(canvas.astype(np.float32)), float(min_x), float(min_y))


    # ------------------------------------------------------------------ #
    # Create Dictionary Output Summarizing aligned Montage
    # ------------------------------------------------------------------ #

    def build_montage_summary(self, mrc_filepath, site_id=None):
        mdoc_path = self._find_mdoc_path(mrc_filepath=mrc_filepath)
        if mdoc_path is None:
            raise FileNotFoundError(f"No .mdoc found next to {os.path.basename(mrc_filepath)}.")

        global_info, pieces_all, mont_sections = self.parse_mdoc(mdoc_path)
        global_stage_z = self._extract_scalar(global_info.get("StageZ", global_info.get("Z")))


        self._validate_coord_key(pieces_all)

        ps_ang = global_info.get("PixelSpacing", 10000.0)
        ps_ang = float(ps_ang[0]) if isinstance(ps_ang, (list, tuple)) else float(ps_ang)
        pixel_spacing_um = ps_ang / 10000.0

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

        self._build_section_pieces(pieces_all)


        alignment = self.coord_key
        key = self.coord_key

        pieces = self.section_pieces[self.section]

        theta = np.deg2rad(self._section_rotation_angle(pieces))
        img, min_x, min_y = self._assemble_montage(mrc_filepath, img_h, img_w, feather_px, key)

        tiles = []
        for p in pieces:
            px = self._piece_xy_position(p, key)
            st = self._piece_stage_xy_position(p)
            stage_z = self._piece_stage_z_position(p, global_stage_z_um=global_stage_z,)
            tiles.append(Tile(z_index=p.get("ZValue"), 
                            piece_z_stage_um=stage_z,
                            piece_x_px=(px or {}).get("piece_x_px"),
                            piece_y_px=(px or {}).get("piece_y_px"),
                            piece_x_stage_um=(st or {}).get("stage_x_um"),
                            piece_y_stage_um=(st or {}).get("stage_y_um")))

        base = os.path.splitext(os.path.basename(mrc_filepath))[0]
        montage_id = f"{site_id}_montage_{base}" if site_id else f"montage_{base}"

        metadata = MontageMetadata(
                                    pixel_spacing_um=pixel_spacing_um,
                                    image_width_px=img_w,
                                    image_height_px=img_h,
                                    piece_spacing_x_px=ps_x,
                                    piece_spacing_y_px=ps_y,
                                    stage_z_um=(float(stage_z) if stage_z is not None else float(global_stage_z) if global_stage_z is not None else 0.0),
                                    magnification=self._extract_scalar(global_info.get("Magnification")),
                                    rotation_deg=float(np.rad2deg(theta)),
                                    raw=dict(global_info),
                                )

        return MRCSummary(
            mrc_path=os.fspath(mrc_filepath),
            montage_id=montage_id,
            metadata = metadata,
            image=img, image_height=img_h, image_width=img_w,
            pixel_spacing_um=pixel_spacing_um,
            feather_pixels=feather_px,
            section=self.section, alignment=alignment, coord_field=key,
            rotation_deg=float(np.rad2deg(theta)),
            min_x_pixels=min_x, min_y_pixels=min_y,
            stage_z_um=float(global_stage_z) if global_stage_z is not None else None,
            tiles=tiles,
            flip_x=self.MONTAGE_FLIP_X, flip_y=self.MONTAGE_FLIP_Y,
        )   


    def run_montage_loader_and_create_summary(self, mrc_filepath):
        if not self.section_pieces:
            self.load_mrc_montage(mrc_filepath=mrc_filepath) 
        self.show(mrc_filepath=mrc_filepath, contrast_percentiles=(1.0, 99.))
        return self.build_montage_summary(mrc_filepath = mrc_filepath)