import csv
import os
import numpy as np
import math
from skimage.transform import estimate_transform, warp, ProjectiveTransform
from datetime import datetime


class CLEMCorrelator:
    REQUIRED_POINT_PAIRS = {
                        "euclidean": 2,
                        "similarity": 2,
                        "affine": 3,
                        "projective": 4,
                    }

    def __init__(self, mrc_reader):
        self.mrc_reader = mrc_reader
        self.last_tform = None

    # ---------------------------------------------------------------------------
    # Helper function
    # ---------------------------------------------------------------------------

    def _resolve_flip(self, flip_value, flip_name):
        if flip_name == "flip_y":
            return True
        if flip_name == "flip_x":
            return bool(getattr(self.mrc_reader, "MONTAGE_FLIP_X", False))
        if flip_value is not None:
            return bool(flip_value)
        if self.mrc_reader is None:
            return False
        return False

    @staticmethod
    def _get_tiff_slice(tiff_stack, c, z, flip_x=False, flip_y=False):
        img = tiff_stack[c, z]
        if flip_x:
            img = np.fliplr(img)
        if flip_y:
            img = np.flipud(img)
        return img
    
    # ---------------------------------------------------------------------------
    # Functions which handel the transformation between the MRC and TIFF coordinate systems
    # ---------------------------------------------------------------------------

    @staticmethod
    def _build_reflection_matrix(tiff_shape, flip_x=False, flip_y=False):
        tiff_h, tiff_w = tiff_shape[-2:]
        F = np.eye(3, dtype=float)

        if flip_x:
            Fx = np.array([
                            [-1.0, 0.0, tiff_w - 1.0],
                            [ 0.0, 1.0,            0.0],
                            [ 0.0, 0.0,            1.0],
                        ])
            F = Fx @ F

        if flip_y:
            Fy = np.array([
                            [1.0,  0.0,             0.0],
                            [0.0, -1.0, tiff_h - 1.0],
                            [0.0,  0.0,             1.0],
                        ])
            F = Fy @ F

        return F
    
    def _apply_matrix(self, M, pts):
        """Apply a 3x3 homogeneous matrix to an (N, 2) array of points."""
        pts = np.asarray(pts, dtype=float)
        hom = np.column_stack([pts, np.ones(len(pts), dtype=float)])
        out = (M @ hom.T).T
        return out[:, :2] / out[:, 2, None]

    @staticmethod
    def _image_center(tiff_shape):
        if tiff_shape is None:
            return None
        h, w = tiff_shape[-2:]
        return np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=float)

    def _rescale_matrix(self, M, new_scale, src_anchor, dst_anchor=None):
        if new_scale <= 0:
            raise ValueError("scale must be positive.")
        M = np.asarray(M, dtype=float)
        A, t = M[:2, :2], M[:2, 2]
        stored_scale = float(np.hypot(A[0, 0], A[1, 0]))
        if stored_scale <= 0:
            raise ValueError("Transform has no positive scale to replace.")
        R = A / stored_scale
        A_new = new_scale * R
        src_anchor = np.asarray(src_anchor, dtype=float)
        if dst_anchor is None:
            dst_anchor = A @ src_anchor + t
        out = np.eye(3, dtype=float)
        out[:2, :2] = A_new
        out[:2, 2] = np.asarray(dst_anchor, dtype=float) - A_new @ src_anchor
        return out, stored_scale
    
    def fit_tiff_to_mrc(self, point_pairs, transform_type, predefined=None, predefined_transform=None,
                        tiff_shape=None, fixed_scale=None, scale_tolerance=0.05, flip_x=None, flip_y=None,
                        initial_transform=None):

        predefined_transform = predefined_transform or predefined

        complete = [(p["tiff"], p["mrc"]) for p in point_pairs if "tiff" in p and "mrc" in p]
        required = 0 if predefined_transform is not None else self.REQUIRED_POINT_PAIRS.get(transform_type, 3)
        if len(complete) < required:
            raise ValueError(f"{transform_type.capitalize()} needs >= {required} pairs (you have {len(complete)}).")

        src = np.asarray([p[0] for p in complete], dtype=float).reshape(-1, 2)
        dst = np.asarray([p[1] for p in complete], dtype=float).reshape(-1, 2)

        flip_x = self._resolve_flip(flip_x, "flip_x")
        flip_y = self._resolve_flip(flip_y, "flip_y")
        F = self._build_reflection_matrix(tiff_shape, flip_x=flip_x, flip_y=flip_y)
        reflected = self._apply_matrix(F, src)          # fit everyone in the reflected frame

        base, extra = self._estimate_base(transform_type, reflected, dst,fixed_scale=fixed_scale, scale_tolerance=scale_tolerance,
            predefined_transform=predefined_transform, initial_transform=initial_transform, tiff_shape=tiff_shape,)

        tform = ProjectiveTransform(matrix=base @ F)
        fit_info = self._fit_diagnostics(tform, src, dst)
        fit_info.update({"flip_x": bool(flip_x), "flip_y": bool(flip_y)})
        fit_info.update(extra)
        if "bounded_scale" in extra:                    # keep the informative summary line
            fit_info["text"] += (
                f"  expected={extra['expected_scale']:.4f}"
                f"  allowed=[{extra['scale_min']:.4f}, {extra['scale_max']:.4f}]"
                f"  free={extra['unconstrained_scale']:.4f}"
                f"  used={extra['bounded_scale']:.4f}"
            )

        self.last_tform = tform
        return tform, fit_info, len(complete)

    def _estimate_base(self, transform_type, reflected, dst, fixed_scale=None, scale_tolerance=0.05,
                       predefined_transform=None, initial_transform=None, tiff_shape=None):

        if predefined_transform is not None:
            base = np.asarray(predefined_transform.params, dtype=float)
            if fixed_scale is None:
                return base, {}
            if len(reflected):
                src_anchor, dst_anchor = reflected.mean(axis=0), dst.mean(axis=0)
            else:
                src_anchor = self._image_center(tiff_shape)
                if src_anchor is None:
                    src_anchor, dst_anchor = np.zeros(2), None   # fall back to origin
                else:
                    dst_anchor = None                            # keep centre's mapping
            base, stored_scale = self._rescale_matrix(base, fixed_scale, src_anchor, dst_anchor)
            extra = { 
                        "expected_scale": float(fixed_scale),
                        "stored_scale": stored_scale,
                        "applied_scale": float(fixed_scale),
                    }
            return base, extra

        if transform_type not in {"similarity", "euclidean"}:
            base = np.asarray(estimate_transform(transform_type, reflected, dst).params, dtype=float)
            return base, {}

        free = np.asarray(estimate_transform("similarity", reflected, dst).params, dtype=float)
        if fixed_scale is None:
            return free, {}

        return self._bound_similarity_scale(free, reflected, dst, fixed_scale, scale_tolerance, initial_transform)
    
    def _bound_similarity_scale(self, free_params, reflected, dst, fixed_scale,
                                scale_tolerance, initial_transform=None):
        """Clamp a free similarity fit's scale into
        [fixed_scale*(1-tol), fixed_scale*(1+tol)], recomputing the translation
        so the transform still maps the reflected-source centroid onto the dst
        centroid."""
        if fixed_scale <= 0:
            raise ValueError("fixed_scale must be positive.")
        if not 0 <= scale_tolerance < 1:
            raise ValueError("scale_tolerance must be between 0 and 1.")

        free_A = np.asarray(free_params[:2, :2], dtype=float)

        if initial_transform is not None:
            init_A = np.asarray(initial_transform.params, dtype=float)[:2, :2]
            init_scale = float(np.hypot(init_A[0, 0], init_A[1, 0]))
            if init_scale > 0:
                free_A = init_A
                unconstrained_scale = init_scale
            else:
                unconstrained_scale = float(np.hypot(free_A[0, 0], free_A[1, 0]))
        else:
            unconstrained_scale = float(np.hypot(free_A[0, 0], free_A[1, 0]))

        if unconstrained_scale <= 0:
            raise ValueError("Could not determine a positive similarity scale.")

        scale_min = fixed_scale * (1.0 - scale_tolerance)
        scale_max = fixed_scale * (1.0 + scale_tolerance)
        fitted_scale = float(np.clip(unconstrained_scale, scale_min, scale_max))

        R = free_A / unconstrained_scale
        translation = dst.mean(axis=0) - fitted_scale * (R @ reflected.mean(axis=0))

        base = np.eye(3, dtype=float)
        base[:2, :2] = fitted_scale * R
        base[:2, 2] = translation

        extra = {
            "expected_scale": float(fixed_scale),
            "unconstrained_scale": unconstrained_scale,
            "bounded_scale": fitted_scale,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "scale_tolerance": scale_tolerance,
            "scale_was_clamped": not np.isclose(fitted_scale, unconstrained_scale),
        }
        return base, extra

    def warp_channels_to_mrc(self, tiff_stack, tform, mrc_shape, flip_x=None, flip_y=None, status_cb=None,):

        mrc_h, mrc_w = mrc_shape
        C, Z = tiff_stack.shape[:2]
        flip_x = self._resolve_flip(flip_x, "flip_x")
        flip_y = self._resolve_flip(flip_y, "flip_y")

        warped_channels = []

        for c in range(C):
            acc = None

            for z in range(Z):
                if status_cb is not None:
                    status_cb(f"Warping channel {c + 1}/{C}, z {z + 1}/{Z}...")

                img = self._get_tiff_slice(tiff_stack, c, z, flip_x, flip_y)

                warped = warp(img, tform.inverse, output_shape=(mrc_h, mrc_w), order=1, preserve_range=True, mode="constant", cval=0,).astype(np.float32)

                if acc is None:
                    acc = warped
                else:
                    np.maximum(acc, warped, out=acc)

            warped_channels.append(acc)

        self.last_tform = tform
        return warped_channels


    def run_apply_transform(self, point_pairs, transform_type, tiff_stack, mrc_shape, flip_x=None, flip_y=None, status_cb=None,
                        initial_transform=None, predefined=None, predefined_transform=None, fixed_scale=None,
                        scale_tolerance=0.05, tiff_shape=None, auto_save=True, save_dir=None, save_format="auto"):

        tform, fit_info, n_pairs = self.fit_tiff_to_mrc(point_pairs,transform_type,predefined=predefined,
            predefined_transform=predefined_transform, tiff_shape=tiff_shape or tiff_stack.shape,
            fixed_scale=fixed_scale, scale_tolerance=scale_tolerance, flip_x=flip_x, flip_y=flip_y,
            initial_transform=initial_transform,)

        warped_channels = self.warp_channels_to_mrc(tiff_stack=tiff_stack, tform=tform, mrc_shape=mrc_shape, flip_x=flip_x, flip_y=flip_y,
                                                    status_cb=status_cb,)

        record = self._build_transform_record(
            tform, transform_type, fit_info=fit_info, n_pairs=n_pairs,
            flip_x=flip_x, flip_y=flip_y, mrc_shape=mrc_shape,
            tiff_shape=tiff_shape or tiff_stack.shape,
            fixed_scale=fixed_scale, scale_tolerance=scale_tolerance,
        )

        result = {
                    "transform": tform,
                    "fit_info": fit_info,
                    "n_pairs": n_pairs,
                    "warped_channels": warped_channels,
                    "record": record,
                }
        if auto_save:
            try:
                result["saved_path"] = self.save_transform(record, save_dir=save_dir, fmt=save_format)
            except Exception as exc:            # never let a save failure kill the warp
                result["saved_path"] = None
                result["save_error"] = str(exc)

        return result
    
    
    # ---------------------------------------------------------------------------
    # Diagnostics functions
    # ---------------------------------------------------------------------------

    def _check_scale(self, fit_info, expected_scale, tolerance=0.05):
        sx = fit_info["scale_x"]
        sy = fit_info["scale_y"]

        lo = expected_scale * (1.0 - tolerance)
        hi = expected_scale * (1.0 + tolerance)

        ok_x = lo <= sx <= hi
        ok_y = lo <= sy <= hi

        return {
                    "expected_scale": expected_scale,
                    "tolerance": tolerance,
                    "scale_min": lo,
                    "scale_max": hi,
                    "ok": ok_x and ok_y,
                    "ok_x": ok_x,
                    "ok_y": ok_y,
                    "message": (f"expected scale={expected_scale:.4f} allowed=[{lo:.4f}, {hi:.4f}] fit x={sx:.4f}, y={sy:.4f}"),
                }
    
    def _fit_diagnostics(self, tform, src, dst):

        M = np.asarray(tform.params, dtype=float)

        sx = math.hypot(M[0, 0], M[1, 0])
        sy = math.hypot(M[0, 1], M[1, 1])
        rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))

        src = np.asarray(src, dtype=float)
        if len(src):
            pred = tform(src)
            rmse = float(np.sqrt(np.mean(np.sum((pred - np.asarray(dst, float)) ** 2, axis=1))))
            rmse_txt = f"{rmse:.1f} px"
        else:
            rmse = None                     # no landmarks to measure against
            rmse_txt = "n/a (no pairs)"

        return {
                    "scale_x": sx,
                    "scale_y": sy,
                    "rotation_deg": rot,
                    "rmse_px": rmse,
                    "text": (f"scale x={sx:.4f}, y={sy:.4f}  rot={rot:.2f} deg   fit RMSE={rmse_txt}"),
                }
    
    # ---------------------------------------------------------------------------
    # TransformRecord: build / save / load / re-apply
    # ---------------------------------------------------------------------------

    @staticmethod
    def _yaml():
        """Return the PyYAML module if importable, else None (so CSV is used)."""
        try:
            import yaml
            return yaml
        except Exception:
            return None

    def _resolve_format(self, fmt):
        fmt = (fmt or "auto").lower()
        if fmt in ("yaml", "yml"):
            return "yaml"
        if fmt == "csv":
            return "csv"
        # "auto": YAML is the most suitable for a small matrix + labelled metadata
        # record; fall back to CSV when PyYAML is unavailable.
        return "yaml" if self._yaml() is not None else "csv"

    def _build_transform_record(self, tform, transform_type, fit_info=None, n_pairs=None,
                                flip_x=None, flip_y=None, mrc_shape=None, tiff_shape=None,
                                fixed_scale=None, scale_tolerance=None, pixel_spacing_um=None,
                                created_at=None):
        from clem_dataclasses import TransformRecord

        fit_info = fit_info or {}
        M = np.asarray(tform.params, dtype=float)

        if pixel_spacing_um is None and self.mrc_reader is not None:
            pixel_spacing_um = getattr(self.mrc_reader, "pixel_spacing_um", None)

        return TransformRecord(
            matrix=M.tolist(),
            transform_type=transform_type,
            flip_x=self._resolve_flip(flip_x, "flip_x"),
            flip_y=self._resolve_flip(flip_y, "flip_y"),
            scale_x=fit_info.get("scale_x"),
            scale_y=fit_info.get("scale_y"),
            rotation_deg=fit_info.get("rotation_deg"),
            rmse_px=fit_info.get("rmse_px"),
            fixed_scale=fit_info.get("expected_scale", fixed_scale),
            scale_tolerance=fit_info.get("scale_tolerance", scale_tolerance),
            n_pairs=n_pairs,
            mrc_shape=tuple(mrc_shape) if mrc_shape is not None else None,
            tiff_shape=tuple(tiff_shape) if tiff_shape is not None else None,
            pixel_spacing_um=pixel_spacing_um,
            created_at=created_at or datetime.now().isoformat(timespec="seconds"),
        )

    def _default_save_dir(self):
        base = getattr(self.mrc_reader, "output_root", None) if self.mrc_reader is not None else None
        return os.path.join(base, "transforms") if base else "transforms"

    def save_transform(self, record, save_dir=None, fmt="auto", filename=None):
        """Write a TransformRecord to disk and return the path.

        The file name always contains the transform type and the creation
        date-time, e.g. ``transform_similarity_20260716-181123.yaml``.
        """
        if save_dir is None:
            save_dir = self._default_save_dir()
        os.makedirs(save_dir, exist_ok=True)

        fmt = self._resolve_format(fmt)
        ext = "yaml" if fmt == "yaml" else "csv"

        if filename is None:
            stamp = record.created_at or datetime.now().isoformat(timespec="seconds")
            # ISO 2026-07-16T18:11:23 -> 20260716-181123
            stamp = stamp.replace("-", "").replace(":", "").replace("T", "-")
            ttype = (record.transform_type or "transform")
            filename = f"transform_{ttype}_{stamp}.{ext}"

        path = os.path.join(save_dir, filename)
        # Never silently clobber a different transform saved in the same second.
        if os.path.exists(path):
            stem, dot, tail = filename.rpartition(".")
            n = 1
            while os.path.exists(path):
                path = os.path.join(save_dir, f"{stem}-{n}{dot}{tail}")
                n += 1

        if fmt == "yaml":
            self._write_yaml(record, path)
        else:
            self._write_csv(record, path)

        record.source_path = path
        return path

    def _write_yaml(self, record, path):
        yaml = self._yaml()
        if yaml is None:
            raise RuntimeError("PyYAML is not available; call save_transform(..., fmt='csv').")
        with open(path, "w") as fh:
            fh.write("# CLEM TIFF->MRC transform (maps TIFF pixel coords -> MRC pixel coords)\n")
            fh.write("# apply as: warp(img, ProjectiveTransform(matrix=matrix).inverse)\n")
            yaml.safe_dump(record.to_dict(), fh, sort_keys=False, default_flow_style=False)

    def _write_csv(self, record, path):
        d = record.to_dict()
        matrix = d.pop("matrix")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["# CLEM TIFF->MRC transform; maps TIFF px -> MRC px"])
            w.writerow(["key", "value"])
            for key, val in d.items():
                if isinstance(val, (list, tuple)):
                    val = ";".join(str(v) for v in val)
                w.writerow([key, "" if val is None else val])
            if matrix is not None:
                for i, row in enumerate(matrix):
                    for j, val in enumerate(row):
                        w.writerow([f"m{i}{j}", repr(float(val))])

    def load_transform(self, path):
        """Load a transform from .yaml/.yml, .csv, or a legacy .txt export.
        Returns a TransformRecord."""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            record = self._read_yaml(path)
        elif ext == ".csv":
            record = self._read_csv(path)
        else:
            record = self._read_legacy_txt(path)
        record.source_path = path
        return record

    def _read_yaml(self, path):
        from clem_dataclasses import TransformRecord
        yaml = self._yaml()
        if yaml is None:
            raise RuntimeError("PyYAML is not available to read a YAML transform.")
        with open(path) as fh:
            data = yaml.safe_load(fh)
        return TransformRecord.from_dict(data)

    def _read_csv(self, path):
        from clem_dataclasses import TransformRecord
        kv, mvals = {}, {}
        with open(path, newline="") as fh:
            for row in csv.reader(fh):
                if not row:
                    continue
                key = row[0].strip()
                if not key or key.startswith("#") or key.lower() == "key":
                    continue
                val = row[1].strip() if len(row) > 1 else ""
                if len(key) == 3 and key[0] == "m" and key[1:].isdigit():
                    mvals[key] = float(val)
                else:
                    kv[key] = val
        d = dict(kv)
        d["matrix"] = ([[mvals.get(f"m{i}{j}", 0.0) for j in range(3)] for i in range(3)]
                       if mvals else None)
        return TransformRecord.from_dict(d)

    def _read_legacy_txt(self, path):
        """Parse the old commented-TSV export (as written by export_transform /
        the UI's Export Transform button)."""
        from clem_dataclasses import TransformRecord
        meta, rows = {}, []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    body = s.lstrip("#").strip()
                    if "=" in body:
                        k, _, v = body.partition("=")
                        meta[k.strip().lower()] = v.strip()
                    continue
                parts = s.replace(",", " ").split()
                try:
                    nums = [float(p) for p in parts]
                except ValueError:
                    continue
                if len(nums) >= 3:
                    rows.append(nums[:3])
        d = {
            "matrix": rows[:3] if len(rows) >= 3 else None,
            "transform_type": meta.get("transform_type"),
            "flip_x": meta.get("flip_x"),
            "flip_y": meta.get("flip_y"),
            "pixel_spacing_um": meta.get("pixel_spacing_um"),
            "n_pairs": meta.get("n_pairs"),
            "mrc_shape": meta.get("mrc_shape_hw"),
            "tiff_shape": meta.get("tiff_shape_czyx"),
        }
        return TransformRecord.from_dict(d)

    def transform_from_record(self, record):
        """Reconstruct a skimage ProjectiveTransform from a TransformRecord."""
        if record.matrix is None:
            raise ValueError("TransformRecord has no matrix to reconstruct.")
        return ProjectiveTransform(matrix=np.asarray(record.matrix, dtype=float))

    def load_transform_from_csv(self, path):
        """Backward-compatible helper: return just the skimage transform
        (used by the UI's Import Transform for .csv files)."""
        return self.transform_from_record(self.load_transform(path))

    def run_apply_loaded_transform(self, record, tiff_stack, mrc_shape, point_pairs=None,
                               scale_limited=True, fixed_scale=None, scale_tolerance=None,
                               flip_x=None, flip_y=None, status_cb=None,
                               auto_save=False, save_dir=None, save_format="auto"):

        if isinstance(record, str):                 # allow passing a path directly
            record = self.load_transform(record)

        if point_pairs is None:
            tform = self.transform_from_record(record)
            note = "re-applied stored transform (no re-fit)"
            new_record = record
            if fixed_scale is not None:
                # Rescale the stored transform about the image centre, no pairs needed.
                center = self._image_center(tiff_stack.shape)
                if center is None:
                    center = np.zeros(2)
                matrix, stored = self._rescale_matrix(tform.params, fixed_scale, center)
                tform = ProjectiveTransform(matrix=matrix)
                note = f"re-applied stored transform, rescaled {stored:.4f}->{float(fixed_scale):.4f}"
                new_record = self._build_transform_record(
                    tform, record.transform_type or "similarity",
                    fit_info={"scale_x": fixed_scale, "scale_y": fixed_scale,
                              "rotation_deg": record.rotation_deg},
                    n_pairs=0, flip_x=flip_x, flip_y=flip_y,
                    mrc_shape=mrc_shape, tiff_shape=tiff_stack.shape)
            fit_info = {
                "scale_x": (fixed_scale if fixed_scale is not None else record.scale_x),
                "scale_y": (fixed_scale if fixed_scale is not None else record.scale_y),
                "rotation_deg": record.rotation_deg, "rmse_px": record.rmse_px,
                "text": note,
            }
            n_pairs = record.n_pairs or 0
        else:
            transform_type = record.transform_type or "similarity"
            if scale_limited:
                # Use the caller's scale if given, otherwise the stored one.
                scale = fixed_scale if fixed_scale is not None else record.mean_scale
                if scale is None:
                    raise ValueError("No scale to limit to: pass fixed_scale=... "
                                     "or use a record that carries a scale.")
                tol = scale_tolerance if scale_tolerance is not None else (record.scale_tolerance or 0.05)
                tform, fit_info, n_pairs = self.fit_tiff_to_mrc(
                    point_pairs, transform_type, tiff_shape=tiff_stack.shape,
                    fixed_scale=scale, scale_tolerance=tol,
                    flip_x=flip_x, flip_y=flip_y)
            else:
                tform, fit_info, n_pairs = self.fit_tiff_to_mrc(
                    point_pairs, transform_type, tiff_shape=tiff_stack.shape,
                    flip_x=flip_x, flip_y=flip_y)
            new_record = self._build_transform_record(
                tform, transform_type, fit_info=fit_info, n_pairs=n_pairs,
                flip_x=flip_x, flip_y=flip_y, mrc_shape=mrc_shape,
                tiff_shape=tiff_stack.shape)

        warped_channels = self.warp_channels_to_mrc(
            tiff_stack=tiff_stack, tform=tform, mrc_shape=mrc_shape,
            flip_x=flip_x, flip_y=flip_y, status_cb=status_cb)

        result = {
            "transform": tform,
            "fit_info": fit_info,
            "n_pairs": n_pairs,
            "warped_channels": warped_channels,
            "record": new_record,
        }
        if auto_save and point_pairs is not None:   # only save a freshly-fit transform
            try:
                result["saved_path"] = self.save_transform(new_record, save_dir=save_dir, fmt=save_format)
            except Exception as exc:
                result["saved_path"] = None
                result["save_error"] = str(exc)
        return result

    # ---------------------------------------------------------------------------
    # Import and export functions
    # ---------------------------------------------------------------------------

    def export_transform(self, path, tform, transform_type, flip_x=False, flip_y=False, mrc_shape=None, tiff_shape=None,
    pixel_spacing_um=None, n_pairs=None,):
        
        M = np.asarray(tform.params, dtype=float)

        with open(path, "w") as fh:
            fh.write("# MRC Registration Tool - transform export\n")
            fh.write("# maps TIFF (source) pixel coords -> MRC (destination) pixel coords\n")
            fh.write("# apply as: warp(img, ProjectiveTransform(matrix=M).inverse)\n")
            fh.write(f"# transform_type = {transform_type}\n")
            fh.write(f"# flip_x = {bool(flip_x)}\n")
            fh.write(f"# flip_y = {bool(flip_y)}\n")

            if mrc_shape is not None:
                h, w = mrc_shape
                fh.write(f"# mrc_shape_hw = {h},{w}\n")

            if tiff_shape is not None:
                fh.write("# tiff_shape_czyx = ")
                fh.write(",".join(str(s) for s in tiff_shape) + "\n")

            if pixel_spacing_um is not None:
                fh.write(f"# pixel_spacing_um = {pixel_spacing_um:.6f}\n")

            if n_pairs is not None:
                fh.write(f"# n_pairs = {n_pairs}\n")

            fh.write("# matrix 3x3 row-major (homogeneous):\n")
            for row in M:
                fh.write("\t".join(f"{v:.10g}" for v in row) + "\n")