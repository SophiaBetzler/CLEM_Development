"""
CLEM correlation: fit and apply a TIFF -> MRC transform.

Simplified coordinate model
---------------------------
The TIFF's rotation/flip relative to the MRC is arbitrary, so instead of
modelling a mirror in the transform we flip the TIFF *for display* until it
matches the (always Y-flipped) MRC, and pick landmarks on that matched view.
Because the images are then in the same handedness, the transform is ALWAYS a
plain proper fit -- there is no reflection matrix and no "case" to choose.

Frames
  * Landmark picks are in the DISPLAY frame of each image (what the user
    clicked): display TIFF pixels for "tiff", display MRC pixels for "mrc".
  * The fitted transform maps DISPLAY-TIFF pixels -> DISPLAY-MRC pixels and is
    fit directly with estimate_transform (no pre/post reflection).
  * Warping flips the raw TIFF by (flip_x, flip_y) -- the TIFF display flip the
    user chose -- to reproduce the matched view, then warps it with the fitted
    transform into the DISPLAY-MRC frame.
  * flip_x / flip_y describe the TIFF display orientation ONLY; they are stored
    in the record so a re-applied transform can reproduce the same matched view
    on a different TIFF.  The MRC display flip is fixed by the UI and never
    enters this module.
"""

import os
import csv
import math
from datetime import datetime

import numpy as np
from skimage.transform import estimate_transform, warp, ProjectiveTransform


class CLEMCorrelator:

    MIN_PAIRS = {"euclidean": 2, "similarity": 2, "affine": 3, "projective": 4}

    def __init__(self, mrc_reader=None):
        self.mrc_reader = mrc_reader
        self.last_transform = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply(M, pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        hom = np.column_stack([pts, np.ones(len(pts), dtype=float)])
        out = (M @ hom.T).T
        return out[:, :2] / out[:, 2:3]

    @staticmethod
    def _matrix_of(tform):
        return np.asarray(tform.params, dtype=float)

    @staticmethod
    def _image_center(shape):
        h, w = shape[-2:]
        return np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=float)

    @staticmethod
    def _flip_slice(img, flip_x, flip_y):
        """Flip a 2-D slice to reproduce the matched (display) view."""
        if flip_x:
            img = np.fliplr(img)
        if flip_y:
            img = np.flipud(img)
        return img

    def _diagnostics(self, M, src, dst):
        M = np.asarray(M, dtype=float)
        sx = math.hypot(M[0, 0], M[1, 0])
        sy = math.hypot(M[0, 1], M[1, 1])
        rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        src = np.asarray(src, dtype=float).reshape(-1, 2)
        if len(src):
            pred = self._apply(M, src)
            rmse = float(np.sqrt(np.mean(np.sum(
                (pred - np.asarray(dst, float).reshape(-1, 2)) ** 2, axis=1))))
            rmse_txt = f"{rmse:.2f} px"
        else:
            rmse, rmse_txt = None, "n/a"
        return {"scale_x": sx, "scale_y": sy, "rotation_deg": rot,
                "rmse_px": rmse, "det": float(np.linalg.det(M[:2, :2])),
                "text": f"scale x={sx:.4f}, y={sy:.4f}  rot={rot:.2f} deg  RMSE={rmse_txt}"}

    def _rescale_about(self, M, new_scale, src_anchor, dst_anchor):
        """Copy of M with linear scale set to new_scale (rotation preserved),
        translated so src_anchor -> dst_anchor.  For concentric re-apply to a
        different magnification."""
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

    # ------------------------------------------------------------------ #
    # Fit  (plain -- no reflection)
    # ------------------------------------------------------------------ #

    def fit(self, point_pairs, transform_type):
        """Fit a transform mapping DISPLAY-TIFF -> DISPLAY-MRC.

        point_pairs : list of {"tiff": (x, y), "mrc": (x, y)} in display pixels.
        Returns (ProjectiveTransform, fit_info dict, n_pairs).
        """
        pairs = [(p["tiff"], p["mrc"]) for p in point_pairs
                 if "tiff" in p and "mrc" in p]
        need = self.MIN_PAIRS.get(transform_type, 3)
        if len(pairs) < need:
            raise ValueError(
                f"{transform_type.capitalize()} needs >= {need} pairs "
                f"(you have {len(pairs)}).")
        src = np.array([p[0] for p in pairs], dtype=float).reshape(-1, 2)
        dst = np.array([p[1] for p in pairs], dtype=float).reshape(-1, 2)
        base = estimate_transform(transform_type, src, dst)
        M = self._matrix_of(base)
        tform = ProjectiveTransform(matrix=M)
        info = self._diagnostics(M, src, dst)
        self.last_transform = tform
        return tform, info, len(pairs)

    # ------------------------------------------------------------------ #
    # Warp  (flip the raw TIFF to the matched view, then warp)
    # ------------------------------------------------------------------ #

    def warp_channels(self, tiff_stack, tform, mrc_shape,
                      flip_x=False, flip_y=True, status_cb=None):
        """Warp every channel (max over Z) of the raw tiff_stack onto the MRC
        grid.  The raw slice is first flipped by (flip_x, flip_y) to reproduce
        the matched display view the transform was fit against.  Returns a list
        of 2-D float32 arrays in the DISPLAY-MRC frame."""
        mrc_h, mrc_w = mrc_shape
        C, Z = tiff_stack.shape[:2]
        out = []
        for c in range(C):
            acc = None
            for z in range(Z):
                if status_cb is not None:
                    status_cb(f"Warping channel {c + 1}/{C}, z {z + 1}/{Z}...")
                img = self._flip_slice(tiff_stack[c, z], flip_x, flip_y)
                warped = warp(img, tform.inverse, output_shape=(mrc_h, mrc_w),
                              order=1, preserve_range=True,
                              mode="constant", cval=0).astype(np.float32)
                acc = warped if acc is None else np.maximum(acc, warped)
            out.append(acc)
        return out

    def warp_slice(self, tiff_stack, c, z, tform, mrc_shape,
                   flip_x=False, flip_y=True):
        """Warp one (channel, z) raw slice onto the MRC grid (DISPLAY-MRC frame),
        applying the same TIFF display flip first."""
        mrc_h, mrc_w = mrc_shape
        img = self._flip_slice(tiff_stack[c, z], flip_x, flip_y)
        return warp(img, tform.inverse, output_shape=(mrc_h, mrc_w),
                    order=1, preserve_range=True, mode="constant",
                    cval=0).astype(np.float32)

    def warp_crop(self, tiff_stack, c, z, tform, x0, y0, cw,
                  flip_x=False, flip_y=True, mrc_shape=None):
        """Warp one (channel, z) raw slice straight into a cw x cw window whose
        top-left corner is (x0, y0) in DISPLAY-MRC pixels.

        Equivalent to warp_slice(...)[y0:y0+cw, x0:x0+cw] but it interpolates
        cw**2 pixels instead of the whole montage. Writing FOV crops used to
        warp the entire montage once per channel per z just to cut a few small
        windows out of it, which is what made it slow.

        Pass mrc_shape to zero everything outside the montage, matching what
        cropping a full warped plane would have produced.
        """
        img = self._flip_slice(tiff_stack[c, z], flip_x, flip_y)
        off = np.asarray([float(x0), float(y0)])
        crop = warp(img, lambda coords: tform.inverse(coords + off),
                    output_shape=(cw, cw), order=1, preserve_range=True,
                    mode="constant", cval=0).astype(np.float32)

        if mrc_shape is not None:
            mrc_h, mrc_w = mrc_shape[-2:]
            if x0 < 0 or y0 < 0 or x0 + cw > mrc_w or y0 + cw > mrc_h:
                yy, xx = np.mgrid[0:cw, 0:cw]
                inside = ((xx + x0 >= 0) & (xx + x0 < mrc_w) &
                          (yy + y0 >= 0) & (yy + y0 < mrc_h))
                crop = np.where(inside, crop, np.float32(0.0))
        return crop

    # ------------------------------------------------------------------ #
    # High-level entry points
    # ------------------------------------------------------------------ #

    def run_fit_and_warp(self, point_pairs, transform_type, tiff_stack, mrc_shape,
                         flip_x=False, flip_y=True, status_cb=None,
                         mrc_pixel_spacing_um=None, tiff_pixel_spacing_um=None,
                         auto_save=False, save_dir=None):
        """Fresh fit from display-frame landmark pairs, then warp.  flip_x/flip_y
        are the TIFF display orientation and are stored in the record."""
        tform, info, n_pairs = self.fit(point_pairs, transform_type)
        warped = self.warp_channels(tiff_stack, tform, mrc_shape,
                                     flip_x=flip_x, flip_y=flip_y, status_cb=status_cb)
        record = self._make_record(tform, transform_type, info, n_pairs,
                                    mrc_shape, tiff_stack.shape, flip_x, flip_y,
                                    mrc_pixel_spacing_um, tiff_pixel_spacing_um)
        result = {"transform": tform, "fit_info": info, "n_pairs": n_pairs,
                  "warped_channels": warped, "record": record}
        if auto_save:
            try:
                result["saved_path"] = self.save_transform(record, save_dir=save_dir)
            except Exception as exc:
                result["saved_path"] = None
                result["save_error"] = str(exc)
        return result

    def run_reapply(self, record, tiff_stack, mrc_shape,
                tiff_pixel_spacing_um=None, mrc_pixel_spacing_um=None,
                status_cb=None, center_on_mrc=True):
        """Re-apply a stored transform: rotation and flips from the record, scale
        from the pixel-size ratio, and the TIFF footprint centred on the MRC.
        No registration translation is carried over."""
        if isinstance(record, str):
            record = self.load_transform(record)
    
        M = np.asarray(record.matrix, dtype=float)
        A = M[:2, :2]
        stored_scale = float(np.hypot(A[0, 0], A[1, 0]))
        rot_deg = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    
        want_scale, scale_src = stored_scale, "stored (pixel sizes unavailable)"
        if tiff_pixel_spacing_um and mrc_pixel_spacing_um:
            want_scale = float(tiff_pixel_spacing_um) / float(mrc_pixel_spacing_um)
            scale_src = (f"{float(tiff_pixel_spacing_um):.6f} / "
                         f"{float(mrc_pixel_spacing_um):.6f} um/px")
    
        src_anchor = self._image_center(tiff_stack.shape)
        if center_on_mrc:
            dst_anchor, anchor_txt = self._image_center(mrc_shape), "MRC centre"
        else:
            dst_anchor = self._apply(M, self._image_center(mrc_shape)[None, :])[0]
            anchor_txt = "previous TIFF centre"
    
        M2, _ = self._rescale_about(M, want_scale, src_anchor, dst_anchor)
        tform = ProjectiveTransform(matrix=M2)
    
        fx, fy = bool(record.flip_x), bool(record.flip_y)
        H, W = tiff_stack.shape[-2:]
        mrc_h, mrc_w = mrc_shape[-2:]
    
        print("[INFO] Re-applying stored transform")
        print(f"       rotation       : {rot_deg:+.3f} deg  (preserved)")
        print(f"       scale stored   : {stored_scale:.6f} MRC px / TIFF px")
        print(f"       scale applied  : {want_scale:.6f} MRC px / TIFF px  <- {scale_src}")
        print(f"       flip_x, flip_y : {fx}, {fy}")
        print(f"       anchored on    : {anchor_txt}")
        print(f"       TIFF centre    : ({src_anchor[0]:.1f}, {src_anchor[1]:.1f}) "
              f"-> ({dst_anchor[0]:.1f}, {dst_anchor[1]:.1f})")
        print(f"       footprint      : {W}x{H} TIFF px -> "
              f"{W * want_scale:.0f}x{H * want_scale:.0f} of {mrc_w}x{mrc_h} MRC px")
        if record.transform_type == "projective":
            print("       [WARN] projective terms are dropped on re-apply (affine only)")
    
        info = self._diagnostics(M2, np.empty((0, 2)), np.empty((0, 2)))
        info["text"] = (f"re-applied: rot {rot_deg:+.2f} deg, "
                        f"scale {stored_scale:.4f} -> {want_scale:.4f}, "
                        f"flips ({fx}, {fy}), centred on {anchor_txt}")
    
        warped = self.warp_channels(tiff_stack, tform, mrc_shape,
                                    flip_x=fx, flip_y=fy, status_cb=status_cb)
        new_record = self._make_record(tform, record.transform_type, info,
                                       record.n_pairs, mrc_shape, tiff_stack.shape,
                                       fx, fy, mrc_pixel_spacing_um, tiff_pixel_spacing_um)
        return {"transform": tform, "fit_info": info,
                "n_pairs": record.n_pairs or 0,
                "warped_channels": warped, "record": new_record}

    def run_reapply_refine(self, record, point_pairs, transform_type, tiff_stack,
                           mrc_shape, flip_x=False, flip_y=True,
                           mrc_pixel_spacing_um=None, tiff_pixel_spacing_um=None,
                           status_cb=None):
        """Fine-tune: a fresh fit on a few landmarks placed on the new TIFF's
        matched view (scale determined by the landmarks)."""
        return self.run_fit_and_warp(
            point_pairs, transform_type, tiff_stack, mrc_shape,
            flip_x=flip_x, flip_y=flip_y, status_cb=status_cb,
            mrc_pixel_spacing_um=mrc_pixel_spacing_um,
            tiff_pixel_spacing_um=tiff_pixel_spacing_um)

    # ------------------------------------------------------------------ #
    # Record build / save / load
    # ------------------------------------------------------------------ #

    def _make_record(self, tform, transform_type, info, n_pairs,
                     mrc_shape, tiff_shape, flip_x, flip_y,
                     mrc_pixel_spacing_um, tiff_pixel_spacing_um):
        from clem_dataclasses import TransformRecord
        M = self._matrix_of(tform)
        if mrc_pixel_spacing_um is None and self.mrc_reader is not None:
            mrc_pixel_spacing_um = getattr(self.mrc_reader, "pixel_spacing_um", None)
        return TransformRecord(
            matrix=M.tolist(),
            transform_type=transform_type,
            flip_x=bool(flip_x), flip_y=bool(flip_y),     # TIFF display orientation
            scale_x=info.get("scale_x"),
            scale_y=info.get("scale_y"),
            rotation_deg=info.get("rotation_deg"),
            rmse_px=info.get("rmse_px"),
            n_pairs=n_pairs,
            mrc_shape=tuple(mrc_shape) if mrc_shape is not None else None,
            tiff_shape=tuple(tiff_shape) if tiff_shape is not None else None,
            pixel_spacing_um=mrc_pixel_spacing_um,
            tiff_pixel_spacing_um=tiff_pixel_spacing_um,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )

    @staticmethod
    def _yaml():
        try:
            import yaml
            return yaml
        except Exception:
            return None

    def _default_save_dir(self):
        base = getattr(self.mrc_reader, "output_root", None) if self.mrc_reader else None
        return os.path.join(base, "transforms") if base else "transforms"

    def save_transform(self, record, save_dir=None, filename=None):
        if save_dir is None:
            save_dir = self._default_save_dir()
        os.makedirs(save_dir, exist_ok=True)
        yaml = self._yaml()
        ext = "yaml" if yaml is not None else "csv"
        if filename is None:
            stamp = (record.created_at or datetime.now().isoformat(timespec="seconds"))
            stamp = stamp.replace("-", "").replace(":", "").replace("T", "-")
            filename = f"transform_{record.transform_type or 'transform'}_{stamp}.{ext}"
        path = os.path.join(save_dir, filename)
        n = 1
        while os.path.exists(path):
            stem, dot, tail = filename.rpartition(".")
            path = os.path.join(save_dir, f"{stem}-{n}{dot}{tail}")
            n += 1
        if yaml is not None:
            with open(path, "w") as fh:
                fh.write("# CLEM TIFF->MRC transform (maps DISPLAY TIFF px -> DISPLAY MRC px)\n")
                fh.write("# flip_x / flip_y are the TIFF display orientation used to match the MRC\n")
                yaml.safe_dump(record.to_dict(), fh, sort_keys=False, default_flow_style=False)
        else:
            self._write_csv(record, path)
        record.source_path = path
        return path

    def _write_csv(self, record, path):
        d = record.to_dict()
        matrix = d.pop("matrix")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["# CLEM TIFF->MRC transform; DISPLAY TIFF px -> DISPLAY MRC px"])
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
        from clem_dataclasses import TransformRecord
        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            yaml = self._yaml()
            if yaml is None:
                raise RuntimeError("PyYAML not available to read a YAML transform.")
            with open(path) as fh:
                data = yaml.safe_load(fh)
            rec = TransformRecord.from_dict(data)
        elif ext == ".csv":
            rec = self._read_csv(path)
        else:
            rec = self._read_legacy_txt(path)
        rec.source_path = path
        return rec

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
        d = {"matrix": rows[:3] if len(rows) >= 3 else None,
             "transform_type": meta.get("transform_type"),
             "flip_x": meta.get("flip_x"), "flip_y": meta.get("flip_y"),
             "mrc_shape": meta.get("mrc_shape_hw"),
             "tiff_shape": meta.get("tiff_shape_czyx"),
             "pixel_spacing_um": meta.get("pixel_spacing_um")}
        return TransformRecord.from_dict(d)

    def transform_from_record(self, record):
        if record.matrix is None:
            raise ValueError("TransformRecord has no matrix.")
        return ProjectiveTransform(matrix=np.asarray(record.matrix, dtype=float))