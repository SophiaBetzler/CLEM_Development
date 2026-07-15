import csv
import os
import numpy as np
import math
from skimage.transform import estimate_transform, warp, ProjectiveTransform


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

    def fit_fixed_scale_reflection(self,point_pairs, tiff_shape, fixed_scale,
                                    scale_tolerance=0.05, flip_x=False, flip_y=False,
                                    initial_transform=None):

        complete = [(p["tiff"], p["mrc"]) for p in point_pairs if "tiff" in p and "mrc" in p]

        if len(complete) < 2:
            raise ValueError("Bounded-scale reflected similarity needs at least 2 pairs; received {len(complete)}.")

        if fixed_scale <= 0:
            raise ValueError("fixed_scale must be positive.")

        if not 0 <= scale_tolerance < 1:
            raise ValueError("scale_tolerance must be between 0 and 1.")

        src = np.asarray([p[0] for p in complete], dtype=float)
        dst = np.asarray([p[1] for p in complete], dtype=float)

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

        src_h = np.column_stack([src, np.ones(len(src), dtype=float),])

        reflected_h = (F @ src_h.T).T
        reflected = (reflected_h[:, :2] / reflected_h[:, 2, None])

        free_tform = estimate_transform("similarity", reflected, dst,)

        free_A = np.asarray(free_tform.params[:2, :2], dtype=float,)

        if initial_transform is not None:
            init_params = np.asarray(initial_transform.params, dtype=float)
            init_A = init_params[:2, :2]
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

        fitted_scale = float(np.clip(unconstrained_scale, scale_min, scale_max,))

        R = free_A / unconstrained_scale

        src_center = reflected.mean(axis=0)
        dst_center = dst.mean(axis=0)

        translation = (dst_center - fitted_scale * (R @ src_center))

        bounded_similarity = np.eye(3, dtype=float)
        bounded_similarity[:2, :2] = fitted_scale * R
        bounded_similarity[:2, 2] = translation

        matrix = bounded_similarity @ F
        tform = ProjectiveTransform(matrix=matrix)

        fit_info = self._fit_diagnostics(tform, src, dst,)

        fit_info.update({
                        "expected_scale": float(fixed_scale),
                        "unconstrained_scale": unconstrained_scale,
                        "bounded_scale": fitted_scale,
                        "scale_min": scale_min,
                        "scale_max": scale_max,
                        "scale_tolerance": scale_tolerance,
                        "scale_was_clamped": not np.isclose(
                            fitted_scale,
                            unconstrained_scale,
                        ),
                        "flip_x": bool(flip_x),
                        "flip_y": bool(flip_y),
                    })

        fit_info["text"] += (
                                f"  expected={fixed_scale:.4f}"
                                f"  allowed=[{scale_min:.4f}, {scale_max:.4f}]"
                                f"  free={unconstrained_scale:.4f}"
                                f"  used={fitted_scale:.4f}"
                            )

        return tform, fit_info, len(complete)

    def fit_tiff_to_mrc(self, point_pairs, transform_type, predefined=None, tiff_shape=None,
                        fixed_scale=None, flip_x=False, flip_y=False, initial_transform=None):
        if predefined is not None:
            tform, fit_info, n_complete = self.fit_fixed_scale_reflection(
                point_pairs,
                tiff_shape=tiff_shape,
                fixed_scale=fixed_scale,
                flip_x=flip_x,
                flip_y=flip_y,
                initial_transform=initial_transform,
            )
        else:
            complete = [(p["tiff"], p["mrc"]) for p in point_pairs if "mrc" in p and "tiff" in p]

            required = self.REQUIRED_POINT_PAIRS.get(transform_type, 3)
            if len(complete) < required:
                raise ValueError(f"{transform_type.capitalize()} needs >= {required} pairs (you have {len(complete)}).")

            src = np.array([p[0] for p in complete], dtype=float)
            dst = np.array([p[1] for p in complete], dtype=float)

            tform = estimate_transform(transform_type, src, dst)
            fit_info = self._fit_diagnostics(tform, src, dst)

            self.last_tform = tform

        return tform, fit_info, len(complete)
    
    def warp_channels_to_mrc(self, tiff_stack, tform, mrc_shape, flip_x=False, flip_y=False, status_cb=None,):

        mrc_h, mrc_w = mrc_shape
        C, Z = tiff_stack.shape[:2]

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


    def apply_transform(self, point_pairs, transform_type, tiff_stack, mrc_shape, flip_x=False, flip_y=False, status_cb=None,
                        initial_transform=None, predefined=None, fixed_scale=None, tiff_shape=None):

        tform, fit_info, n_pairs = self.fit_tiff_to_mrc(point_pairs,transform_type,predefined=predefined,
            tiff_shape=tiff_shape or tiff_stack.shape, fixed_scale=fixed_scale, flip_x=flip_x, flip_y=flip_y,
            initial_transform=initial_transform,)

        warped_channels = self.warp_channels_to_mrc(tiff_stack=tiff_stack, tform=tform, mrc_shape=mrc_shape, flip_x=flip_x, flip_y=flip_y,
                                                    status_cb=status_cb,)

        return {
                    "transform": tform,
                    "fit_info": fit_info,
                    "n_pairs": n_pairs,
                    "warped_channels": warped_channels,
                }
    
    
    
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

        pred = tform(src)
        rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))

        return {
                    "scale_x": sx,
                    "scale_y": sy,
                    "rotation_deg": rot,
                    "rmse_px": rmse,
                    "text": (f"scale x={sx:.4f}, y={sy:.4f}  rot={rot:.2f} deg   fit RMSE={rmse:.1f} px"),
                }
    
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