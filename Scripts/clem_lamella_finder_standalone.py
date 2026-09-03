#!/usr/bin/env python3
"""
Standalone Lamella Finder + Cluster Matching + SerialEM Navigator Export
========================================================================
No SPACEtomo imports. YOLO is called directly via ultralytics on the
binned whole-grid image, replicating what SPACEtomo's WGModel does:

    atlas (full res, tiles placed per mdoc) --area resample--> 400 nm/px image
        --> single whole-montage YOLO pass (no tiling, no square rescale)
        --> box centers backprojected to full-res montage pixels (float scale)

Pipeline:
  1. Read atlas montage tiles + mdoc with YOUR mrc reader (see MrcReader
     adapter below) and place tiles at PieceCoordinates (no stitching).
  2. Bin to WG_MODEL_PIX_SIZE, run YOLO weights directly.
  3. Extract lamella positions from AutoLamella experiment.yaml.
  4. Cluster-match detections <-> YAML lamellae (rotation/flip search +
     Hungarian assignment).
  5. Write SerialEM navigator points with petname notes.

Dependencies:
    pip install numpy pyyaml pillow scipy ultralytics
    (torch comes with ultralytics)

Usage:
    python lamella_finder_standalone.py \
        --mrc atlas.mrc --mdoc atlas.mrc.mdoc --yaml experiment.yaml \
        --weights /path/to/wg_model.pt --out lamellae.nav
"""

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

M_TO_UM = 1_000_000.0

# ── Model parameters ─────────────────────────────────────────────────────────
WG_MODEL_PIX_SIZE = 400.0     # nm/px the weights were trained at (fixed)


# ══════════════════════════════════════════════════════════════════════════
#  YOUR MRC READER GOES HERE
# ══════════════════════════════════════════════════════════════════════════
# Replace the body of read_mrc_stack() with a call to your own reader.
# Contract: given the .mrc path, return a numpy array of shape
# (n_tiles, ny, nx) — one 2-D image per montage piece, in ZValue order.

def read_mrc_stack(mrc_path):
    # --- swap in your reader, e.g.: ---
    # from mymrcreader import MrcReader
    # return MrcReader(mrc_path).data          # (n, ny, nx)
    # ----------------------------------
    import mrcfile  # fallback so the script runs out of the box
    with mrcfile.open(mrc_path, permissive=True) as m:
        data = np.asarray(m.data)
    return data[None] if data.ndim == 2 else data


# ══════════════════════════════════════════════════════════════════════════
#  Step 1 - mdoc parsing and unstitched montage
# ══════════════════════════════════════════════════════════════════════════

def parse_mdoc(mdoc_path):
    """Minimal mdoc parser -> (frames, pixel size in Angstrom)."""
    frames, current, pixel_size = [], None, None
    with open(mdoc_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("[T"):
                continue
            if line.startswith("[ZValue"):
                if current is not None:
                    frames.append(current)
                current = {}
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key == "PixelSpacing" and pixel_size is None:
                    pixel_size = float(value)
                if current is not None:      # ignore global header block
                    current[key] = value
    if current is not None and current:
        frames.append(current)
    if pixel_size is None:
        raise ValueError(f"No PixelSpacing found in {mdoc_path}")
    return frames, pixel_size


def build_unstitched_montage(mrc_path, mdoc_path):
    """Direct tile placement at PieceCoordinates. Returns
    (canvas, pixel_size_A, origin_shift, frames, tile_shape)."""
    frames, pixel_size = parse_mdoc(mdoc_path)
    data = read_mrc_stack(mrc_path)
    tile_h, tile_w = data.shape[1], data.shape[2]

    # SerialEM writes piece placement under different keys depending on
    # version/processing state: raw acquisition -> PieceCoordinates,
    # after alignment -> AlignedPieceCoordsVS / AlignedPieceCoords.
    coord_key = None
    for key in ("PieceCoordinates", "AlignedPieceCoordsVS", "AlignedPieceCoords"):
        if frames and key in frames[0]:
            coord_key = key
            break
    if coord_key is None:
        avail = sorted(frames[0].keys()) if frames else []
        raise ValueError(
            "No piece coordinates found in mdoc (looked for PieceCoordinates, "
            "AlignedPieceCoordsVS, AlignedPieceCoords).\n"
            f"Keys present in first frame: {avail}\n"
            "Is this really a montage mdoc?")
    print(f"      using '{coord_key}' for tile placement")

    coords = []
    for frame in frames:
        pc = frame.get(coord_key)
        if pc is None:
            raise ValueError(f"Frame missing {coord_key} in mdoc")
        parts = list(map(float, pc.split()))
        x, y = parts[0], parts[1]
        coords.append((int(round(x)), int(round(y))))
    coords = np.array(coords)
    origin_shift = coords.min(axis=0).copy()
    coords -= origin_shift

    canvas = np.zeros((coords[:, 1].max() + tile_h,
                       coords[:, 0].max() + tile_w), dtype=data.dtype)
    for i, (x, y) in enumerate(coords):
        canvas[y:y + tile_h, x:x + tile_w] = data[i]
    return canvas, pixel_size, origin_shift, frames, (tile_h, tile_w), coord_key


def montage_to_stage_transform(frames, origin_shift, tile_shape,
                               coord_key="PieceCoordinates"):
    """Least-squares affine: montage pixel (x, y) -> stage (X, Y) um,
    fit from (piece-coordinate center, StagePosition) pairs."""
    pts_pix, pts_stage = [], []
    th, tw = tile_shape
    for frame in frames:
        pc, spv = frame.get(coord_key), frame.get("StagePosition")
        if pc is None or spv is None:
            continue
        parts = list(map(float, pc.split()))
        x, y = parts[0], parts[1]
        sx, sy = map(float, spv.split()[:2])
        pts_pix.append((x - origin_shift[0] + tw / 2, y - origin_shift[1] + th / 2))
        pts_stage.append((sx, sy))
    if len(pts_pix) < 3:
        return None
    P = np.hstack([np.array(pts_pix), np.ones((len(pts_pix), 1))])
    A, *_ = np.linalg.lstsq(P, np.array(pts_stage), rcond=None)

    def f(px, py):
        v = np.array([px, py, 1.0]) @ A
        return float(v[0]), float(v[1])
    return f


# ══════════════════════════════════════════════════════════════════════════
#  Step 2 - direct YOLO lamella detection (no SPACEtomo)
# ══════════════════════════════════════════════════════════════════════════

def rescale_atlas(atlas, atlas_pix_size_A):
    """
    Resample the atlas to exactly WG_MODEL_PIX_SIZE (nm/px) with PIL
    (area-average). No cropping, no integer binning - the back-projection
    to full-resolution montage pixels is an exact float scale.

    Returns (img8 uint8 array at 400 nm/px, (scale_x, scale_y)) where
        full_res_px_x = binned_px_x * scale_x
        full_res_px_y = binned_px_y * scale_y
    """
    atlas_pix_nm = atlas_pix_size_A / 10.0
    if atlas_pix_nm > WG_MODEL_PIX_SIZE:
        raise ValueError(
            f"Atlas pixel ({atlas_pix_nm:.1f} nm) larger than model pixel "
            f"({WG_MODEL_PIX_SIZE} nm) - cannot downscale.")
    h, w = atlas.shape
    new_w = int(round(w * atlas_pix_nm / WG_MODEL_PIX_SIZE))
    new_h = int(round(h * atlas_pix_nm / WG_MODEL_PIX_SIZE))

    # normalize BEFORE resizing, ignoring empty (zero-filled) montage gaps
    a = atlas.astype(np.float64)
    mask = a != 0
    lo, hi = np.percentile(a[mask] if mask.any() else a, [1, 99])
    img8 = np.clip((a - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)

    img8 = np.array(Image.fromarray(img8).resize((new_w, new_h), Image.BOX))
    scale = (w / new_w, h / new_h)   # exact float, per axis
    return img8, scale


def detect_lamellae(atlas, atlas_pix_size_A, weights_path,
                    prob_threshold=0.0, save_png=None):
    """
    Run YOLO directly on the binned whole-grid image.
    Returns list of dicts: px, py (full-res atlas pixel center), cat, prob.
    """
    from ultralytics import YOLO

    img8, (scale_x, scale_y) = rescale_atlas(atlas, atlas_pix_size_A)
    pil = Image.fromarray(img8).convert("RGB")

    model = YOLO(str(weights_path))
    print(f"      model classes: {model.names}")

    # Single pass over the whole montage at native 400 nm/px. YOLO is fully
    # convolutional; imgsz = image size rounded up to a multiple of 32
    # (network stride), so no content rescaling happens.
    H, W = img8.shape
    results = model(pil,
                    imgsz=(math.ceil(H / 32) * 32, math.ceil(W / 32) * 32),
                    conf=prob_threshold or 0.25,
                    verbose=False)[0]

    detections = []
    boxes_for_overlay = []
    for b in results.boxes:
        prob = float(b.conf.item())
        if prob < prob_threshold:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()    # in 400nm-image px
        cls = int(b.cls.item())
        cat = model.names.get(cls, str(cls))
        # center in 400nm-image px -> ORIGINAL montage px (exact float scale)
        px = (x1 + x2) / 2 * scale_x
        py = (y1 + y2) / 2 * scale_y
        detections.append(dict(px=float(px), py=float(py),
                               cat=cat, prob=float(prob)))
        boxes_for_overlay.append(((x1, y1, x2, y2), cat, prob))

    if save_png:
        from PIL import ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(pil)
        for (x1, y1, x2, y2), cat, prob in boxes_for_overlay:
            label = f"{cat} {prob:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            tb = draw.textbbox((x1, max(y1 - 34, 0)), label, font=font)
            draw.rectangle(tb, fill=(255, 0, 0))
            draw.text((x1, max(y1 - 34, 0)), label, fill=(255, 255, 255),
                      font=font)
        pil.save(save_png)
        print(f"      overlay with {len(boxes_for_overlay)} detections -> {save_png}")

    return detections


# ══════════════════════════════════════════════════════════════════════════
#  Step 3 - experiment.yaml extraction (no CLEM reference)
# ══════════════════════════════════════════════════════════════════════════

def extract_yaml_lamellae(yaml_path):
    """Projected (z=0, 0 deg tilt) lamella XY in um from AutoLamella yaml.
    No reference subtraction - the cluster match absorbs the global offset."""
    with open(yaml_path, encoding="utf-8", errors="replace") as fh:
        data = yaml.safe_load(fh)

    out = []
    for pos in data.get("positions", []):
        poi = pos.get("poi") or {}
        sp = (((pos.get("poses") or {}).get("MILLING") or {})
              .get("stage_position") or {})
        sx = float(sp.get("x") or 0) * M_TO_UM
        sy = float(sp.get("y") or 0) * M_TO_UM
        sz = float(sp.get("z") or 0) * M_TO_UM
        t = float(sp.get("t") or 0)                    # radians
        px = float(poi.get("x") or 0) * M_TO_UM
        py = float(poi.get("y") or 0) * M_TO_UM

        if pos.get("milling_angle") is not None and abs(math.cos(t)) > 1e-12:
            cx = sx + px
            cy = sy + py * math.cos(t)
            cz = sz + py * math.sin(t)
            x, y = cx, cy + cz * math.tan(t)
        else:
            x, y = sx + px, sy + py

        out.append(dict(name=pos.get("petname", "") or f"pos{pos.get('number', '')}",
                        number=pos.get("number", ""), x=x, y=y))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Step 4 - cluster matching
# ══════════════════════════════════════════════════════════════════════════

def _assign(cost):
    if HAVE_SCIPY:
        return linear_sum_assignment(cost)
    ri, ci = [], []
    c = cost.copy()
    for _ in range(min(c.shape)):
        i, j = np.unravel_index(np.argmin(c), c.shape)
        ri.append(i); ci.append(j)
        c[i, :] = np.inf; c[:, j] = np.inf
    return np.array(ri), np.array(ci)


def match_clusters(det_xy, yaml_xy, try_flips=True, angles_deg=range(0, 360, 5)):
    """Match two point clouds robust to global translation/rotation/flip.
    Returns (pairs [(det_i, yaml_i, residual_um)], R, rms)."""
    D = np.asarray(det_xy, float)
    Y = np.asarray(yaml_xy, float)
    Dc = D - D.mean(axis=0)
    Yc = Y - Y.mean(axis=0)

    flips = [np.diag([1, 1]), np.diag([-1, 1]),
             np.diag([1, -1]), np.diag([-1, -1])] if try_flips else [np.eye(2)]

    best = None
    for F, ang in product(flips, angles_deg):
        a = math.radians(ang)
        R = np.array([[math.cos(a), -math.sin(a)],
                      [math.sin(a),  math.cos(a)]]) @ F
        Yt = Yc @ R.T
        cost = np.linalg.norm(Dc[:, None, :] - Yt[None, :, :], axis=2)
        ri, ci = _assign(cost)
        rms = float(np.sqrt((cost[ri, ci] ** 2).mean()))
        if best is None or rms < best[0]:
            best = (rms, ri, ci, R, cost)

    rms, ri, ci, R, cost = best
    pairs = [(int(i), int(j), float(cost[i, j])) for i, j in zip(ri, ci)]
    return pairs, R, rms


# ══════════════════════════════════════════════════════════════════════════
#  Step 5 - SerialEM navigator output
# ══════════════════════════════════════════════════════════════════════════

def write_serialem_script(items, script_path, map_item=1, flip_y_height=None):
    """
    Emit a SerialEM script that adds one nav point per lamella via
    AddImagePosAsNavPoint, letting SerialEM do the pixel->stage conversion
    with its own (aligned) montage geometry.

    items: dicts with px, py (full-res montage pixel coords) and note.
    map_item: navigator table index of the montage map to load.
    flip_y_height: if set (image height in px), Y is flipped (H - py) to
        convert from array row (top-down) to SerialEM image convention
        (bottom-up). Verify with one test point and toggle --no-flip-y.
    """
    lines = ["ScriptName LamellaNavPoints",
             f"LoadOtherMap {map_item} A"]
    for it in items:
        x = it["px"]
        y = (flip_y_height - it["py"]) if flip_y_height else it["py"]
        lines.append(f"AddImagePosAsNavPoint A {x:.1f} {y:.1f}")
        lines.append("ReportNumTableItems")
        lines.append(f"ChangeItemNote $reportedValue1 {it['note']}")
    Path(script_path).write_text("\n".join(lines) + "\n")


def write_navigator(items, out_path, append_to=None):
    lines = []
    start_index = 1
    if append_to and Path(append_to).exists():
        text = Path(append_to).read_text()
        lines.append(text.rstrip("\n"))
        import re
        nums = [int(m) for m in re.findall(r"\[Item\s*=\s*(\d+)", text)]
        start_index = (max(nums) + 1) if nums else 1
    else:
        lines.append("AdocVersion = 2.00\n")

    for k, it in enumerate(items):
        idx = start_index + k
        x, y, z = it["stage_x"], it["stage_y"], it.get("stage_z", 0.0)
        lines.append(f"""
[Item = {idx}]
Color = 0
StageXYZ = {x:.3f} {y:.3f} {z:.3f}
NumPts = 1
Regis = 1
Type = 0
Note = {it['note']}
PtsX = {x:.3f}
PtsY = {y:.3f}""")

    Path(out_path).write_text("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Standalone lamella detection + yaml matching + nav export.")
    ap.add_argument("--mrc", required=True)
    ap.add_argument("--mdoc", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--weights", required=True, help="YOLO .pt weights file")
    ap.add_argument("--out", default="add_lamella_points.txt",
                    help="Output SerialEM script file")
    ap.add_argument("--map-item", type=int, default=1,
                    help="Navigator table index of the montage map")
    ap.add_argument("--no-flip-y", action="store_true",
                    help="Do not flip Y (use if points land mirrored vertically)")
    ap.add_argument("--prob-threshold", type=float, default=0.0)
    ap.add_argument("--max-residual", type=float, default=50.0)
    ap.add_argument("--no-flips", action="store_true")
    ap.add_argument("--save-png", default=None, help="Optionally save the model input image for inspection")
    args = ap.parse_args()

    print("[1/5] Building atlas montage ...")
    atlas, pix_A, origin_shift, frames, tile_shape, coord_key = \
        build_unstitched_montage(args.mrc, args.mdoc)
    print(f"      atlas {atlas.shape}, pixel size {pix_A:.2f} A")

    print("[2/5] Running YOLO lamella detection ...")
    detections = detect_lamellae(atlas, pix_A, args.weights,
                                 args.prob_threshold, args.save_png)
    if not detections:
        sys.exit("No lamellae detected.")
    print(f"      {len(detections)} detections")

    # For cluster matching, convert detection pixels to um so both point
    # clouds share a physical scale (matching handles rotation/flip/offset,
    # but not scale). No stage transform needed anymore - SerialEM does the
    # pixel->stage conversion itself via AddImagePosAsNavPoint.
    pix_um = pix_A / 1e4   # A/px -> um/px
    for d in detections:
        d["sx"], d["sy"] = d["px"] * pix_um, d["py"] * pix_um

    print("[3/5] Extracting lamella positions from experiment.yaml ...")
    yaml_lam = extract_yaml_lamellae(args.yaml)
    print(f"      {len(yaml_lam)} lamellae in YAML")

    print("[4/5] Cluster matching ...")
    pairs, R, rms = match_clusters([(d["sx"], d["sy"]) for d in detections],
                                   [(l["x"], l["y"]) for l in yaml_lam],
                                   try_flips=not args.no_flips)
    print(f"      RMS residual: {rms:.2f} um")
    if rms > args.max_residual:
        print(f"      WARNING: residual exceeds {args.max_residual} um - "
              "check coordinate frames / flips before trusting identities.")

    items = []
    for di, yi, res in sorted(pairs, key=lambda p: p[1]):
        d, l = detections[di], yaml_lam[yi]
        note = (f"{l['name']} | cat={d['cat']} p={d['prob']:.2f} "
                f"| match residual {res:.1f} um")
        print(f"      det px({d['px']:9.1f},{d['py']:9.1f}) -> {l['name']:<15} "
              f"res {res:6.1f} um  p={d['prob']:.2f}")
        items.append(dict(px=d["px"], py=d["py"], label=l["name"], note=note))

    for di in sorted(set(range(len(detections))) - {p[0] for p in pairs}):
        d = detections[di]
        items.append(dict(px=d["px"], py=d["py"], label="unmatched",
                          note=f"UNMATCHED detection cat={d['cat']} p={d['prob']:.2f}"))
        print(f"      det px({d['px']:9.1f},{d['py']:9.1f}) -> UNMATCHED")

    print("[5/5] Writing SerialEM script ...")
    write_serialem_script(items, args.out, map_item=args.map_item,
                          flip_y_height=None if args.no_flip_y else atlas.shape[0])
    print(f"      wrote {len(items)} AddImagePosAsNavPoint calls -> {args.out}")
    print("      In SerialEM: check --map-item points at the montage map in the")
    print("      Navigator, then run this script from the script editor.")

    summary = Path(args.out).with_suffix(".match.json")
    summary.write_text(json.dumps(dict(rms_um=rms, pairs=[
        dict(det=di, yaml=yi, name=yaml_lam[yi]["name"], residual_um=res)
        for di, yi, res in pairs]), indent=2))
    print(f"      match summary -> {summary}")


if __name__ == "__main__":
    main()