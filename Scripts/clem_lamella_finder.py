import numpy as np
from itertools import product
import math
import os
from PIL import Image
from pathlib import Path
try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

class LamellaFinder:
    def __init__(self, mrc_filepath, mrc_reader, tem):
        self.mrc_reader = mrc_reader
        self.tem = tem
        self.atlas = mrc_reader.build_montage_summary(mrc_filepath)

    def rescale_atlas(self):

        atlas_pix_nm = self.atlas.pixel_spacing_um * 1000
        model_pixel_spacing_nm = 400

        if atlas_pix_nm > model_pixel_spacing_nm:
            raise ValueError(
                f"Atlas pixel ({atlas_pix_nm:.1f} nm) larger than model pixel "
                f"(400 nm) - cannot downscale.")
        h, w = self.atlas.image.shape
        new_w = int(round(w * atlas_pix_nm / model_pixel_spacing_nm))
        new_h = int(round(h * atlas_pix_nm / model_pixel_spacing_nm))

        # normalize BEFORE resizing, ignoring empty (zero-filled) montage gaps
        a = self.atlas.image.astype(np.float64)
        mask = a != 0
        lo, hi = np.percentile(a[mask] if mask.any() else a, [1, 99])
        img8 = np.clip((a - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)

        img8 = np.array(Image.fromarray(img8).resize((new_w, new_h), Image.BOX))
        atlas_rgb = Image.fromarray(img8).convert("RGB")
        scale = (w / new_w, h / new_h)   
        return atlas_rgb

    def detect_lamellae(self, atlas_rgb, scale,prob_threshold=0.5, save_png=None):
  
        from ultralytics import YOLO
        model = YOLO(r"\model_weights\2024_07_26_lamella_detect_400nm_yolo8.pt")
        H, W = atlas_rgb.shape
        results = model(atlas_rgb, imgsz=(math.ceil(H / 32) * 32, math.ceil(W / 32) * 32), conf=prob_threshold or 0.25, verbose=False)[0]

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
            px = (x1 + x2) / 2 * scale[0]
            py = (y1 + y2) / 2 * scale[1]
            detections.append(dict(px=float(px), py=float(py), cat=cat, prob=float(prob)))
            boxes_for_overlay.append(((x1, y1, x2, y2), cat, prob))

        if save_png:
            from PIL import ImageDraw, ImageFont
            try:
                font = ImageFont.truetype("arial.ttf", 28)
            except OSError:
                font = ImageFont.load_default()
            draw = ImageDraw.Draw(atlas_rgb)
            for (x1, y1, x2, y2), cat, prob in boxes_for_overlay:
                label = f"{cat} {prob:.2f}"
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                tb = draw.textbbox((x1, max(y1 - 34, 0)), label, font=font)
                draw.rectangle(tb, fill=(255, 0, 0))
                draw.text((x1, max(y1 - 34, 0)), label, fill=(255, 255, 255), font=font)
            atlas_rgb.save(os.path(self.tem.output_root, "atlas_detected_lamellae.png"))

        detections_xy = [(d["px"] * self.atlas.pixel_spacing_um, d["py"] * self.atlas.pixel_spacing_um) for d in detections]
        return detections, detections_xy

    
    def extract_yaml_lamellae(self):
        import yaml
        with open(os.path.join(self.tem.output_root, "experiment.yaml"), encoding="utf-8", errors="replace") as fh:
            data = yaml.safe_load(fh)

        available_lamellae = []
        for pos in data.get("positions", []):
            defect = pos.get("defect") or {}
            defect_state = str(defect.get("state") or "NONE").upper()
            if defect_state != "NONE":
                continue

            poi = pos.get("poi") or {}
            sp = (((pos.get("poses") or {}).get("MILLING") or {})
                .get("stage_position") or {})
            sx = float(sp.get("x") or 0) * 1_000_000.0
            sy = float(sp.get("y") or 0) * 1_000_000.0
            sz = float(sp.get("z") or 0) * 1_000_000.0
            t = float(sp.get("t") or 0)                    # radians
            px = float(poi.get("x") or 0) * 1_000_000.0
            py = float(poi.get("y") or 0) * 1_000_000.0

            if pos.get("milling_angle") is not None and abs(math.cos(t)) > 1e-12:
                cx = sx + px
                cy = sy + py * math.cos(t)
                cz = sz + py * math.sin(t)
                x, y = cx, cy + cz * math.tan(t)
            else:
                x, y = sx + px, sy + py

            available_lamellae.append(dict(name=pos.get("petname", "") or f"pos{pos.get('number', '')}", number=pos.get("number", ""), x=x, y=y))
            available_lamellae_xy = [(lamella["x"], lamella["y"]) for lamella in available_lamellae]
        return available_lamellae, available_lamellae_xy

    def _assign(self, cost):
        if HAVE_SCIPY:
            return linear_sum_assignment(cost)
        ri, ci = [], []
        c = cost.copy()
        for _ in range(min(c.shape)):
            i, j = np.unravel_index(np.argmin(c), c.shape)
            ri.append(i); ci.append(j)
            c[i, :] = np.inf; c[:, j] = np.inf
        return np.array(ri), np.array(ci)


    def match_clusters(self, detections_xy, available_lamellae_xy, try_flips=True, angles_deg=range(0, 360, 5)):
        D = np.asarray(detections_xy, float)
        Y = np.asarray(available_lamellae_xy, float)
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
            ri, ci = self._assign(cost)
            rms = float(np.sqrt((cost[ri, ci] ** 2).mean()))
            if best is None or rms < best[0]:
                best = (rms, ri, ci, R, cost)

        rms, ri, ci, R, cost = best
        pairs = [(int(i), int(j), float(cost[i, j])) for i, j in zip(ri, ci)]
        return pairs, R, rms

    def run_id_lamellae(self):
        atlas_scaled, scale = self.rescale_atlas()
        lamella_centers, detections_xy = self.detect_lamellae(atlas_rgb=atlas_scaled, scale=scale)
        available_lamellae, available_lamellae_xy = self.extract_yaml_lamellae()
        pairs, R, rms = self.match_clusters(detections_xy=detections_xy, available_lamellae_xy=available_lamellae_xy)
        matched_lamellae = []
        for detection_index, yaml_index, residual_um in pairs:
            detection = lamella_centers[detection_index]
            lamella = available_lamellae[yaml_index]
            matched_lamellae.append({
                                        "name": lamella["name"],
                                        "px": detection["px"],
                                        "py": detection["py"],
                                        "category": detection["cat"],
                                        "probability": detection["prob"],
                                        "residual_um": residual_um,
                                    })
        for lamella in matched_lamellae:
            self.tem.load_mrc_in_nav(Path(self.mrc_filepath).name, buffer="M",)
            _, _, stage_z, _ = self.tem.get_stage_position()
            self.tem.add_nav_point_with_note(x=lamella["px"], y=lamella["py"], stage_z=stage_z, note=lamella["name"], buffer="M")
