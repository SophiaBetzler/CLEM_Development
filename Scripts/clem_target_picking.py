import numpy as np
import os


class CLEMPicker:


    MONTAGE_FLIP_X = False
    MONTAGE_FLIP_Y = True

    def __init__(self, montage):
        self.montage  = montage

        self.pix_um          = montage["pixel_spacing_um"]
        self.coord_field      = montage.get("coord_field", "?")
        self.img_h, self.img_w = montage["img_hw"]
        self.min_x, self.min_y = montage["min_x"], montage["min_y"]
        self.image           = montage["image"]
        self.H, self.W       = self.image.shape[:2]

        theta = np.deg2rad(montage["rotation_deg"])
        self.cos, self.sin = np.cos(theta), np.sin(theta)

        self.tiles = self._build_lookup()
        self.picks = []
        M = montage.get("stage_matrix")
        self._M = np.asarray(M, float) if M is not None else None


    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #

    def _flip(self, arr):
        if self.MONTAGE_FLIP_X: arr = np.fliplr(arr)
        if self.MONTAGE_FLIP_Y: arr = np.flipud(arr)
        return arr

    def _unflip_x(self, x): 
        return (self.W - 1 - x) if self.MONTAGE_FLIP_X else x

    def _unflip_y(self, y): 
        return (self.H - 1 - y) if self.MONTAGE_FLIP_Y else y

    def _build_lookup(self):

        out = []
        for piece in self.montage["tiles"]:
            print(self.montage["tiles"])
            px = piece.get("px")
            stage = piece.get("stage")
            if px is None or stage is None:
                continue
            out.append({
                        "z": piece.get("z"), "stage_z": piece.get("stage_z"),
                        "cx": float(px[0]) - self.min_x + self.img_w / 2.0, 
                        "cy": float(px[1]) - self.min_y + self.img_h / 2.0,
                        "sx": float(stage[0]), "sy": float(stage[1])
                    })
        
        if not out:
            raise ValueError(f"No usable pieces (each needs a pixel origin and a StagePosition.).")
        return out

    def _piece_for(self, px, py):
        inside = [t for t in self.tiles
                  if t["cx"] - self.img_w / 2 <= px < t["cx"] + self.img_w / 2
                  and t["cy"] - self.img_h / 2 <= py < t["cy"] + self.img_h / 2]
        pool = inside if inside else self.tiles
        return min(pool, key=lambda t: (px - t["cx"]) ** 2 + (py - t["cy"]) ** 2)

    def _auto_brightness_contrast(self, img, percentiles=(1.0, 99.8), ignore_zeros=True, ignore_whites=True, white_cutoff=0.995):

        img = np.nan_to_num(img.astype(np.float32))
        sample = img[np.isfinite(img)]

        mask = np.ones(sample.shape, dtype=bool)
        if ignore_zeros:
            mask &= sample > 0
        if ignore_whites:
            mask &= sample < white_cutoff

        trimmed = sample[mask]
        if trimmed.size:
            sample = trimmed

        if not sample.size:
            return np.zeros_like(img)

        lo, hi = np.percentile(sample, percentiles)
        if hi <= lo:
            lo, hi = sample.min(), sample.max()
        if hi <= lo:
            return np.zeros_like(img)

        return np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    

    # ------------------------------------------------------------------ #
    # Pixel to stage conversion
    # ------------------------------------------------------------------ #

    def pixel_to_stage_conversion(self, px, py, tile=None):
        if tile is None:
            tile = self._piece_for(px, py)
        
        dx, dy = px - tile["cx"], py - tile["cy"]

        if self._M is not None:
            stage_x = tile['sx'] + self._M[0, 0] * dx + self._M[0, 1] * dy
            stage_y = tile['sy'] + self._M[1, 0] * dx + self._M[1, 1] * dy
        else:
            dx_um, dy_um = dx * self.pix_um, dy * self.pix_um
            stage_x = tile["sx"] + (self.cos * dx_um - self.sin * dy_um)
            stage_y = tile["sy"] + (self.sin * dx_um + self.cos * dy_um)
        

        return stage_x, stage_y, tile 
    
    # ------------------------------------------------------------------ #
    # Picker
    # ------------------------------------------------------------------ #
    
    def picker(self):
        import matplotlib.pyplot as plt
        max_px = 2000
        ds   = max(1, max(self.H, self.W) // max_px)
        disp = self.image[::ds, ::ds] if ds > 1 else self.image
        disp = self._flip(disp)          # display-only flip (+Y up, SerialEM)
        disp = self._auto_brightness_contrast(disp)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(disp, cmap="gray", origin="upper", aspect="equal", vmin=0.0, vmax=1.0, extent=[-0.5, self.W - 0.5, self.H - 0.5, -0.5])
        ax.set_title(f"Left-click to pick; close when done")
        ax.axis("off")

        def on_click(event):
            if event.button != 1 or event.inaxes is not ax or event.xdata is None:
                return
            cpx, cpy = event.xdata, event.ydata
            px = self._unflip_x(cpx)    
            py = self._unflip_y(cpy)
           
            stage_x, stage_y, tile = self.pixel_to_stage_conversion(px, py)
            self.picks.append({"px": px, "py": py, "z": tile["z"], "stage_z": tile["stage_z"], "stage": (stage_x, stage_y)})
            print(f"point {len(self.picks)}: tile z={tile['z']}  stage=({stage_x:.3f}, {stage_y:.3f}) um  stageZ={tile['stage_z']}")

            ax.plot(cpx, cpy, "+", color="cyan", markersize=12, markeredgewidth=1.5)
            ax.text(cpx + 8, cpy - 8, str(len(self.picks)), color="cyan", fontsize=9, fontweight="bold")
            ax.annotate(f"{stage_x:.1f}, {stage_y:.1f}", (cpx, cpy), xytext=(cpx + 8, cpy + 18), color="yellow", fontsize=7)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", on_click)
        plt.tight_layout()
        plt.show()
        return self.picks


if __name__ == "__main__":

    PATH = "C:\\Users\\CZII\\Desktop\\Test_Data\\"
    FILENAME = "12-chief-dog_montage_20260616-07-47-11.mrc"

