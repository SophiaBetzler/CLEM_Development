import numpy as np
import matplotlib.pyplot as plt
import mrcfile
import os
import serialem as sem
### Some of the functionality is derived from spaceTomo to help transfer the coordinates to paceTomo

class CLEMPicker:


    MONTAGE_FLIP_X = False
    MONTAGE_FLIP_Y = True

    def __init__(self, montage, tem_communication):
        
        self.montage  = montage

        self.pix_um                 = montage["pixel_spacing_um"]
        self.coord_field            = montage.get("coord_field", "?")
        self.img_h, self.img_w      = montage["img_hw"]
        self.min_x, self.min_y      = montage["min_x"], montage["min_y"]
        self.image                  = montage["image"]
        self.position                   = montage["position"]
        self.path                   = montage["path"]
        self.tem                    = tem_communication
        self.H, self.W              = self.image.shape[:2]

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
    
    def _save_as_mrc_reference(self, crop, file_path):

        data = np.ascontiguousarray(crop, dtype=np.float32)
        pix_size_nm = self.pix_um * 1000.0                       # microns/px -> nm/px

        with mrcfile.new(str(file_path), overwrite=True) as mrc:
            mrc.set_data(data)
            mrc.voxel_size = (pix_size_nm * 10, pix_size_nm * 10, pix_size_nm * 10)  # Angstrom
            mrc.update_header_from_data()
        return file_path
    

    # ------------------------------------------------------------------ #
    # Pixel to stage conversion
    # ------------------------------------------------------------------ #

    def _pixel_to_stage_conversion(self, px, py, tile=None):
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
    # Correlation of target accross magnification
    # ------------------------------------------------------------------ #

    def _get_cropped_image_target_from_lmm(self, pick, fov):

        fov_px = tuple(f / self.pix_um for f in fov)      # (x, y) in pixels
        fov_x, fov_y = fov_px

        px, py = pick['px'], pick['py']

        x0_des, x1_des = int(px - fov_x / 2), int(px + fov_x / 2)   # cols = x
        y0_des, y1_des = int(py - fov_y / 2), int(py + fov_y / 2)   # rows = y

        print(f"Map dimensions: {self.image.shape}\nCrop coords:\nx: {x0_des}, {x1_des}\ny: {y0_des}, {y1_des}")

        y0, y1 = max(0, y0_des), min(self.image.shape[0], y1_des)
        x0, x1 = max(0, x0_des), min(self.image.shape[1], x1_des)

        image_crop = self.image[y0:y1, x0:x1].astype(np.float32, copy=True)

        saturated = image_crop >= 1.0
        valid = image_crop[~saturated]
        mean = float(valid.mean()) if valid.size else 0.0
        image_crop[saturated] = mean

        pad_top    = max(0, -y0_des)
        pad_bottom = max(0, y1_des - self.image.shape[0])
        pad_left   = max(0, -x0_des)
        pad_right  = max(0, x1_des - self.image.shape[1])

        if pad_top or pad_bottom or pad_left or pad_right:
            image_crop = np.pad(image_crop,((pad_top, pad_bottom), (pad_left, pad_right)),   mode="constant", constant_values=mean,)
            print("WARNING: Target position is close to the edge of the map and was padded.")
        image_crop_flip = self._flip(image_crop)
        img_crop_path = os.path.join(self.path + '_' + 'pick_reference' + '_' + str(pick['pick_id']) +   '.mrc')
        self._save_as_mrc_reference(image_crop_flip, img_crop_path)

        plt.imshow(image_crop_flip)
        plt.show()

        return image_crop, image_crop_flip
    
    def _get_cropped_image_target_from_preset(self, fov, pick_presets):

        with mrcfile.open(os.path.join(self.path + f"_pick_{pick_presets['pick_id']}_{pick_presets['mode']}"), permissive=True) as mrc:
            image = np.asarray(mrc.data, dtype=np.float32)
            pix_um = mrc.voxel_size.x / 10000.0

        fov_px = tuple(f / pix_um for f in fov)      # (x, y) in pixels
        fov_x, fov_y = fov_px

        x0_des, x1_des = int(fov_x / 2), int(fov_x / 2)   # cols = x
        y0_des, y1_des = int(fov_y / 2), int(fov_y / 2)   # rows = y

        print(f"Map dimensions: {image.shape}\nCrop coords:\nx: {x0_des}, {x1_des}\ny: {y0_des}, {y1_des}")

        y0, y1 = max(0, y0_des), min(image.shape[0], y1_des)
        x0, x1 = max(0, x0_des), min(image.shape[1], x1_des)

        image_crop = image[y0:y1, x0:x1].astype(np.float32, copy=True)

        saturated = image_crop >= 1.0
        valid = image_crop[~saturated]
        mean = float(valid.mean()) if valid.size else 0.0
        image_crop[saturated] = mean

        pad_top    = max(0, -y0_des)
        pad_bottom = max(0, y1_des - image.shape[0])
        pad_left   = max(0, -x0_des)
        pad_right  = max(0, x1_des - image.shape[1])

        if pad_top or pad_bottom or pad_left or pad_right:
            image_crop = np.pad(image_crop,((pad_top, pad_bottom), (pad_left, pad_right)),   mode="constant", constant_values=mean,)
            print("WARNING: Target position is close to the edge of the map and was padded.")
        image_crop_flip = self._flip(image_crop)
        img_crop_path = os.path.join(self.path + '_' + 'pick_reference' + '_' + str(pick_presets['pick_id']) + '_2' +   '.mrc')
        self._save_as_mrc_reference(image_crop_flip, img_crop_path)

        plt.imshow(image_crop_flip)
        plt.show()

        return image_crop, image_crop_flip
    
    def _align_target_at_higher_mag(self, pick_id, target_stage_pos, mode):

        buffer = 'P' # this doesn't get rolled over so the reference stays here
        pick_path = os.path.join(self.path + '_' + 'pick_reference' + '_' + str(pick_id) + '.mrc')
        self.tem._load_mrc_in_nav(pick_path, buffer=buffer)
        self.tem.precise_stage_move(stage_position=target_stage_pos)

        (live_img_X, live_img_Y, live_img_px) = self.tem.acquire_image(mode=mode)
        
        if live_img_X*live_img_px > self.img_h:
            raise ValueError("Magnficiation mismatch the reference image must have a wider field of view than the higher mag image it should be correlated to.")
        
        print('[INFO] Running serialEM alignment routine.')
        self.tem.run_serialem_alignment_routine(buffer=buffer, pick_id=pick_id, mode=mode)
        self.tem.acquire_image(mode=mode)
        self.tem.image_from_buffer()

    # ------------------------------------------------------------------ #
    # Group Picks and Define Target
    # ------------------------------------------------------------------ #
    def group_picks(self, radius_um=5.0, lone_offset_um=2.5):
    
        if not self.pick_fine_aligned:
            return []

        stage_position = np.array([(pick["stage"], pick["stage_z"]) for pick in self.pick_fine_aligned], float)     
        n = len(self.pick_fine_aligned)
        assigned = [False] * n

        order = sorted(range(n),
                    key=lambda i: -int(np.sum(np.linalg.norm(stage_position - stage_position[i], axis=1) <= radius_um)))

        groups = []
        for seed in order:
            if assigned[seed]:
                continue
            members = [i for i in range(n)
                    if not assigned[i] and np.linalg.norm(stage_position[i] - stage_position[seed]) <= radius_um]
            # refine: drop members outside radius of the moving centroid
            for _ in range(10):
                centre = pos[members].mean(axis=0)
                kept = [i for i in members if np.linalg.norm(pos[i] - centre) <= radius_um]
                if len(kept) == len(members):
                    break
                members = kept if kept else [seed]
            for i in members:
                assigned[i] = True

            grp = [self.picks[i] for i in members]

            if len(grp) > 1:
                cpx = float(np.mean([p["px"] for p in grp]))
                cpy = float(np.mean([p["py"] for p in grp]))
        else:
            off_px = lone_offset_um / self.pix_um            # um -> montage pixels
            min_sep_px = min_target_sep_um / self.pix_um     # required clearance, pixels
            lone = grp[0]

            # direction toward the centroid of all OTHER picks (pixel space)
            others = [p for p in self.picks if p is not lone]
            if others:
                ocx = np.mean([p["px"] for p in others])
                ocy = np.mean([p["py"] for p in others])
                vec = np.array([ocx - lone["px"], ocy - lone["py"]], float)
                norm = np.linalg.norm(vec)
                direction = vec / norm if norm > 1e-6 else np.array([1.0, 0.0])
            else:
                direction = np.array([1.0, 0.0])

            all_px = np.array([[p["px"], p["py"]] for p in self.picks], float)

            def _clear(cx, cy):
                """True if (cx, cy) is at least min_sep_px from every pick."""
                return np.all(np.linalg.norm(all_px - np.array([cx, cy]), axis=1) >= min_sep_px)

            # try the requested offset; if it overlaps a pick, push further out,
            # then rotate the direction, until a clear spot is found
            cpx, cpy = lone["px"] + direction[0] * off_px, lone["py"] + direction[1] * off_px
            if not _clear(cpx, cpy):
                found = False
                for scale in (1.5, 2.0, 2.5, 3.0):           # push outward first
                    cx = lone["px"] + direction[0] * off_px * scale
                    cy = lone["py"] + direction[1] * off_px * scale
                    if _clear(cx, cy):
                        cpx, cpy, found = cx, cy, True
                        break
                if not found:                                 # then sweep the angle
                    for deg in range(30, 360, 30):
                        a = np.deg2rad(deg)
                        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ direction
                        cx = lone["px"] + rot[0] * off_px
                        cy = lone["py"] + rot[1] * off_px
                        if _clear(cx, cy):
                            cpx, cpy, found = cx, cy, True
                            break
                if not found:
                    print(f"WARNING: could not place tracking target clear of all picks "
                          f"for pick {lone.get('pick_id')}; using best-effort position.")

            sx, sy, tile = self._pixel_to_stage_conversion(cpx, cpy)
            groups.append({"tracking": {"px": cpx, "py": cpy, "stage": (sx, sy), "stage_z": tile["stage_z"]}, "picks": grp,})
        return groups

    def write_pacetomo_group(self, group, user_name):

        #### ADD REFERENCE IMAGE LOGIC AND FIX EUCENTIC LOGIC
        #### GO THROUGH WHOLE SCRIPT AGAIN TO MAKE SURE THAT IT MAKES SENSE

        # --- origin = tracking centre's specimen shift (captured after centring it) ---
        SS0 = np.array(group["tracking"]["specimen_shift"])      # absolute, microns

        ordered = [group["tracking"]] + group["picks"]           # target 0 first

        tgts_path = os.path.join(self.path, user_name + "_tgts.txt")
        blocks = []
        for i, t in enumerate(ordered):
            nnn = str(i + 1).zfill(3)
            SS = np.array(t["specimen_shift"]) - SS0             # relative to tracking
            sx, sy = t["stage"]
            tgtfile = os.path.basename(t["tgtfile"])             # reference saved per target
            blocks.append(
                f"_tgt = {nnn}\n"
                f"tgtfile = {tgtfile}\n"
                f"tsfile = {user_name}_ts_{nnn}.mrc\n"
                f"SSX = {SS[0]}\n"
                f"SSY = {SS[1]}\n"
                f"stageX = {sx}\n"
                f"stageY = {sy}\n"
                f"skip = False\n\n"
            )
        with open(tgts_path, "w") as f:
            f.write("".join(blocks))

        # --- Navigator anchor: tracking reference -> map item, Acquire, note = tgts file ---
        self.tem._load_mrc_in_nav(group["tracking"]["tgtfile"], buffer="A")
        map_index = int(sem.NewMap(0, user_name + "_tgts.txt"))
        sem.SetItemAcquire(map_index)
        sem.ChangeItemLabel(map_index, "001")
        return tgts_path

    # ------------------------------------------------------------------ #
    # Picker
    # ------------------------------------------------------------------ #
    
    def run_auto_picker(self):
        self.picker()
        os.makedirs(os.path.join(self.path, self.position), exist_ok=True)
        self.tem.add_stage_pos_to_nav(picks=self.picks)
        groups = self.group_picks(radius_um=5.0)

        for g_i, group in enumerate(groups):
            user_name = f"{self.position}_area{g_i+1}"
            center = group["tracking"]
            center["pick_id"] = f"{user_name}_track"
            self._get_cropped_image_target_from_lmm(center, (10, 10)) 
            center_summary = self._align_target_at_higher_mag(pick_id=center["pick_id"], target_stage_pos=(center["stage"][0], center["stage"][1], center["stage_z"]), mode="View", eucentric=True)
            img_crop, img_crop_flipped = self._get_cropped_image_target_from_preset(center, (2, 2))
            center_summary_high_mag = self._align_target_at_higher_mag(pick_id=f"{center['pick_id']}_2", target_stage_pos=(center['stage'][0], pick['stage'][1], pick['stage_z']), presets=center_summary, mode='Record', eucentric=False)
            center.update(center_summary_high_mag)
            self.tem.acquire_image(mode='View', )

        for pick in group['picks']:
            img_crop, img_crop_flipped = self._get_cropped_image_target_from_lmm(pick, (10, 10))
            pick_summary = self._align_target_at_higher_mag(pick_id=pick['pick_id'], target_stage_pos=(pick['stage'][0], pick['stage'][1], pick['stage_z']), mode='View', eucentric=True)
            img_crop, img_crop_flipped = self._get_cropped_image_target_from_preset(pick, (2, 2))
            pick_summary_high_mag = self._align_target_at_higher_mag(pick_id=f"{pick['pick_id']}_2", target_stage_pos=(pick['stage'][0], pick['stage'][1], pick['stage_z']), presets=pick_summary, mode='Record', eucentric=False)
            pick.update(pick_summary_high_mag)

            ref_dir = os.path.join(self.path, "targets")
            os.makedirs(ref_dir, exist_ok=True)

            self.tem.acquire(mode='View', lowdose=True, save=True, position=None, token=pick['pick_id'])
            self.tem.acquire(mode='Record', lowdose=True, save=True, position=None, token=pick['pick_id'])

         


        self.write_pacetomo_group()
        

    def picker(self):

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
            pick_id = len(self.picks) + 1
            stage_x, stage_y, tile = self._pixel_to_stage_conversion(px, py)
            self.picks.append({"pick_id": pick_id, "px": px, "py": py, "z": tile["z"], "stage_z": tile["stage_z"], "stage": (stage_x, stage_y)})
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

