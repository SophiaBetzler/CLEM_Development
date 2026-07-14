# ============================================================================
# COMPLETE CLEMPICKER CLASS - ALL REQUIRED FUNCTIONS
# ============================================================================
# Comprehensive implementation with all utilities and workflow methods

import numpy as np
import matplotlib.pyplot as plt
import mrcfile
import os
from clem_dataclasses import Pick, TargetGroup
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TileCenter:
    cx: float
    cy: float
    sx: float
    sy: float
    z_index: Optional[int] = None
    stage_z_um: Optional[float] = None


class CLEMPicker:

    def __init__(self, site_data, tem_communication):

        self.site_data = site_data
        mrc_summary = site_data.mrc

        self.pixel_spacing_um = mrc_summary.pixel_spacing_um
        self.coord_field = mrc_summary.coord_field
        self.image_height, self.image_width = mrc_summary.image_height, mrc_summary.image_width
        self.min_x_pixels, self.min_y_pixels = mrc_summary.min_x_pixels, mrc_summary.min_y_pixels
        self.image = mrc_summary.image
        self.mrc = mrc_summary
        self.output_coord_mode = "stage" # or "image"
        self.nav_map_buffer = "A"

        self.site_id = site_data.site_id
        self.site_output_root = site_data.path
        self.tem = tem_communication
        self.H, self.W = self.image.shape[:2]
        
        self.flip_x = bool(getattr(mrc_summary, 'flip_x', False))
        self.flip_y = bool(getattr(mrc_summary, 'flip_y', False))

        theta = np.deg2rad(getattr(mrc_summary, 'rotation_deg', 0.0))
        self.cos, self.sin = np.cos(theta), np.sin(theta)

        self.tiles = self._build_lookup()

        if mrc_summary.stage_matrix is not None:
            self._M = np.asarray(mrc_summary.stage_matrix, dtype=float)
        else:
            self._M = None
             
    # ════════════════════════════════════════════════════════════════════════
    # Helper and Coordinate Conversion Methods
    # ════════════════════════════════════════════════════════════════════════

    def _build_lookup(self) -> List[TileCenter]:

        out = []
        for t in self.mrc.tiles:
            if t.pixel_x_um is None or t.stage_x_um is None:
                continue
            out.append(TileCenter(
                z_index=t.z_index,
                stage_z_um=t.stage_z_um,
                cx=t.pixel_x_um - self.min_x_pixels + self.image_width / 2.0,
                cy=t.pixel_y_um - self.min_y_pixels + self.image_height / 2.0,
                sx=t.stage_x_um,
                sy=t.stage_y_um,
            ))
        
        if not out:
            raise ValueError("No usable tiles (each needs pixel origin and stage position).")
        return out

    def _piece_for(self, px: float, py: float) -> TileCenter:    
        inside = [t for t in self.tiles
                  if t.cx - self.image_width / 2 <= px < t.cx + self.image_width / 2
                  and t.cy - self.image_height / 2 <= py < t.cy + self.image_height / 2]
        pool = inside if inside else self.tiles
        return min(pool, key=lambda t: (px - t.cx) ** 2 + (py - t.cy) ** 2)

    def convert_display_to_montage_orientation(self, dx: float, dy: float) -> Tuple[float, float]:
        """Convert display coordinates to montage coordinates."""
        px = (self.W - 1 - dx) if self.flip_x else dx
        py = (self.H - 1 - dy) if self.flip_y else dy
        return px, py

    def convert_montage_to_display_orientation(self, px: float, py: float) -> Tuple[float, float]:
        """Convert montage coordinates to display coordinates."""
        dx = (self.W - 1 - px) if self.flip_x else px
        dy = (self.H - 1 - py) if self.flip_y else py
        return dx, dy

    def _pixel_to_stage_conversion(self, px: float, py: float, 
                                  tile: Optional[TileCenter] = None) -> Tuple[float, float, TileCenter]:
        """Convert pixel coordinates to stage coordinates using transformation matrix or rotation."""
        
        if tile is None:
            tile = self._piece_for(px, py)
        
        dx, dy = px - tile.cx, py - tile.cy
        
        if self._M is not None:
            stage_x = tile.sx + self._M[0, 0] * dx + self._M[0, 1] * dy
            stage_y = tile.sy + self._M[1, 0] * dx + self._M[1, 1] * dy
        else:
            dx_um, dy_um = dx * self.pixel_spacing_um, dy * self.pixel_spacing_um
            stage_x = tile.sx + (self.cos * dx_um - self.sin * dy_um)
            stage_y = tile.sy + (self.sin * dx_um + self.cos * dy_um)
        
        return stage_x, stage_y, tile
    
    def set_output_coord_mode(self, mode: str, buffer=None):
        if mode not in ("stage", "image"):
            raise ValueError("mode must be 'stage' or 'image'")
        self.output_coord_mode = mode
        if buffer is not None:
            self.nav_map_buffer = buffer

    # ════════════════════════════════════════════════════════════════════════
    # Pick Creation and Management
    # ════════════════════════════════════════════════════════════════════════

    def make_pick_dataclass(self, px: float, py: float, pick_id: Optional[str] = None) -> Pick:
        """Create a Pick dataclass from pixel coordinates."""
        
        stage_x, stage_y, tile = self._pixel_to_stage_conversion(px, py)
        
        if pick_id is None:
            pick_id = str(len(self.site_data.picks) + 1)
        
        return Pick(
            pick_id=str(pick_id),
            pixel_x_um=px,
            pixel_y_um=py,
            stage_x_um=stage_x,
            stage_y_um=stage_y,
            stage_z_um=tile.stage_z_um,
        )

    def add_pick_from_pixel(self, px: float, py: float, pick_id: Optional[str] = None) -> Pick:
        """Add pick from pixel coordinates and return it."""
        
        pick = self.make_pick_dataclass(px, py, pick_id=pick_id)
        self.site_data.picks.append(pick)
        return pick

    def add_pick_from_display(self, dx: float, dy: float, pick_id: Optional[str] = None) -> Pick:
        """Add pick from display coordinates."""
        
        px, py = self.convert_display_to_montage_orientation(dx, dy)
        if hasattr(self.tem, 'display_pick_in_nav'):
            self.tem.display_pick_in_nav()
        return self.add_pick_from_pixel(px, py, pick_id=pick_id)

    def remove_last_pick(self) -> Optional[Pick]:
        """Remove and return last pick."""
        
        if self.site_data.picks:
            return self.site_data.picks.pop()
        return None

    def clear_picks(self) -> None:
        """Clear all picks."""
        self.site_data.picks.clear()

    def get_picks_count(self) -> int:
        """Get number of picks."""
        return len(self.site_data.picks)

    def get_all_picks(self) -> List[Pick]:
        """Get all picks."""
        return self.site_data.picks.copy()
    
    def _add_nav_point(self, pick, group_id, label):
        nav_idx = self.tem.add_nav_point(self, pick=pick, group_id=group_id, label=label, output_coord_mode = self.output_coord_mode)
        return nav_idx
    
    def add_group_to_navigator(self, group: TargetGroup, record_mag: int) -> Dict[str, int]:

        target = group.tracking
        nav_indices = {target.pick_id: self.tem.add_nav_point(pick=target, label=f"001_{target.pick_id}")}
        for i, pick in enumerate(group.picks):
            if pick.pick_id == target.pick_id:
                continue
            nav_indices[pick.pick_id] = self.tem.add_nav_point(pick=pick, label=f"{str(i+2).zfill(3)}_{pick.pick_id}")
        
        return nav_indices


    # ════════════════════════════════════════════════════════════════════════
    # Image Cropping and Preprocessing
    # ════════════════════════════════════════════════════════════════════════

    def crop_montage_around_pick(self, pick: Pick, fov_um: Tuple[float, float]) -> np.ndarray:
        
        fov_x = fov_um[0] / self.pixel_spacing_um
        fov_y = fov_um[1] / self.pixel_spacing_um
        
        px, py = pick.pixel_x_um, pick.pixel_y_um
        x0_des, x1_des = int(px - fov_x / 2), int(px + fov_x / 2)
        y0_des, y1_des = int(py - fov_y / 2), int(py + fov_y / 2)
        
        y0, y1 = max(0, y0_des), min(self.image.shape[0], y1_des)
        x0, x1 = max(0, x0_des), min(self.image.shape[1], x1_des)
        crop = self.image[y0:y1, x0:x1].astype(np.float32, copy=True)

        saturated = crop >= 1.0
        valid = crop[~saturated]
        mean = float(valid.mean()) if valid.size else 0.0
        crop[saturated] = mean

        pad = ((max(0, -y0_des), max(0, y1_des - self.image.shape[0])),
               (max(0, -x0_des), max(0, x1_des - self.image.shape[1])))
        if any(sum(p) for p in pad):
            crop = np.pad(crop, pad, mode="constant", constant_values=mean)
            print("[WARN] Target near montage edge; crop was padded.")
        
        return crop

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        return image

    def display_crop(self, crop: np.ndarray, title: str = "Crop") -> None:

        plt.figure(figsize=(6, 6))
        plt.imshow(crop, cmap='gray')
        plt.title(title)
        plt.colorbar()
        plt.show()

    # ════════════════════════════════════════════════════════════════════════
    # Pick Grouping and Tracking Target Selection
    # ════════════════════════════════════════════════════════════════════════

    def group_picks(self, radius_um: float = 7.5, lone_offset_um: float = 2.5, 
                   min_seperation_um: Optional[float] = None) -> List[TargetGroup]: 
        picks = self.site_data.picks
        if not picks:
            return []
        
        if min_seperation_um is None:
            min_seperation_um = lone_offset_um

        XY = np.array([[p.stage_x_um, p.stage_y_um] for p in picks], dtype=float)
        n = len(picks)
        assigned = [False] * n

        neigh = [int(np.sum(np.linalg.norm(XY - XY[i], axis=1) <= radius_um)) for i in range(n)]
        order = sorted(range(n), key=lambda i: -neigh[i])
        
        groups = []
        
        for seed in order:
            if assigned[seed]:
                continue

            members = [i for i in range(n) if not assigned[i] and np.linalg.norm(XY[i] - XY[seed]) <= radius_um]

            for _ in range(10):
                centre = XY[members].mean(axis=0)
                kept = [i for i in members if np.linalg.norm(XY[i] - centre) <= radius_um]
                if len(kept) == len(members):
                    break
                members = kept if kept else [seed]

            for i in members:
                assigned[i] = True

            group_id = str(len(groups) + 1)
            member_picks = [picks[i] for i in members]
            tracking_target = self._tracking_target_for(member_picks, picks, lone_offset_um, 
                                                       min_seperation_um, group_id)
            
            groups.append(TargetGroup(group_id=group_id, tracking=tracking_target, picks=member_picks))
        
        print(f"\n[INFO] Created {len(groups)} groups from {n} picks")
        return groups

    def create_tracking_target_for_group(self, group: List[Pick], all_picks: List[Pick], 
                            lone_offset_um: float, min_sep_um: float, 
                            group_id: str) -> Pick:

        if len(group) > 2:
            cx = float(np.mean([p.stage_x_um for p in group]))
            cy = float(np.mean([p.stage_y_um for p in group]))
            return min(group, key=lambda p: (p.stage_x_um - cx) ** 2 + (p.stage_y_um - cy) ** 2)
        
        if len(group) == 2:
            return group[0]

        lone = group[0]
        track_id = f"group{group_id}_track"
        off_px = lone_offset_um / self.pixel_spacing_um
        min_sep_px = min_sep_um / self.pixel_spacing_um

        others = [p for p in all_picks if p is not lone]
        if others:
            ocx = float(np.mean([p.pixel_x_um for p in others]))
            ocy = float(np.mean([p.pixel_y_um for p in others]))
            vec = np.array([ocx - lone.pixel_x_um, ocy - lone.pixel_y_um], float)
            nrm = np.linalg.norm(vec)
            direction = vec / nrm if nrm > 1e-6 else np.array([1.0, 0.0])
        else:
            direction = np.array([1.0, 0.0])

        all_xy = np.array([[p.pixel_x_um, p.pixel_y_um] for p in all_picks], float)
        
        def clear(cx, cy):
            return bool(np.all(np.linalg.norm(all_xy - np.array([cx, cy]), axis=1) >= min_sep_px))
        
        cpx = lone.pixel_x_um + direction[0] * off_px
        cpy = lone.pixel_y_um + direction[1] * off_px
        
        if not clear(cpx, cpy):
            placed = False
            # Try scaling outward
            for scale in (1.5, 2.0, 2.5, 3.0):
                cx = lone.pixel_x_um + direction[0] * off_px * scale
                cy = lone.pixel_y_um + direction[1] * off_px * scale
                if clear(cx, cy):
                    cpx, cpy, placed = cx, cy, True
                    break
            
            if not placed:
                # Try rotating direction
                for deg in range(30, 360, 30):
                    a = np.deg2rad(deg)
                    rot = np.array([[np.cos(a), -np.sin(a)],
                                   [np.sin(a), np.cos(a)]]) @ direction
                    cx = lone.pixel_x_um + rot[0] * off_px
                    cy = lone.pixel_y_um + rot[1] * off_px
                    if clear(cx, cy):
                        cpx, cpy, placed = cx, cy, True
                        break
            
            if not placed:
                print(f"[WARN] Could not place tracking target clear of all picks for group {group_id}")
        
        return self.make_pick_dataclass(cpx, cpy, pick_id=track_id)

    # ════════════════════════════════════════════════════════════════════════
    # Reference Image Extraction
    # ════════════════════════════════════════════════════════════════════════

    def extract_reference_crops_for_group(self, group: TargetGroup, 
                                        crop_fov_um: float = 5.0,
                                        output_subfolder: str = 'references') -> Dict[str, str]:
        
        print(f"\n[INFO] Extracting reference crops for group {group.group_id}...")
        
        output_folder = os.path.join(self.site_output_root, output_subfolder)
        os.makedirs(output_folder, exist_ok=True)
        
        pick_crops = {}
        
        for pick in group.picks:
            if pick.pick_id == group.tracking.pick_id:
                continue
            
            crop = self.crop_montage_around_pick(pick, fov=(crop_fov_um, crop_fov_um))
            
            crop_filename = f"{pick.pick_id}_crop_montage.mrc"
            crop_path = os.path.join(output_folder, crop_filename)
            
            with mrcfile.new(crop_path, overwrite=True) as mrc:
                mrc.set_data(crop)
                mrc.voxel_size = self.pixel_spacing_um * 10000  # nm to Angstroms
                mrc.update_header_from_data()
            
            pick_crops[pick.pick_id] = crop_path
            print(f"  Saved: {crop_filename}")
        
        return pick_crops

    # ════════════════════════════════════════════════════════════════════════
    # Target Refinement
    # ════════════════════════════════════════════════════════════════════════

    def refine_target_stage_position(self, target_pick: Pick,
                                    montage_mag: int = 2000,
                                    record_mag: int = 40000,
                                    view_mag: int = 15000,
                                    output_subfolder: str = 'references') -> Pick:
        
        print(f"\n[INFO] Refining target {target_pick.pick_id}...")
        
        output_folder = os.path.join(self.site_output_root, output_subfolder)
        os.makedirs(output_folder, exist_ok=True)
        
        # Create reference crop at montage
        montage_crop = self.crop_montage_around_pick(target_pick, fov=(5.0, 5.0))
        
        montage_ref_path = os.path.join(output_folder,
                                       f"{target_pick.pick_id}_crop_montage_reference.mrc")
        with mrcfile.new(montage_ref_path, overwrite=True) as mrc:
            mrc.set_data(montage_crop)
            mrc.voxel_size = self.pixel_spacing_um * 10000
            mrc.update_header_from_data()
        
        # Align at higher magnification
        alignment_result = self.align_target_at_higher_mag(
            pick_id=target_pick.pick_id,
            target_stage_pos=(target_pick.stage_x_um, target_pick.stage_y_um, target_pick.stage_z_um),
            reference_image_path=montage_ref_path,
            mode='Record'
        )
        
        refined_x, refined_y, refined_z = alignment_result['refined_stage']
        
        # Save record image
        record_filename = f"{target_pick.pick_id}_record_{record_mag}x.mrc"
        record_path = os.path.join(output_folder, record_filename)
        self.tem.save_image_from_buffer(output_path=record_path, buffer='B')
        
        # Save view crop
        self.tem.set_magnification(view_mag)
        self.tem.acquire_image(mode='View')
        
        view_image = self.tem.get_image_from_buffer(buffer='A')
        img_props = self.tem.get_image_properties()
        
        fov_px = int(5.0 / img_props['pixel_size_um'])
        center_y, center_x = view_image.shape[0] // 2, view_image.shape[1] // 2
        
        y0 = max(0, center_y - fov_px // 2)
        y1 = min(view_image.shape[0], center_y + fov_px // 2)
        x0 = max(0, center_x - fov_px // 2)
        x1 = min(view_image.shape[1], center_x + fov_px // 2)
        
        view_crop = view_image[y0:y1, x0:x1].astype(np.float32)
        
        view_filename = f"{target_pick.pick_id}_view_{view_mag}x.mrc"
        view_path = os.path.join(output_folder, view_filename)
        
        with mrcfile.new(view_path, overwrite=True) as mrc:
            mrc.set_data(view_crop)
            mrc.voxel_size = img_props['pixel_size_um'] * 10000
            mrc.update_header_from_data()
        
        # UPDATE Pick object with refinement data
        target_pick.refined_stage_x = refined_x
        target_pick.refined_stage_y = refined_y
        target_pick.refined_stage_z = refined_z
        target_pick.record_img_path = record_path
        target_pick.view_crop_path = view_path
        target_pick.is_tracking_target = True
        target_pick.refinement_quality = 'good'
        
        return target_pick

    def align_target_at_higher_mag(self, pick_id: str, target_stage_pos: Tuple[float, float, float],
                                  reference_image_path: str, mode: str = 'Record') -> Dict:
        
        buffer = 'P'  # Persistent buffer
        
        print(f"[INFO] Loading reference and aligning {pick_id}...")

        self.tem._load_mrc_in_nav(reference_image_path, buffer=buffer)

        self.tem.precise_stage_move(tage_x_um=target_stage_pos[0], stage_y_um=target_stage_pos[1], stage_z_um=target_stage_pos[2])

        self.tem.acquire_image(mode=mode)
        
        # Run alignment routine (includes AlignBetweenMags + fine refinement)
        print('[INFO] Running SerialEM alignment routine.')
        self.tem.run_serialem_alignment_routine(buffer=buffer,
            pick_id=pick_id,
            mode=mode,
            mag_compensation=True
        )
        
        # Get refined stage position
        refined_stage_x, refined_stage_y, refined_stage_z = self.tem.report_stage_position()[:3]
        
        return {
            'pick_id': pick_id,
            'refined_stage': (refined_stage_x, refined_stage_y, refined_stage_z),
        }

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 8: Image Shift Calculation
    # ════════════════════════════════════════════════════════════════════════

    def calculate_image_shift_for_pick(self, pick: Pick, target_pick: Pick,
                                      target_magnification: int,
                                      calibration_calculator) -> Pick:

        if pick.pick_id == target_pick.pick_id:
            print(f"  {pick.pick_id} (tracking): no shift needed")
            pick.image_shift_x = 0.0
            pick.image_shift_y = 0.0
            return pick

        offset_x_px = pick.pixel_x_um - target_pick.pixel_x_um
        offset_y_px = pick.pixel_y_um - target_pick.pixel_y_um

        offset_x_um = offset_x_px * self.pixel_spacing_um
        offset_y_um = offset_y_px * self.pixel_spacing_um

        shift_x, shift_y = calibration_calculator.apply_matrix_to_shift(offset_x_um, offset_y_um,target_magnification)

        pick.image_shift_x = shift_x
        pick.image_shift_y = shift_y
        
        print(f"  {pick.pick_id}: shift=({shift_x:+.6f}, {shift_y:+.6f}) µm")
        
        return pick

    def calculate_image_shifts_for_group(self, group: TargetGroup, 
                                        target_magnification: int,
                                        calibration_calculator) -> TargetGroup:

        print(f"\n[INFO] Calculating image shifts for group {group.group_id}...")
        
        for pick in group.picks:
            self.calculate_image_shift_for_pick(pick, group.tracking, 
                                               target_magnification, calibration_calculator)
        
        return group

    # ════════════════════════════════════════════════════════════════════════
    # File Generation (xg1)
    # ════════════════════════════════════════════════════════════════════════

    def generate_xg1_file(self, group: TargetGroup, record_mag: int,
                         output_folder: str) -> str:
        """
        Generate xg1 file for paceTomo using Pick objects.
        
        Reads all data directly from Pick fields.
        
        Parameters
        ----------
        group : TargetGroup
            Group with refined picks
        record_mag : int
            Record magnification
        output_folder : str
            Where to save xg1 file
        
        Returns
        -------
        xg1_path : str
            Path to generated xg1 file
        """
        
        os.makedirs(output_folder, exist_ok=True)
        
        target = group.tracking
        
        xg1_filename = f"xg1_group_{group.group_id}.txt"
        xg1_path = os.path.join(output_folder, xg1_filename)
        
        print(f"\n[INFO] Generating xg1 file: {xg1_filename}")
        
        lines = []
        
        # Header
        lines.append("# Specimen Grid Group File (xg1)")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append(f"# Group: {group.group_id}")
        lines.append(f"# Magnification: {record_mag}x")
        lines.append("")
        
        # ─────────────────────────────────────────────────────────────────
        # Tracking target (001)
        # ─────────────────────────────────────────────────────────────────
        
        # Get stage position (refined if available, else original)
        if target.refined_stage_x is not None:
            target_stage_x = target.refined_stage_x
            target_stage_y = target.refined_stage_y
            target_stage_z = target.refined_stage_z
        else:
            target_stage_x = target.stage_x_um
            target_stage_y = target.stage_y_um
            target_stage_z = target.stage_z_um
        
        # Get image shift
        shift_x, shift_y = target.get_image_shift()
        
        lines.append("_target = 001")
        lines.append(f"ID = {target.pick_id}")
        lines.append(f"StageX = {target_stage_x:.6f}")
        lines.append(f"StageY = {target_stage_y:.6f}")
        lines.append(f"StageZ = {target_stage_z:.6f}")
        lines.append(f"ImageShiftX = {shift_x:.6f}")
        lines.append(f"ImageShiftY = {shift_y:.6f}")
        
        if target.record_img_path:
            lines.append(f"RecordImage = {os.path.basename(target.record_img_path)}")
        if target.view_crop_path:
            lines.append(f"ViewCrop = {os.path.basename(target.view_crop_path)}")
        
        lines.append("")
        
        # ─────────────────────────────────────────────────────────────────
        # Other picks (002, 003, ...)
        # ─────────────────────────────────────────────────────────────────
        
        for i, pick in enumerate(group.picks):
            if pick.pick_id == target.pick_id:
                continue
            
            target_num = str(i + 2).zfill(3)
            
            # Get stage position
            if pick.refined_stage_x is not None:
                pick_stage_x = pick.refined_stage_x
                pick_stage_y = pick.refined_stage_y
                pick_stage_z = pick.refined_stage_z
            else:
                pick_stage_x = pick.stage_x_um
                pick_stage_y = pick.stage_y_um
                pick_stage_z = pick.stage_z_um
            
            # Get image shift
            shift_x, shift_y = pick.get_image_shift()
            
            lines.append(f"_target = {target_num}")
            lines.append(f"ID = {pick.pick_id}")
            lines.append(f"StageX = {pick_stage_x:.6f}")
            lines.append(f"StageY = {pick_stage_y:.6f}")
            lines.append(f"StageZ = {pick_stage_z:.6f}")
            lines.append(f"ImageShiftX = {shift_x:.6f}")
            lines.append(f"ImageShiftY = {shift_y:.6f}")
            
            if pick.record_img_path:
                lines.append(f"RecordImage = {os.path.basename(pick.record_img_path)}")
            if pick.view_crop_path:
                lines.append(f"ViewCrop = {os.path.basename(pick.view_crop_path)}")
            
            lines.append("")
        
        # Write file
        with open(xg1_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"  Saved: {xg1_path}")
        return xg1_path

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 11: Workflow Orchestration
    # ════════════════════════════════════════════════════════════════════════

    def process_group_complete(self, group: TargetGroup, 
                              montage_mag: int = 2000,
                              record_mag: int = 40000,
                              view_mag: int = 15000,
                              calibration_calculator = None,
                              output_folder: Optional[str] = None) -> Dict:
        """
        Complete workflow for processing one group.
        
        Steps:
        1. Extract reference crops for non-targets
        2. Refine target stage position
        3. Calculate image shifts for all picks
        4. Generate xg1 file
        5. Create navigator entries
        
        Parameters
        ----------
        group : TargetGroup
            Group to process
        montage_mag : int
            Montage magnification
        record_mag : int
            Record magnification
        view_mag : int
            View magnification
        calibration_calculator : CalibratedImageShiftCalculator
            Calibration for image shifts
        output_folder : str
            Main project folder for xg1 output
        
        Returns
        -------
        results : dict
            Summary of processing results
        """
        
        print(f"\n{'='*70}")
        print(f"Processing Group {group.group_id}")
        print(f"{'='*70}")
        
        # Step 1: Extract reference crops
        ref_crops = self.extract_reference_crops_for_group(group)
        
        # Step 2: Refine target
        target = self.refine_target_stage_position(
            group.tracking, montage_mag, record_mag, view_mag
        )
        
        # Step 3: Calculate image shifts
        if calibration_calculator is not None:
            self.calculate_image_shifts_for_group(
                group, record_mag, calibration_calculator
            )
        
        # Step 4: Generate xg1
        if output_folder is None:
            output_folder = self.site_output_root
        
        xg1_path = self.generate_xg1_file(group, record_mag, output_folder)
        
        # Step 5: Create navigator
        nav_indices = self.add_group_to_navigator(group, record_mag)
        
        return {
            'group_id': group.group_id,
            'target': target,
            'reference_crops': ref_crops,
            'xg1_file': xg1_path,
            'navigator_indices': nav_indices,
        }

    def process_all_groups_complete(self, groups: List[TargetGroup],
                                   montage_mag: int = 2000,
                                   record_mag: int = 40000,
                                   view_mag: int = 15000,
                                   calibration_calculator = None,
                                   output_folder: Optional[str] = None) -> List[Dict]:
        """
        Complete workflow for processing all groups.
        
        Parameters
        ----------
        groups : list of TargetGroup
            Groups to process
        montage_mag : int
            Montage magnification
        record_mag : int
            Record magnification
        view_mag : int
            View magnification
        calibration_calculator : CalibratedImageShiftCalculator
            Calibration for image shifts
        output_folder : str
            Main project folder
        
        Returns
        -------
        results : list of dict
            Results for each group
        """
        
        results = []
        
        for group in groups:
            result = self.process_group_complete(
                group, montage_mag, record_mag, view_mag,
                calibration_calculator, output_folder
            )
            results.append(result)
        
        print(f"\n{'='*70}")
        print(f"Completed processing {len(groups)} groups!")
        print(f"{'='*70}")
        
        return results