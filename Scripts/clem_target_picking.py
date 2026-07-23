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
        if site_data.mrc is None:
            raise ValueError("site_data.mrc must be populated before creating CLEMPicker")

        self.mrc_summary = site_data.mrc

        self.pixel_spacing_um = self.mrc_summary.pixel_spacing_um
        self.coord_field = self.mrc_summary.coord_field
        self.output_coord_mode = "image" # or "stage"
        self.nav_map_buffer = "A"

        self.site_id = site_data.site_id
        self.site_output_root = site_data.path
        self.tem = tem_communication
        self.mrc_reader = self.tem.mrc_reader
        self.H, self.W = self.mrc_summary.image.shape[:2]
        
        self.flip_x = bool(getattr(self.mrc_summary, 'flip_x', False))
        self.flip_y = bool(getattr(self.mrc_summary, 'flip_y', False))

        theta = np.deg2rad(getattr(self.mrc_summary, 'rotation_deg', 0.0))
        self.cos, self.sin = np.cos(theta), np.sin(theta)

        self.tiles = self._build_lookup()

        if self.mrc_summary.stage_matrix is not None:
            self._M = np.asarray(self.mrc_summary.stage_matrix, dtype=float)
        else:
            self._M = None
             
    # ════════════════════════════════════════════════════════════════════════
    # Helper and Coordinate Conversion Methods
    # ════════════════════════════════════════════════════════════════════════

    def _build_lookup(self) -> List[TileCenter]:

        out = []
        for t in self.mrc_summary.tiles:
            if t.pixel_x_um is None or t.stage_x_um is None:
                continue
            out.append(TileCenter(
                z_index=t.z_index,
                stage_z_um=t.stage_z_um,
                cx=t.pixel_x_um - self.mrc_summary.min_x_pixels + self.mrc_summary.image_width / 2.0,
                cy=t.pixel_y_um - self.mrc_summary.min_y_pixels + self.mrc_summary.image_height / 2.0,
                sx=t.stage_x_um,
                sy=t.stage_y_um,
            ))
        
        if not out:
            raise ValueError("No usable tiles (each needs pixel origin and stage position).")
        return out

    def _piece_for(self, px: float, py: float) -> TileCenter:    
        inside = [t for t in self.tiles
                  if t.cx - self.mrc_summary.image_width / 2 <= px < t.cx + self.mrc_summary.image_width / 2
                  and t.cy - self.mrc_summary.image_height / 2 <= py < t.cy + self.mrc_summary.image_height / 2]
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

    def _shift_from_stage(self, pick, target_pick, stage_to_is):
        d = np.array([pick.stage_x_um - target_pick.stage_x_um, pick.stage_y_um - target_pick.stage_y_um], float)
        return stage_to_is @ d
    
    def _shift_from_image(self, pick, target_pick, cam_to_is, montage_px_per_record_px):
        d_cam = np.array([pick.pixel_x_um - target_pick.pixel_x_um, pick.pixel_y_um - target_pick.pixel_y_um], float) * montage_px_per_record_px
        return cam_to_is @ d_cam

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
    
    def add_group_to_navigator(self, group: TargetGroup) -> Dict[str, int]:
        target = group.tracking
        print(f"[INFO] Adding nav point 1.")
        nav_indices = {target.pick_id: self.tem.add_nav_point(
            pick=target, label=f"001_{target.pick_id}",
            output_coord_mode=self.output_coord_mode, image_height=self.H)}
        for i, pick in enumerate(group.picks):
            if pick.pick_id == target.pick_id:
                continue
            nav_indices[pick.pick_id] = self.tem.add_nav_point(
                pick=pick, label=f"{str(i+2).zfill(3)}_{pick.pick_id}",
                output_coord_mode=self.output_coord_mode, image_height=self.H)
        return nav_indices


    # ════════════════════════════════════════════════════════════════════════
    # Image Display Function
    # ════════════════════════════════════════════════════════════════════════

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
            tracking_target = self.create_tracking_target_for_group(member_picks, picks, lone_offset_um, 
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
    # Target Refinement
    # ════════════════════════════════════════════════════════════════════════

    def refine_target_stage_position(self, target_pick: Pick, site_id=None) -> Pick:
        
        print(f"\n[INFO] Refining target {target_pick.pick_id}...")
  
        montage_crop_at_target_position = self.mrc_reader._crop_centered_at_pixel_coord(self.mrc_summary.image, target_pick.pixel_x_um, target_pick.pixel_y_um, fov_um=2.0)

        target_crop_ref_path = os.path.join(site_id.path, f"{target_pick.pick_id}_target_reference.mrc")

        with mrcfile.new(target_crop_ref_path, overwrite=True) as mrc:
            mrc.set_data(montage_crop_at_target_position)
            mrc.voxel_size = self.pixel_spacing_um * 10000
            mrc.update_header_from_data()

        target_pick.view_crop_path = target_crop_ref_path

        alignment_result = self.tem.align_target_at_higher_mag(
            label=target_pick.pick_id,
            target_stage_pos=(target_pick.stage_x_um, target_pick.stage_y_um, target_pick.stage_z_um),
            reference_image_path=target_crop_ref_path, mode='Search')
        
        target_pick.search_img_path = self.tem.acquire_image(mode='Search', save=True, site_id=site_id, label='tg_s')
        refined_x, refined_y, refined_z = alignment_result['refined_stage']

        alignment_result = self.tem.align_target_at_higher_mag(
            target_stage_pos=(target_pick.stage_x_um, target_pick.stage_y_um, target_pick.stage_z_um),
            reference_image_path=target_pick.search_img_path, mode='Record')
        
        target_pick.search_img_path = self.tem.acquire_image(mode='Record', save=True, site_id=site_id, label='tg_r')
        refined_x, refined_y, refined_z = alignment_result['refined_stage']


        target_pick.refined_stage_x = refined_x
        target_pick.refined_stage_y = refined_y
        target_pick.refined_stage_z = refined_z
        target_pick.refinement_quality='good'


    # ════════════════════════════════════════════════════════════════════════
    # SECTION 8: Image Shift Calculation
    # ════════════════════════════════════════════════════════════════════════

    def calculate_image_shift_for_pick(self, pick: Pick, target_pick: Pick, transform, source="image", extra=None) -> Pick:

        if pick.pick_id == target_pick.pick_id:
            print(f"  {pick.pick_id} (tracking): no shift needed")
            pick.image_shift_x = 0.0
            pick.image_shift_y = 0.0
            return pick
        
        if source == 'stage':
            is_x, is_y = self._shift_from_stage(pick, target_pick, transform)
        else:
            is_x, is_y = self._shift_from_image(pick, target_pick, transform, extra)

        

        pick.image_shift_x = is_x
        pick.image_shift_y = is_y
        
        print(f"{pick.pick_id}: shift=({is_x:+.6f}, {is_y:+.6f}) µm")
        
        return pick

    def calculate_image_shifts_for_group(self, group: TargetGroup, source='image', mode='R') -> TargetGroup:

        if source == 'stage':
            transform = self.tem.report_stage_to_is_matrix(mode=mode)
            extra = None
        else:
            transform = self.tem.report_camera_to_is_matrix(mode=mode)
            record_props = self.tem.get_image_properties(mode=mode)
            extra = self.pixel_spacing_um / record_props["pixel_size_um"]

        if transform is None:
            for pick in group.picks:
                pick.image_shift_x = 0.0
                pick.image_shift_y = 0.0
            return group
        
        for pick in group.picks:
            self.calculate_image_shift_for_pick(pick=pick, target_pick=group.tracking, transform=transform, extra=extra, source=source)
        return group

    # ════════════════════════════════════════════════════════════════════════
    # File Generation (xg1)
    # ════════════════════════════════════════════════════════════════════════

    def generate_xg1_file(self, group: TargetGroup, output_folder: str) -> str:
        
        os.makedirs(output_folder, exist_ok=True)
        
        target = group.tracking
        
        xg1_filename = f"xg1_group_{group.group_id}.txt"
        xg1_path = os.path.join(output_folder, xg1_filename)
        record_img_properties = self.tem.get_image_properties(mode='Record')

        print(f"\n[INFO] Generating xg1 file: {xg1_filename}")
        
        lines = []

        lines.append("# Specimen Grid Group File (xg1)")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append(f"# Group: {group.group_id}")
        lines.append(f"# Magnification: {record_img_properties['magnification']}x")
        lines.append("")

        if target.refined_stage_x is not None:
            target_stage_x = target.refined_stage_x
            target_stage_y = target.refined_stage_y
            target_stage_z = target.refined_stage_z
        else:
            target_stage_x = target.stage_x_um
            target_stage_y = target.stage_y_um
            target_stage_z = target.stage_z_um
 
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
    # Run paceTomo target/group generation for all picks/groups
    # ════════════════════════════════════════════════════════════════════════
    
    def run_create_groups_for_pacetomo(self, site_id, radius_um=7.5, crop_fov=2.0, output_folder=None, shift_source='image'):

        if output_folder is None:
            output_folder = self.site_output_root          # was self.site_output.root

        os.makedirs(output_folder, exist_ok=True)

        groups = self.group_picks(radius_um=radius_um)

        xg1_files = []
        for group in groups:
            if not self.tem.offline:                        # refine needs the scope
                self.refine_target_stage_position(target_pick=group.tracking, site_id=site_id)
            self.calculate_image_shifts_for_group(group, source=shift_source)
            ref_crops = self.mrc_reader.write_mrc_crops(mrc_image=self.mrc_summary.image, picks=group.picks,
                fov_um=crop_fov, output_root=os.path.join(output_folder, f"group{group.group_id}"),  
                skip_pick_id=group.tracking.pick_id, )
            nav_indices = self.add_group_to_navigator(group)
            xg1_files.append(self.generate_xg1_file(group, output_folder))

        return groups, xg1_files
