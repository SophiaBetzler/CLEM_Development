# ============================================================================
# COMPLETE CLEMPICKER CLASS - ALL REQUIRED FUNCTIONS
# ============================================================================
# Comprehensive implementation with all utilities and workflow methods

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
import os
from clem_dataclasses import Pick, TargetGroup
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


class CLEMPicker:

    def __init__(self, mrc_dataclass, tem_communication, site_data=None):

        self.mrc_dataclass = mrc_dataclass
        if self.mrc_dataclass is None:
            raise ValueError("mrc_dataclass must be populated before creating CLEMPicker")

        self.pixel_spacing_um = self.mrc_dataclass.pixel_spacing_um
        self.coord_field = self.mrc_dataclass.coord_field
        self.nav_map_buffer = "S"

        p = Path(self.mrc_dataclass.mrc_path)
        self.site_id = p.parent.name
        self.site_output_root = p.parent
        self.tem = tem_communication
        self.mrc_reader = self.tem.mrc_reader
        
        if site_data is None:
            sc = getattr(self.tem, "site_collection", None)
            site_data = sc.site_data.get(self.site_id) if sc is not None else None
        self.site_data = site_data
      
    # ════════════════════════════════════════════════════════════════════════
    # Helper and Coordinate Conversion Methods
    # ════════════════════════════════════════════════════════════════════════

    def convert_display_to_montage_orientation(self, dx: float, dy: float) -> Tuple[float, float]:
        px = (self.mrc_dataclass.image_width - 1 - dx) if self.mrc_dataclass.flip_x else dx
        py = (self.mrc_dataclass.image_height - 1 - dy) if self.mrc_dataclass.flip_y else dy
        return px, py

    def convert_montage_to_display_orientation(self, px: float, py: float) -> Tuple[float, float]:
        """Convert montage coordinates to display coordinates."""
        dx = (self.mrc_dataclass.image_width - 1 - px) if self.mrc_dataclass.flip_x else px
        dy = (self.mrc_dataclass.image_height - 1 - py) if self.mrc_dataclass.flip_y else py
        return dx, dy

    # ════════════════════════════════════════════════════════════════════════
    # Pick Creation and Management
    # ════════════════════════════════════════════════════════════════════════

    def make_pick_dataclass(self, px: float, py: float, pick_id: Optional[str] = None) -> Pick:
        """Create a Pick dataclass from pixel coordinates."""
              
        if pick_id is None:
            pick_id = str(len(self.mrc_dataclass.picks) + 1)
        
        return Pick(
                        pick_id=str(pick_id),
                        image_coord_x=px,
                        image_coord_y=py,
                    )

    def add_pick_from_pixel(self, px: float, py: float, pick_id: Optional[str] = None) -> Pick:
        pick = self.make_pick_dataclass(px, py, pick_id=pick_id)
        self.mrc_dataclass.picks.append(pick)
        return pick

    def add_pick_from_display(self, px: float, py: float, pick_id: Optional[str] = None) -> Pick:     
        #px, py = self.convert_display_to_montage_orientation(dx, dy)
        return self.add_pick_from_pixel(px, py, pick_id=pick_id)

    def remove_last_pick(self) -> Optional[Pick]:
        """Remove and return last pick."""
        
        if self.mrc_dataclass.picks:
            return self.mrc_dataclass.picks.pop()
        return None

    def clear_picks(self) -> None:
        """Clear all picks."""
        self.mrc_dataclass.picks.clear()

    def get_picks_count(self) -> int:
        """Get number of picks."""
        return len(self.mrc_dataclass.picks)

    def get_all_picks(self) -> List[Pick]:
        """Get all picks."""
        return self.mrc_dataclass.picks.copy()

    def add_picks_to_navigator(self, mrc_dataclass, buffer='S') -> int:
        self.tem.add_nav_point(mrc_dataclass=mrc_dataclass, buffer=buffer, stage_z_um=self.resolve_stage_z())
    
    def resolve_stage_z(self):
        if self.mrc_dataclass.stage_z_um is not None:
            return float(self.mrc_dataclass.stage_z_um)
        if self.mrc_dataclass.metadata is not None and self.mrc_dataclass.metadata.stage_z_um:
            return float(self.mrc_dataclass.metadata.stage_z_um)
        zs = [t.piece_z_stage_um for t in self.mrc_dataclass.tiles if t.piece_z_stage_um is not None]
        if zs:
            return float(np.median(zs))
        if self.site_data is not None and self.site_data.stage_position:
            return float(self.site_datastage_position[2])
        return -999.0

    # ════════════════════════════════════════════════════════════════════════
    # Pick Grouping and Tracking Target Selection
    # ════════════════════════════════════════════════════════════════════════

    def group_picks(self, radius_um: float = 7.5) -> List[TargetGroup]:
        picks = self.mrc_dataclass.picks
        if not picks:
            print("[WARN] No picks available to group.")
            return []

        # Work entirely in montage pixels — stage coords aren't set until the
        # nav points are added, so cluster in the same frame the picks carry.
        radius_px = radius_um / self.mrc_dataclass.pixel_spacing_um

        XY = np.array([[p.image_coord_x, p.image_coord_y] for p in picks], dtype=float)
        n = len(picks)
        assigned = [False] * n

        neigh = [int(np.sum(np.linalg.norm(XY - XY[i], axis=1) <= radius_px)) for i in range(n)]
        order = sorted(range(n), key=lambda i: -neigh[i])

        groups = []
        for seed in order:
            if assigned[seed]:
                continue

            members = [i for i in range(n)
                    if not assigned[i] and np.linalg.norm(XY[i] - XY[seed]) <= radius_px]

            for _ in range(10):
                centre = XY[members].mean(axis=0)
                kept = [i for i in members if np.linalg.norm(XY[i] - centre) <= radius_px]
                if len(kept) == len(members):
                    break
                members = kept if kept else [seed]

            for i in members:
                assigned[i] = True

            group_id = str(len(groups) + 1)
            member_picks = [picks[i] for i in members]
            tracking_target = self.create_tracking_target_for_group(member_picks)

            groups.append(TargetGroup(group_id=group_id, tracking=tracking_target, picks=member_picks))

        print(f"\n[INFO] Created {len(groups)} groups from {n} picks")
        return groups

    def create_tracking_target_for_group(self, group: List[Pick]) -> Pick:
        if not group:
            raise ValueError("Cannot select a tracking target from an empty group.")

        if len(group) == 1:
            return group[0]

        cx = float(np.mean([p.image_coord_x for p in group]))
        cy = float(np.mean([p.image_coord_y for p in group]))
        return min(group, key=lambda p: (p.image_coord_x - cx) ** 2 + (p.image_coord_y - cy) ** 2)

    # ════════════════════════════════════════════════════════════════════════
    # Target Refinement
    # ════════════════════════════════════════════════════════════════════════
#### NOT REWORKED YET - NEEDS TO BE REWRITTEN TO USE NEW ALIGNMENT FUNCTION
    def refine_target_stage_position(self, target_pick: Pick) -> Pick:
        
        print(f"\n[INFO] Refining target {target_pick.pick_id}...")
  
        montage_crop_at_target_position = self.mrc_reader._crop_centered_at_pixel_coord(self.mrc_summary.image, target_pick.pixel_x_um, target_pick.pixel_y_um, fov_um=2.0)

        target_crop_ref_path = os.path.join(self.site_data.path, f"{target_pick.pick_id}_target_reference.mrc")

        with mrcfile.new(target_crop_ref_path, overwrite=True) as mrc:
            mrc.set_data(montage_crop_at_target_position)
            mrc.voxel_size = self.mrc_dataclass.pixel_spacing_um * 10000
            mrc.update_header_from_data()

        target_pick.view_crop_path = target_crop_ref_path

        alignment_result = self.tem.align_target_at_higher_mag(
            label=target_pick.pick_id,
            target_stage_pos=(target_pick.stage_x_um, target_pick.stage_y_um, target_pick.stage_z_um),
            reference_image_path=target_crop_ref_path, mode='Search')
        
        target_pick.search_img_path = self.tem.acquire_image(mode='Search', save=True, site_id=self.site_id, label='tg_s')
        refined_x, refined_y, refined_z = alignment_result['refined_stage']

        alignment_result = self.tem.align_target_at_higher_mag(
            target_stage_pos=(target_pick.stage_x_um, target_pick.stage_y_um, target_pick.stage_z_um),
            reference_image_path=target_pick.search_img_path, mode='Record')
        
        target_pick.search_img_path = self.tem.acquire_image(mode='Record', save=True, site_id=self.site_id, label='tg_r')
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

    def calculate_image_shifts_for_group(self, group: TargetGroup, source='image', mode='Record') -> TargetGroup:

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
    
    def run_create_groups_for_pacetomo(self, radius_um, crop_fov, output_folder=None,
                                   warp_slice=None, n_channels=0, n_z=1):

        if output_folder is None:
            output_folder = self.site_output_root          # was self.site_output.root

        os.makedirs(output_folder, exist_ok=True)

        groups = self.group_picks(radius_um=radius_um)

        xg1_files = []
        for group in groups:
            # if not self.tem.offline:                        # refine needs the scope
            #     self.refine_target_stage_position(target_pick=group.tracking)
            #self.calculate_image_shifts_for_group(group, source=shift_source)
            ref_crops = self.mrc_reader.write_mrc_crops(mrc_dataclass=self.mrc_dataclass, fov_um=crop_fov,
                    pixel_spacing_um=self.mrc_dataclass.pixel_spacing_um, output_root=os.path.join(str(self.site_output_root), "picks", "crop"), skip_pick_id=group.tracking.pick_id)
            
            nav_indices = self.add_picks_to_navigator(self.mrc_dataclass, buffer=self.nav_map_buffer)
            if warp_slice is not None and n_channels > 0:
                self.mrc_reader.write_multichannel_crops(
                    mrc_dataclass=self.mrc_dataclass,
                    warp_slice=warp_slice, n_channels=n_channels, n_z=n_z,
                    fov_um=crop_fov,
                    output_root=os.path.join(str(self.site_output_root), "picks", "target_overlays"),
                    pixel_spacing_um=self.mrc_dataclass.pixel_spacing_um)
        #xg1_files.append(self.generate_xg1_file(group, output_folder))

        return groups, xg1_files
