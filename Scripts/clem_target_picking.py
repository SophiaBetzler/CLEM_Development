# ============================================================================
# COMPLETE CLEMPICKER CLASS - ALL REQUIRED FUNCTIONS
# ============================================================================
# Comprehensive implementation with all utilities and workflow methods

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mrcfile
import os
from clem_dataclasses import Pick
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
            # AllSitesDataCollection stores sites in .sites, keyed by site_id.
            site_data = sc.get_site(self.site_id) if sc is not None else None
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

    def add_picks_to_navigator(self, mrc_dataclass, buffer='S', site_label=None):
        """Return every pick to SerialEM as a navigator point.

        Notes carry the site label and the pick number, so a navigator item can
        be matched to its crop files. Defaults to this picker's site id.
        Returns the navigator indices.
        """
        if site_label is None:
            site_label = self.site_id
        return self.tem.add_nav_point(mrc_dataclass=mrc_dataclass, buffer=buffer,
                                      stage_z_um=self.resolve_stage_z(),
                                      site_label=site_label)

    def resolve_stage_z(self):
        if self.mrc_dataclass.stage_z_um is not None:
            return float(self.mrc_dataclass.stage_z_um)
        if self.mrc_dataclass.metadata is not None and self.mrc_dataclass.metadata.stage_z_um:
            return float(self.mrc_dataclass.metadata.stage_z_um)
        zs = [t.piece_z_stage_um for t in self.mrc_dataclass.tiles if t.piece_z_stage_um is not None]
        if zs:
            return float(np.median(zs))
        if self.site_data is not None and self.site_data.stage_position:
            return float(self.site_data.stage_position[2])
        return -999.0


