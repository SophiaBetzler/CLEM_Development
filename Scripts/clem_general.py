import os
import csv
from pathlib import Path
        
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np



class ExecutiveControls:

    NA_VALUES = {"", "na", "NA", "n/a", "N/A", "none", "None"}

    def __init__(self, tem_communication, mrc_reader, sample_type, milling_angle=0.0):

        self.tem = tem_communication
        self.sample_type = sample_type
        self.milling_angle = milling_angle
        self.mrc = mrc_reader

        self. montage_settings = {'lamella': {"stage_tilt": -self.milling_angle, "fov_um_x": 15.0, "fov_um_y": 30.0},
                                    'airyscan': {"stage_tilt": 0.0, "fov_um_x": 105.0, "fov_um_y": 105.0}}

    # ---------------------------------------------------------------------------
    # Data loading (CSV file management)
    # ---------------------------------------------------------------------------

    def _csv_value_or_none(self, value):
        if value is None:
            return None
        value = str(value).strip()
        if value in self.NA_VALUES:
            return None
        return value


    def _csv_float_or_none(self, value):
        value = self._csv_value_or_none(value)
        if value is None:
            return None
        return float(value)

    def _import_csv_file(self, filename):
        csv_path = os.path.join(self.tem.output_root, filename)

        with open(csv_path, newline="") as f:
            rows = (line for line in f if not line.lstrip().startswith("#"))
            reader = csv.DictReader(rows)

            sites = []
            for row in reader:
                site = dict(row)

                site["name"] = row.get("name") or row.get("label")
                site["stage_x_um"] = self._csv_float_or_none(row.get("stage_x_um"))
                site["stage_y_um"] = self._csv_float_or_none(row.get("stage_y_um"))
                site["stage_z_um"] = self._csv_float_or_none(row.get("stage_z_um"))

                if "stage_tilt" in row:
                    site["stage_tilt"] = self._csv_float_or_none(row.get("stage_tilt"))

                sites.append(site)

        return sites
    
    def _write_sites_csv(self, sites, filename):
        csv_path = os.path.join(self.path, filename)

        fieldnames = list(sites[0].keys())

        if "stage_z_um" not in fieldnames:
            fieldnames.append("stage_z_um")

        if "stage_tilt" not in fieldnames:
            fieldnames.append("stage_tilt")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for site in sites:
                writer.writerow({
                    key: ("na" if site.get(key) is None else site.get(key)) for key in fieldnames})

    def run_experiment_setup(self):

        self.tem._create_nav_file()
        input("[ToDO] Please load the experiment file and the settings file. ENTER")

        self.tem._reset_defocus()
        self.tem.precise_stage_move(stage_x_um=0.0, stage_y_um=0.0, stage_z_um=0.0, stage_tilt=0.0)

        self.tem.acquire_image(mode='View', imaging_state='LMM')

        input('[ToDO] Move feature suitable for eucentricity alignment to the center of the stage (shift + right click + drag). ENTER')
        
        self.tem.set_eucentricity(level='rough')
        self.tem.acquire_montage(imaging_state='LMM', fov_um_x=3000.0, fov_um_y=3000.0, stage_tilt=0.0)

    def run_acquire_position_montages(self):
        
        sites = self._import_csv_file(filename='tem_stage_positions.csv')
        updated_sites = []

        for site_number, site in enumerate(sites):
            site_id = site.get("name") or f"site_{site_number+1:02d}"
            print(f"[INFO] Acquiring montage for {site_id}.")
            
            self.tem.precise_stage_move(stage_x_um=site["stage_x_um"], stage_y_um=site["stage_y_um"], stage_z_um=site["stage_z_um"])
            self.tem.acquire_image(mode='View', imaging_state='grid_square') 
            input("Please move the center of the grid square / lamella to the center of the field of view. ENTER")
            self.tem.set_eucentricity(level="rough_fine")
            self.tem.precise_stage_move(stage_tilt=self.milling_angle)
            stage_x_um, stage_y_um, stage_z_um, stage_tilt = self.tem.report_stage_position()
            self.tem.acquire_montage(stage_tilt=self.milling_angle, fov_um_x=self.montage_settings[self.sample_type]['fov_um_x'], fov_um_y=self.montage_settings[self.sample_type]['fov_um_y'], position=f"{site['name']}")
            updated_site = dict(site)
            updated_site["stage_x_um"] = stage_x_um
            updated_site["stage_y_um"] = stage_y_um
            updated_site["stage_z_um"] = stage_z_um
            updated_site["stage_tilt"] = stage_tilt
            updated_sites.append(updated_site)

        self._write_sites_csv(updated_sites, filename='tem_stage_positions_refined.csv')


### I think the run_clem_alignment function should be changed anyways to call the overlay tool.

    def run_clem_alignment(self):
        from clem_ui import RegistrationApp   
        tem_stage_positions = self._import_csv_file('tem_stage_positions_refined.csv')
        for site in tem_stage_positions:
            print(site['name'])
            ui = RegistrationApp(mrc_reader=self.mrc, site_id=site['name'])
            ui.mainloop()
        #     montage_summary = self.mrc.run_montage_loader_and_create_summary(os.path.join(self.path, montage_file))
        #     from clem_target_picking import CLEMPicker
        #     target_picker = CLEMPicker(montage_settings=montage_summary, tem_communication=self.tem)
        #     target_picker.run_auto_picker()

     


@dataclass
class SiteDataSummary:
    site_id: str
    path: str
    mrc: Optional[dict] = None
    tiff: Optional[dict] = None
    registration: Optional[dict] = None
    acquisition: Optional[dict] = None
    picks: list = field(default_factory=list)

    # ======================================================================= #
    # MRC (TEM reference montage)
    # ======================================================================= #

    def load_mrc_data_summary(self, mrc_reader, mrc_filepath):
        summary = mrc_reader.build_montage_summary(mrc_filepath=mrc_filepath)

        src = summary.get("mrc", summary)

        tiles = []
        for t in src.get("tiles", []):
            tiles.append({
                "z_index": t.get("z_index", t.get("z")),
                "stage_z_um": t.get("stage_z_um", t.get("stage_z")),
                "pixel_coordinates_um": _coords_to_dict(
                    t.get("pixel_coordinates_um", t.get("px")),
                    "x_um", "y_um"),
                "stage_xy_um": _coords_to_dict(
                    t.get("stage_xy_um", t.get("stage")),
                    "stage_x_um", "stage_y_um"),
            })

        self.mrc = {
            "mrc_path": os.fspath(mrc_filepath),
            "image": src.get("image"),
            "image_height_width": src.get("mrc_image_height_width"),
            "pixel_spacing_um": src.get("pixel_spacing_um"),
            "feather_pixels": src.get("feather_pixels", getattr(mrc_reader, "_feather_px", None)),
            "section": src.get("section"),
            "alignment": src.get("alignment"),
            "rotation_deg": src.get("rotation_deg"),
            "min_x_pixels": src.get("min_x_pixels"),
            "min_y_pixels": src.get("min_y_pixels"),
            "tiles": src.get("tiles"),
            "stage_fit": src.get("stage_fit"),
            "stage_matrix": src.get("stage_matrix"),
        }
        return self

    # ======================================================================= #
    # TIFF (fluorescence channels)
    # ======================================================================= #
    def load_tiff_data_summary(self, mrc_reader, tiff_filepath):
        """Populate self.tiff using the reader's load_ome_tiff().

        self.tiff schema:
            ome_path                : str
            stack_czyx              : np.ndarray  (Channels, Z, Y, X)
            num_channels            : int
            num_z_slices            : int
            stack_height_width      : (y, x) in pixels
            info                    : str
        """
        stack_czyx, info = mrc_reader.load_ome_tiff(tiff_filepath)
        c, z, y, x = stack_czyx.shape
        self.tiff = {
            "ome_path": os.fspath(tiff_filepath),
            "stack_czyx": stack_czyx,
            "num_channels": int(c),
            "num_z_slices": int(z),
            "stack_height_width": (int(y), int(x)),
            "info": info,
        }
        return self

    # ======================================================================= #
    # Registration (TIFF -> MRC transform + warped channels)
    # ======================================================================= #
    def set_registration(self, correlator_result, transform_type=None,
                          flip_x=False, flip_y=False):
        """Store the result dict returned by CLEMCorrelator.apply_transform().

        self.registration schema:
            transform_type          : str
            transform_matrix        : np.ndarray (3x3) | skimage transform
            fit_info                : dict   (scale_x, scale_y, rotation_deg, rmse_px)
            num_point_pairs         : int
            flip_x, flip_y          : bool
            warped_channels         : list[np.ndarray]  (each in MRC grid space)
        """
        r = correlator_result
        self.registration = {
            "transform_type": transform_type or r.get("transform_type"),
            "transform_matrix": r.get("transform"),
            "fit_info": r.get("fit_info"),
            "num_point_pairs": r.get("n_pairs"),
            "flip_x": bool(flip_x),
            "flip_y": bool(flip_y),
            "warped_channels": r.get("warped_channels"),
        }
        return self

    # ======================================================================= #
    # Acquisition (stage position / tilt for this site)
    # ======================================================================= #
    def set_acquisition_from_csv_row(self, row):
        """Fill self.acquisition from a site row produced by
        ExecutiveControls._import_csv_file().

        self.acquisition schema:
            stage_x_um, stage_y_um, stage_z_um : float | None
            stage_tilt                         : float | None
        """
        self.acquisition = {
            "stage_x_um": row.get("stage_x_um"),
            "stage_y_um": row.get("stage_y_um"),
            "stage_z_um": row.get("stage_z_um"),
            "stage_tilt": row.get("stage_tilt"),
        }
        return self

    # ======================================================================= #
    # Picks (target positions)
    # ======================================================================= #
    def add_pick(self, pick_id, pixel_xy_um=None, stage_xy_um=None,
                 stage_z_um=None, notes=None):
        """Append one pick. Coordinates are dicts to match the scheme:
            pixel_xy_um : {"x_um", "y_um"}
            stage_xy_um : {"stage_x_um", "stage_y_um"}
        """
        self.picks.append({
            "pick_id": pick_id,
            "pixel_coordinates_um": _coords_to_dict(pixel_xy_um, "x_um", "y_um"),
            "stage_xy_um": _coords_to_dict(stage_xy_um,
                                           "stage_x_um", "stage_y_um"),
            "stage_z_um": stage_z_um,
            "notes": notes,
        })
        return self

    # ======================================================================= #
    # Convenience accessors (avoid deep-dict typos at call sites)
    # ======================================================================= #
    @property
    def pixel_spacing_um(self):
        return self.mrc["pixel_spacing_um"] if self.mrc else None

    @property
    def montage_image(self):
        return self.mrc["image"] if self.mrc else None

    @property
    def warped_channels(self):
        return self.registration["warped_channels"] if self.registration else None

    def validate(self):
        """Light sanity checks. Raises ValueError on clear inconsistencies;
        returns a list of non-fatal warnings."""
        warnings = []
        if self.mrc and self.mrc.get("pixel_spacing_um") in (None, 0):
            warnings.append("mrc.pixel_spacing_um is missing or zero.")
        if self.tiff and self.registration:
            n_ch = self.tiff["num_channels"]
            warped = self.registration.get("warped_channels") or []
            if warped and len(warped) != n_ch:
                warnings.append(
                    f"warped_channels ({len(warped)}) != num_channels ({n_ch}).")
        return warnings

    # ======================================================================= #
    # Persistence  (single pickle file per site)
    # ======================================================================= #
    def default_pickle_path(self):
        return os.path.join(self.path, f"{self.site_id}.pkl")

    def save(self, pickle_path=None):
        """Pickle the whole object (arrays included) to one file."""
        pickle_path = pickle_path or self.default_pickle_path()
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return pickle_path

    @classmethod
    def load(cls, pickle_path):
        with open(pickle_path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{pickle_path} did not contain a SiteData object.")
        return obj

    def __repr__(self):
        stages = [name for name in ("mrc", "tiff", "registration", "acquisition")
                  if getattr(self, name) is not None]
        return (f"SiteData(site_id={self.site_id!r}, "
                f"loaded={stages}, n_picks={len(self.picks)})")