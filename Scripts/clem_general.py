import os
import csv
from pathlib import Path      


from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import numpy as np



class ExecutiveControls:

    NA_VALUES = {"", "na", "NA", "n/a", "N/A", "none", "None"}

    def __init__(self, tem_communication, mrc_reader, sample_type, milling_angle, site_collection):

        self.tem = tem_communication
        self.sample_type = sample_type
        self.milling_angle = milling_angle
        self.mrc_reader = mrc_reader
        self.site_collection = site_collection
        self.site_summaries = {}

        if self.site_collection is not None:
            self.tem.site_collection = self.site_collection
            self.mrc_reader.site_collection = self.site_collection

        self.montage_settings = {'lamella': {"stage_tilt": -self.milling_angle, "fov_um_x": 15.0, "fov_um_y": 35.0},
                                    'airyscan': {"stage_tilt": 0.0, "fov_um_x": 35.0, "fov_um_y": 35.0, "fov_um_x_high_mag": 25.0, "fov_um_y_high_mag": 25.0},}

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
        print(f"[INFO] Importing csv file {csv_path}.")

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
        csv_path = os.path.join(self.tem.output_root, filename)

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

        self.tem.create_nav_file()
        input("[ToDO] Please load the experiment file and the settings file. ENTER")
             
        self.tem.reset_defocus()
        self.tem.precise_stage_move(stage_x_um=0.0, stage_y_um=0.0, stage_z_um=0.0, stage_tilt=0.0)

        self.tem.acquire_image(mode='View', imaging_state='LMM')
        input('[ToDO] Move feature suitable for eucentricity alignment to the center of the stage (shift + right click + drag). ENTER')
        self.tem.set_eucentricity(level='rough')
        self.tem.acquire_montage(site_data=None, mode='Search', imaging_state='LMM', fov_um_x=3000.0, fov_um_y=3000.0, eucentricity=False)
        self.tem.acquire_image(mode='View', imaging_state='grid_square') 
        self.tem.return_control_to_serialem(message="[ToDo] Run 'align to marker' alignment. ENTER.")

    def run_acquire_position_montages(self):
        from clem_dataclasses import SiteDataSummary
        input('[ToDo] Use either clem_airyscan_tool or clem_arctis_tool to create tem_stage_positions.csv and copy it to the experiment folder. ENTER')
        sites = self._import_csv_file(filename='tem_stage_positions.csv')
        for site_number, site in enumerate(sites):
            site_id = site.get("name") or f"site_{site_number+1:02d}"
            site_dir = os.path.join(self.tem.output_root, site_id)
            os.makedirs(site_dir, exist_ok=True)
            print(f"[INFO] Acquiring montage for {site_id}.")
            self.tem.precise_stage_move(stage_x_um=site["stage_x_um"], stage_y_um=site["stage_y_um"], stage_z_um=site["stage_z_um"])
            self.tem.acquire_image(mode='View', imaging_state='grid_square', save=False) 
            input("Please move the center of the grid square / lamella to the center of the field of view. ENTER")
            self.tem.acquire_image(mode='View', imaging_state='grid_square') 
            input("Please move the center of the grid square / lamella to the center of the field of view. ENTER")
            self.tem.set_eucentricity(level='rough_fine')
            stage_x_um, stage_y_um, stage_z_um, stage_tilt = self.tem.report_stage_position()
            site_data = SiteDataSummary(site_id=site_id, path=site_dir, stage_position=[stage_x_um, stage_y_um, stage_z_um, stage_tilt], milling_angle=self.milling_angle)
            self.tem.acquire_image(mode='View', imaging_state='grid_square', label=f"{site_id}_overview", save=True)
            self.register_site_data(site_data, active=False)
        self.tem.acquire_image(mode='View', imaging_state='grid_square') 
        self.tem.acquire_image(mode='View', imaging_state=None) # HERE THE SWITCH TO LOWDOSE MODE
        self.tem.return_control_to_serialem(message="Run 'align to marker' alignment. Press Continue to resume.")
        for site_id, site_data in self.site_collection.sites.items():
            self.tem.precise_stage_move(stage_x_um = site_data.stage_position[0], stage_y_um = site_data.stage_position[1], stage_z_um = site_data.stage_position[2])
            nav_idx = self.tem.find_nav_item_with_note(f"{site_id}_overview")
            if self.sample_type == 'airyscan':
                self.tem.acquire_montage_at_nav_item(site_data=site_data, mode='View', nav_idx=nav_idx, fov_um_x=self.montage_settings[self.sample_type]['fov_um_x'], fov_um_y=self.montage_settings[self.sample_type]['fov_um_y'], eucentricity=True) 
            elif self.sample_type == 'lamella':
                _, montage_path = self.tem.acquire_montage_at_nav_item(mode='Search', nav_idx=nav_idx, fov_um_x=self.montage_settings[self.sample_type]['fov_um_x'], fov_um_y=self.montage_settings[self.sample_type]['fov_um_y'], eucentricity=True)
                self.mrc.load_mrc_montage(montage_path) 
                site_data.mrc = self.mrc.build_montage_summary(montage_path)    
            site_data.save()
            

    def register_site_data(self, site_data, active=True):
        if self.site_collection is not None:
            self.site_collection.add_site(site_data)
            if active:
                self.site_collection.set_active_site(site_data.site_id)
        if site_data.site_id is not None:
            self.site_summaries[site_data.site_id] = site_data
        return site_data

    def run_clem_alignment(self):
        from clem_ui import RegistrationApp   
        from clem_dataclasses import SiteDataSummary
        tem_stage_positions = self._import_csv_file('tem_stage_positions_refined.csv')
        seen = set()
        for site in tem_stage_positions:
            site_id = site["name"]
            if site_id in seen:
                raise ValueError(f"Dublicate site_id {site_id!r} in CSV file.")
            seen.add(site_id)
            site_data = SiteDataSummary(site_id=site_id, path=os.path.join(self.mrc.output_root, site_id))
            site_data.set_acquisition_from_csv_row(site)
            print(site_data.path)
            nav_idx = self.tem.find_nav_item_with_note(Path(site_data.mrc.path).stem)
            self.tem.find_buffer_of_montage(idx=nav_idx, buffer="A")
            self.register_site_data(site_data, active=True)
            ui = RegistrationApp(mrc_reader=self.mrc, site_data=site_data, tem_communication=self.tem)
            ui.mainloop()

            site_data.save()                    # -> <folder>/<site_id>_<timestamp>.pkl
            self.register_site_data(site_data, active=True)
            self.site_summaries[site_id] = site

        return self.site_summaries

    def run_high_magnification_clem_alignment_step(self, site_data):
        H, W = site_data.mrc.image.shape
        center_px = (W / 2.0, H / 2.0)
        mrc_path = self.mrc.write_mrc_crops(site_data, fov_um=2.0, output_root=site_data.path, label='crop_center_', skip_pick_id=None)
        self.tem.align_target_at_higher_mag(reference_image_path=mrc_path) 
        self.tem.acquire_montage(mode='Search', fov_um_x=5.0, fov_um_y=5.0, stage_tilt=self.milling_angle, site_id=site_data.site_id, eucentricity=True)
        from clem_ui import RegistrationApp 
        ui = RegistrationApp(mrc_reader=self.mrc, site_data=site_data, tem_communication=self.tem)
        ui.mainloop()
        # MISSING IMPORT TRANSFORMATION AND STORE TRANSFORMATION
        

