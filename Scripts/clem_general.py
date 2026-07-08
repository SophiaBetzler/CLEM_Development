import os
import csv
from pathlib import Path

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

     
        
        