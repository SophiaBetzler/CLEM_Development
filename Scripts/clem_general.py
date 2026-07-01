import os
import csv
from pathlib import Path

class ExecutiveControls:

    def __init__(self, tem_communication, mrc_reader, sample_type, path, milling_angle=0.0):

        self.tem = tem_communication
        self.sample_type = sample_type
        self.milling_angle = milling_angle
        self.mrc = mrc_reader
        self.path = path

        self. montage_settings = {'lamella': {"stage_tilt": -self.milling_angle, "fov_um_x": 15.0, "fov_um_y": 30.0},
                                    'airyscan': {"stage_tilt": 0.0, "fov_um_x": 105.0, "fov_um_y": 105.0}}

    def _import_csv_file(self, filename):

        with open(os.path.join(self.path, filename), newline="") as f:
            reader = csv.DictReader(f)
            positions = list(reader) 
        
        return positions

    def _write_csv_file(self, position, name, stage_position, stage_z, stage_tilt):
        csv_path=os.path.join(self.path, position,'tem_stage_pos_refined.csv')

        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'stage_x_um', 'stage_y_um', 'stage_z_um', 'stage_tilt'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({'name': name,'stage_x_um': stage_position[0],'stage_y_um': stage_position[1],'stage_z_um': stage_z,'stage_tilt': stage_tilt,})
    
    def _identify_montage_file(self):

        matches = list(Path(self.path).glob("*montage*.mrc"))
        if not matches:
            raise FileNotFoundError(f"No montage .mrc found in {self.path}")
        montage_path = str(max(matches, key=lambda p: p.stat().st_mtime))
        return montage_path
    
    def run_experiment_setup(self):
    
        input("[ToDO] Please load the experiment file and the settings file. ENTER")

        self.tem._reset_defocus()
        self.tem.precise_stage_move([0.0, 0.0, 0.0], tilt_angle=0.0)

        self.set_low_dose_imaging_state('LMM')
        self.tem.acquire_image(mode='View')

        input('[ToDO] Move feature suitable for eucentricity alignment to the center of the stage (shift + right click + drag). ENTER')
        
        self.tem.set_eucentricity(level='rough')

        self.tem.acquire_montage(imaging_state='atlas', fov_um_x=3000.0, fov_um_y=3000.0, position=None, tilt_angle=0.0)

    def run_acquire_position_montages(self):
        
        tem_stage_positions = self._import_csv_file('tem_stage_positions.csv')

        for group_ID, position in enumerate(tem_stage_positions):
            print(f"[INFO] Acquiring montage for {position["name"]}.")

            self.tem.precise_stage_move()
            input("Please move the center of the grid square / lamella to the center of the field of view. ENTER")
            self.tem.set_eucentricity(level="rough_fine")
            stage_position, stage_tilt = self.tem.report_stage_position()
            self._write_csv_file(position=position["name"], stage_position=stage_position[0:2], stage_z=stage_position[2], stage_tilt=stage_tilt, name=position["name"])
            self.tem.acquire_montage(fov_um_x=self.montage_settings[self.sample_type]['fov_um_x'], fov_um_y=self.montage_settings[self.sample_type]['fov_um_y'], low_dose=True, position=f"{position["name"]}_montage")

    def run_clem_alignment(self, position):
        montage_file = self._identify_montage_file(os.path.join(self.path, position))
        montage_summary = self.mrc.run_montage_loader_and_create_summary(os.path.join(self.path, montage_file))
        from clem_target_picking import CLEMPicker
        target_picker = CLEMPicker(montage_settings=montage_summary, tem_communication=self.tem)
        target_picker.run_auto_picker()

     
        
        