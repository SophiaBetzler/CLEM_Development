import os
import argparse
import time
import sys
from pathlib import Path
sys.path.append(r"C:\Program Files\SerialEM\PythonModules")
print(sys.executable)
print(sys.version)
import serialem as sem
from datetime import datetime
import numpy as np
import math
import tkinter as tk

class TEMComm:

    

    def __init__(self, path, mrc_reader, offline=False):
        self.output_root = path
        self.mrc_reader = mrc_reader
        self.offline = offline
        self.ACQUIRE = {
                        "View":    sem.View,
                        "Record":  sem.Record,
                        "Search":  sem.Search,
                        "Preview": sem.Preview,
                    }
        if self.offline:
            sem.NoMessageBoxOnError()
            print("[INFO] SerialEM is running in offline mode. No commands will be sent to the microscope.")
   

    # ---------------------------------------------------------------------------
    # navigator handling and data storage
    # ---------------------------------------------------------------------------

    def create_nav_file(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")

        if sem.ReportIfNavOpen() > 0:
            sem.CloseNavigator()
            
        nav_file = os.path.join(self.output_root, "nav_file" + "_" + timestamp + '.nav')
        sem.OpenNavigator(nav_file)


    def add_nav_point(self, mrc_dataclass, buffer="S"):
        self.set_montage_to_buffer(note=mrc_dataclass.montage_id, buffer=buffer)
        if len(mrc_dataclass.groups) == 0 and len(mrc_dataclass.picks) != 0:
            for pick in mrc_dataclass.picks[:-1]:
                idx = []
                idx = sem.AddImagePosAsNavPoint(buffer, pick.image_coord_x, pick.image_coord_y, mrc_dataclass.stage_z_um, 0, 1)
                sem.ChangeItemNote(idx, pick.pick_id)
            idx = sem.AddImagePosAsNavPoint(buffer, mrc_dataclass.picks[-1].image_coord_x, mrc_dataclass.picks[-1].image_coord_y, mrc_dataclass.stage_z_um, 0, 0)
            sem.ChangeItemNote(idx, mrc_dataclass.picks[-1].pick_id)
        elif len(mrc_dataclass.groups) != 0:
            for group in mrc_dataclass.groups:
                for pick in mrc_dataclass.picks[:-1]:
                    idx = []
                    idx = sem.AddImagePosAsNavPoint(buffer, pick.image_coord_x, pick.image_coord_y, group.group_id, 1)
                    sem.ChangeItemNote(idx, pick.pick_id)
                idx = sem.AddImagePosAsNavPoint(buffer, mrc_dataclass.picks[-1].image_coord_x, mrc_dataclass.picks[-1].image_coord_y, mrc_dataclass.stage_z_um, group.group_id, 0)
                sem.ChangeItemNote(idx, mrc_dataclass.picks[-1].pick_id)
        elif len(mrc_dataclass.groups) == 0 and len(mrc_dataclass.picks) == 0:
            raise ValueError(f"No picks exists for this montage {mrc_dataclass.montage_id}.")
        else:
            return

    def load_mrc_in_nav(self, site_data=None, mrc_dataclass=None, buffer=None):

        if mrc_dataclass is None:
            mrc_dataclass = self.mrc_reader.identify_latest_montage_file(site_data)
                
        if mrc_dataclass.mrc_path.lower().endswith(".mdoc"):
            mrc_path = mrc_path[:-len(".mdoc")]
        else: 
            mrc_path = mrc_dataclass.mrc_path.lower()

        if sem.ReportIfNavOpen() == 0: self._create_nav_file()

        idx =int(sem.NavIndexWithNote(mrc_dataclass.montage_id))

        if buffer is None: buffer = 'A'
        if idx > 0:
            sem.LoadOtherMap(idx, buffer)
        else:
            while sem.ReportFileNumber() > 0:
                sem.CloseFile()
            try:
                sem.OpenOldFile(mrc_path)
                sem.ReadFile(0)
                sem.NewMap(0, mrc_dataclass.montage_id)
                if buffer != "A": sem.Copy("A", buffer)
            except Exception as e:
                print(f"{e}")
        
    def save_buffer_image(self, site_data, label=None):
        timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
        mag, *_ = sem.ReportMag()
        filename_parts = [site_data.site_id, label, f"mag_{mag}", timestamp, '.mrc']
        filename = "_".join(str(part).strip("_") for part in filename_parts if part is not None and str(part) != "")
        os.makedirs(site_data.path, exist_ok=True)
        file_path = os.path.join(site_data.path, filename)
        sem.OpenNewFile(file_path)
        sem.Save()                 
        sem.CloseFile()
        return filename, timestamp

    def find_nav_item_with_note(self, note):
        idx = int(sem.NavIndexWithNote(note))
        return idx if idx > 0 else None
    
    def set_montage_to_buffer(self, note, buffer='S'):
        try:
            map_idx = int(sem.NavIndexWithNote(note))
        except Exception:
            raise RuntimeError(f"Could not find nav item with note {note}. Please ensure the montage is loaded in the navigator.")
        if map_idx > 0:
            sem.LoadOtherMap(map_idx, buffer)
        
    # ---------------------------------------------------------------------------
    # export image properties
    # ---------------------------------------------------------------------------

    def get_image_properties(self, mode, imaging_state=None):

        self.prepare_imaging_state(mode=mode,imaging_state=imaging_state)
        self.ACQUIRE[mode]()
        sem.Delay(1, 'sec')
        image_x_pxl, image_y_pxl, binning, exposure, pxl_size_nm, param_set = sem.ImageProperties("A")
        magnification= int(sem.ReportMag()[0])
        return {
                'img_width_px': int(image_x_pxl),
                'img_height_px': int(image_y_pxl),
                'pixel_spacing_um': pxl_size_nm/1000,
                'magnification': magnification,
                'binning': int(binning),
                'exposure': exposure,
                'param_set': param_set,
                }

    # ---------------------------------------------------------------------------
    # pause serialEM execution
    # ---------------------------------------------------------------------------
    
    def wait_for_continue_trigger(self, message):
        win = tk.Tk()
        win.title("Paused")
        win.attributes("-topmost", True)
        tk.Label(win, text=message, padx=20, pady=12, wraplength=320).pack()
        tk.Button(win, text="Continue", width=16, command=win.quit).pack(pady=(0,12))
        win.protocol("WM_DELETE_WINDOW", win.quit)
        win.mainloop()
        win.destroy()

        
    def return_control_to_serialem(self, message):
        try:
            sem.Exit(0)
        except Exception:
            pass
        self.wait_for_continue_trigger(message)
        print("[INFO] Re-connecting to serialEM.")
        sem.ConnectToSEM()
        
    # ---------------------------------------------------------------------------
    # fundamental microscope controls
    # ---------------------------------------------------------------------------

    def acquire_image(self, mode, imaging_state=None, save=False, site_data=None, label=None, buffer=None):

        if self.offline:
            print(f"[INFO] Image acquisition triggered with {imaging_state}, {mode}.")
            return
        
        self.prepare_imaging_state(imaging_state=imaging_state, mode=mode)
        self.ACQUIRE[mode]()


        image_properties = self.get_image_properties(mode=mode, imaging_state=imaging_state)

        if save:
            filename, timestamp = self.save_buffer_image(site_data, label=label)
            if sem.ReportIfNavOpen() == 0:
                self._create_nav_file()
            nav_idx = int(sem.NewMap(0, filename))
                          
            image_parameters = {'image_path': os.path.join(site_data.path,  filename),
                                'image_id' : filename,
                                'image_height' : image_properties["image_height_px"],
                                'image_width' : image_properties["image_width_px"],
                                'pixel_spacing_um' : image_properties["pixel_spacing_um"],
                                'magnification' : image_properties["magnificiation"],
                                'stage_to_camera_matrix' : self.report_stage_to_camera_matrix(),
                                'is_to_camera_matrix' : self.report_image_shift_to_camera_matrix(),
                                'timestamp' : timestamp,
                                }
            site_data.add_image(label=label, image_parameters=image_parameters)

    def reset_defocus(self):
        if self.offline:
            print("[INFO] Running reset defocus.")
            return
        if sem.ReportLowDose()[0] == 0:
            sem.SetEucentricFocus()
            sem.Delay(2, 'sec')
            sem.ResetDefocus()
            sem.Delay(2, 'sec')
        else:
            print("[INFO] Defocus reset and eucentric focus skipped because the microscope is in LM mode.")

    def set_eucentricity(self, level):
        if self.offline:
            print("[INFO] Eucentricity routine.")
            return
        self._reset_defocus()
        eucentricity_settings = {'fine': 2, 'rough': 1, 'rough_fine': 3}
        sem.Eucentricity(eucentricity_settings[level])

    def report_stage_position(self):
        stage_x, stage_y, stage_z = sem.ReportStageXYZ()
        stage_tilt = sem.ReportTiltAngle()
        return stage_x, stage_y, stage_z, stage_tilt

    def prepare_imaging_state(self, mode=None, imaging_state=None):
        if imaging_state is None:
            self.set_low_dose_imaging_state()
            if mode is not None:
                sem.GoToLowDoseArea(mode)
            return "lowdose"
        else:
            self.set_non_low_dose_imaging_state(imaging_state)
        sem.Delay(2, 'sec')
        return imaging_state

    def set_low_dose_imaging_state(self):
        low_dose_mode_state = sem.ReportLowDose()[0]
        if low_dose_mode_state:
            return
        else: 
            sem.SetLowDoseMode(1)
            sem.Delay(1, 'sec')
            if self.offline is False:
                sem.NormalizeLenses(7)
            
    def set_non_low_dose_imaging_state(self, imaging_state):
        low_dose_mode_state = sem.ReportLowDose()[0]
        if low_dose_mode_state: 
            sem.SetLowDoseMode(0)
            sem.Delay(1, 'sec')
        
        defocus = {'grid_square': -10.0, 'LMM': -50.0}
        if self.offline is False:
            sem.ResetImageShift()
        
        if imaging_state in ['LMM', 'grid_square']:
            sem.GoToImagingState(imaging_state)
            sem.Delay(1, 'sec')
            if self.offline is False:
                sem.NormalizeLenses(7)
            sem.Delay(1, 'sec')
            sem.SetDefocus(float(defocus[imaging_state]))
        else:
            raise ValueError(f"imaging_state {imaging_state} is not in the list of pre-defined imaging states.")

    def precise_stage_move(self, stage_x_um=None, stage_y_um=None, stage_z_um=None, stage_tilt=0.0):

        if self.offline:
            print(f"[INFO] Stage move to {stage_x_um, stage_y_um, stage_z_um, stage_tilt}")

        backlashTilt = 2.0
        backlashXY = 2.0

        if stage_tilt is not None:
            current_tilt = sem.ReportTiltAngle()
            if current_tilt != stage_tilt:
                if current_tilt > stage_tilt:
                    backlashTilt = backlashTilt * (-1)
                sem.TiltTo(stage_tilt + backlashTilt)
                sem.TiltTo(stage_tilt)
                sem.Delay(1, 'sec')

        if stage_x_um is not None and stage_y_um is not None:
            if stage_z_um is None:
                sem.MoveStageTo(stage_x_um - backlashXY, stage_y_um - backlashXY)
                sem.Delay(1, 'sec')
                sem.MoveStageTo(stage_x_um, stage_y_um)
            else:
                sem.MoveStageTo(stage_x_um - backlashXY, stage_y_um - backlashXY, stage_z_um)
                sem.Delay(1, 'sec')
                sem.MoveStageTo(stage_x_um, stage_y_um, stage_z_um)
                     
        sem.Delay(2, 'sec')
    
    def precise_stage_move_relative(self, stage_x_um, stage_y_um, stage_z_um=None):

        backlashXY = 2.0

        if stage_x_um is None or stage_y_um is None:
            return

        if stage_z_um is None:
            sem.MoveStage(stage_x_um - backlashXY, stage_y_um - backlashXY)
            sem.Delay(1, 'sec')
            sem.MoveStage(backlashXY, backlashXY)
        else:
            sem.MoveStage(stage_x_um - backlashXY, stage_y_um - backlashXY, stage_z_um)
            sem.Delay(1, 'sec')
            sem.MoveStage(backlashXY, backlashXY, 0.0)


    # ---------------------------------------------------------------------------
    # Montage control
    # ---------------------------------------------------------------------------
    
    def acquire_montage_at_nav_item(self, site_data, nav_idx, fov_um_x, fov_um_y, mode='Search', imaging_state=None, eucentricity=False, label=None):
        if self.offline is False:
            sem.MoveToNavItem(nav_idx)
            sem.Delay(2, 'sec')
        self.acquire_montage(site_data=site_data, mode=mode, fov_um_x=fov_um_x, fov_um_y=fov_um_y, imaging_state=imaging_state, eucentricity=eucentricity, label=label)


    def acquire_montage(self, site_data, fov_um_x, fov_um_y, mode='Search', imaging_state=None, eucentricity=False, label=None):

        if self.offline and site_data is not None:
            print(f"Montage collected for {site_data.site_id}.")
        elif self.offline:
            print(f"Acquiring Montage.")

        self.prepare_imaging_state(mode=mode, imaging_state=imaging_state)
        
        if eucentricity is True:
            self.set_eucentricity(level='rough_fine')
        
        image_properties = self.get_image_properties(mode=mode, imaging_state=imaging_state)

        tile_fov_um_x = image_properties["img_width_px"] * image_properties["pixel_spacing_um"]
        tile_fov_um_y = image_properties["img_height_px"] * image_properties["pixel_spacing_um"]

        overlap_fraction = 0.15
        overlap_pxl_x = int(overlap_fraction * image_properties["img_width_px"])
        overlap_pxl_y = int(overlap_fraction * image_properties["img_height_px"])

        step_um_x = tile_fov_um_x * (1.0 - overlap_fraction)
        step_um_y = tile_fov_um_y * (1.0 - overlap_fraction)

        nx = max(1, math.ceil((fov_um_x - tile_fov_um_x) / step_um_x) + 1)
        ny = max(1, math.ceil((fov_um_y - tile_fov_um_y) / step_um_y) + 1)

        timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
        mag, *_ = sem.ReportMag()
        if site_data is not None and site_data.site_id is not None:
            filepath = os.path.join(self.output_root, site_data.site_id, site_data.site_id + '_montage_' +'mag_' + str(mag) + '_' + timestamp + '.mrc')
            os.makedirs(os.path.join(self.output_root, site_data.site_id), exist_ok=True)
        else:
            filepath = os.path.join(self.output_root, 'Montage_' +'mag_' + str(mag) + '_' + timestamp + '.mrc')

        if site_data is not None:
            self.precise_stage_move(stage_tilt=site_data.milling_angle)
        else:
            self.precise_stage_move(stage_tilt=0.0)
        
        if imaging_state is None:
            if mode == 'View':
                print('imaging mode is view')
                sem.ParamSetToUseForMontage(2, 1)
            elif mode == 'Search':
                print('imaging mode is search')
                sem.ParamSetToUseForMontage(3, 1)
            else:
                raise ValueError('Not a valid imaging mode for a montage.')
        print(filepath)    
        sem.OpenNewMontage(nx, ny, filepath)
        sem.SetMontageParams(int(1),    # useStage = 1 (stage montage, required for hybrid usage of both image shift and stage shift)
                        overlap_pxl_x,    # overlap in x in pxl
                        overlap_pxl_y,    # overlap in y in pxl
                        int(image_properties["img_width_px"]),     # set to image size currently in buffer
                        int(image_properties["img_height_px"]),     # set to image size currently in buffer
                        0,              # skip correlation
                        int(image_properties["binning"]),
                        -1.0)  
        
                  # max image shift in micron, only relevant when first argument is 2, otherwise set to -1

        sem.Montage()
        if site_data is not None and site_data.site_id is not None:
            montage_id = f"{site_data.site_id}_montage_{timestamp}"                  
        else:
            montage_id = f"Montage_{timestamp}"

        sem.NewMap(0, montage_id)
        if site_data is not None:
            self.mrc_reader.build_montage_summary(site_data=site_data, label=label, timestamp=timestamp)

   ### HAVEN'T CLEANED UP BELOW YET - NEEDS TO BE REWRITTEN TO USE NEW ALIGNMENT FUNCTION
            
    # ---------------------------------------------------------------------------
    # Alignment routines
    # ---------------------------------------------------------------------------

    def run_serialem_alignment_routine(self, buffer, mode, label=None, mag_compensation=True, debug_display=False, eucentric=False):

        if eucentric is True:
            self.set_eucentricity(level='fine')

        if debug_display is True:
            debug = int(1)
        else:
            debug = int(0)

        if mag_compensation is True:
            sem.AlignBetweenMags(buffer, -1, -1, -1, 0, -1, int(0)) # buffer, center X in the reference, center Y in the reference, max allowed shift (negative means field of view, positive microns), scale (default: 4%), rotation (default: 3 degrees), avoid_image_shift
            self._save_buffer_image(acquisition_type='mag_adjusted_image', label=label)
            buffer = "Q"
            sem.Copy("B", buffer) 

        max_iter = 5
        iteration = 0
        stage_shift = np.array([np.inf, np.inf])
        
        while np.linalg.norm(stage_shift) > 0.1 and iteration < max_iter:
            self.acquire_image(mode=mode, save=False, label=f"pre_align_{label}")
            sem.AlignTo(buffer, int(1), int(0), int(0), debug) #don't apply imageshift, #trimming, #correlation peak handling
            required_shift = sem.ReportAlignShift()
            stage_shift = np.array(required_shift[4:6]) / 1000
            ss2s = np.array(sem.SpecimenToStageMatrix(0)).reshape((2, 2))
            applied_stage_shift = ss2s @ stage_shift     
            self.precise_stage_move_relative(stage_x_um=applied_stage_shift[0], stage_y_um=applied_stage_shift[1])
            iteration += 1 
            if debug_display is True:
                sem.AddBufToStackWindow("A", 0, 0, 0, 0, f"CC_{iteration}")
                sem.Copy("B", "A")

        if iteration == 5:
            raise RuntimeError("Align routine didn't converge within 5 iterations.")
        
        self.acquire_image(mode=mode)
        sem.AlignTo(buffer, int(0), int(0), int(0), debug) 
        shift = sem.ReportAlignShift()
        print(f"[INFO] The misalignment determined for the final image is{np.linalg.norm(shift[4:6])/1000} um.")
        if np.linalg.norm(shift[4:6]) > 0.1:
            raise RuntimeError("Alignment failed.")
        stage_x_um, stage_y_um, stage_z_um = self.tem.report_stage_position()[:3]

        return stage_x_um, stage_y_um, stage_z_um     

    def align_target_at_higher_mag(self, reference_image_path, target_stage_pos=None, pick_id=None, mode ='Record', label='None'):
        
        buffer = 'P'  # Persistent buffer
        
        if label is None and pick_id is not None:
            label = pick_id
        
        print(f"[INFO] Loading reference and aligning {label}...")

        self.load_mrc_in_nav(reference_image_path, buffer=buffer)
        if target_stage_pos is not None:
            self.precise_stage_move(stage_x_um=target_stage_pos[0], stage_y_um=target_stage_pos[1], stage_z_um=target_stage_pos[2])
        self.acquire_image(mode=mode)

        refined_stage_x, refined_stage_y, refined_stage_z = self.run_serialem_alignment_routine(buffer=buffer, label=label, mode=mode, mag_compensation=True)

        if pick_id is not None:
            return {
                        'pick_id': pick_id,
                        'refined_stage': (refined_stage_x, refined_stage_y, refined_stage_z),
                    }
        else:
            (refined_stage_x, refined_stage_y, refined_stage_z)



    # ---------------------------------------------------------------------------
    # Functions for troubleshooting and debugging
    # ---------------------------------------------------------------------------

    def show_nav_adjustment(self):
        rows = []
        for i, p in enumerate(self.picks, start=1):
            coord = p.get(self.rotation)
            if coord is None:
                continue
            sx, sy = float(coord[0]), float(coord[1])
            ax, ay = sem.AdjustStagePosForNav(sx, sy)
            ax, ay = float(ax), float(ay)
            rows.append({"pick_id": i, "stage_z_um": p.get("stage_z_um"),
                        "raw_stage_xy_um": (sx, sy), "adjusted_stage_xy_um": (ax, ay),
                        "delta_stage_xy_um": (ax - sx, ay - sy)})

        if not rows:
            print(f"No picks with a '{self.rotation}' coordinate to test.")
            return rows

        print(f"\nAdjustStagePosForNav effect (coord = '{self.rotation}')")
        print(f"{'#':>3}  {'stage_z_um':>8}  "
            f"{'stage_x_um':>10} {'stage_y_um':>10}   "
            f"{'adjusted_stage_x':>10} {'adjusted_stage_y':>10}   "
            f"{'delta_stage_x':>8} {'delta_stage_y':>8}")
        print("-" * 86)
        for r in rows:
            zs = "n/a" if r["stage_z_um"] is None else f"{r['stage_z_um']:.3f}"
            print(f"{r['pick_id']:>3}  {zs:>8}  "
                f"{r['raw_stage_xy_um'][0]:>10.3f} {r['raw_stage_xy_um'][1]:>10.3f}   "
                f"{r['adjusted_stage_xy_um'][0]:>10.3f} {r['adjusted_stage_xy_um'][1]:>10.3f}   "
                f"{r['delta_stage_xy_um'][0]:>8.4f} {r['delta_stage_xy_um'][1]:>8.4f}")

        dxs = [r["delta_stage_xy_um"][0] for r in rows]
        dys = [r["delta_stage_xy_um"][1] for r in rows]
        print("-" * 86)
        print(f"mean delta : ({sum(dxs)/len(dxs):.4f}, {sum(dys)/len(dys):.4f}) um")
        print(f"delta spread (max-min): "
            f"({max(dxs)-min(dxs):.4g}, {max(dys)-min(dys):.4g}) um")
        if (max(dxs)-min(dxs)) < 1e-3 and (max(dys)-min(dys)) < 1e-3:
            print("=> delta CONSTANT: rigid shift, relative geometry preserved.")
        else:
            print("=> delta VARIES: not a pure rigid shift (check mag/tilt/IS state).")

        zvals = [r["stage_z_um"] for r in rows if r["stage_z_um"] is not None]
        if not zvals:
            print("NOTE: stage_z is None for all picks - mdoc has no stage Z, "
                "supply a default before AddStagePosAsNavPoint.")
        return rows
    

