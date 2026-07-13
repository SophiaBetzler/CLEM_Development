import os
import sys
from pathlib import Path
sys.path.append(r"C:\Program Files\SerialEM\PythonModules")
print(sys.executable)
print(sys.version)
import serialem as sem
from datetime import datetime
import numpy as np
import math

class TEMComm:

    

    def __init__(self, path, mrc_reader, rotation=True, offline=False):
        self.output_root = path
        self.mrc_reader = mrc_reader
        self.rotation = rotation
        self.ACQUIRE = {
                        "View":    sem.View,
                        "Record":  sem.Record,
                        "Search":  sem.Search,
                        "Preview": sem.Preview,
                    }
        if offline:
            sem.NoMessageBoxOnError()
            print("[INFO] SerialEM is running in offline mode. No commands will be sent to the microscope.")
   

    # ---------------------------------------------------------------------------
    # control image acquisition and navigator handling
    # ---------------------------------------------------------------------------

    def _create_nav_file(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")

        if sem.ReportIfNavOpen() > 0:
            sem.CloseNavigator()
            
        nav_file = os.path.join(self.output_root, "nav_file" + "_" + timestamp + '.nav')
        sem.OpenNavigator(nav_file)

    def add_stage_pos_to_nav(self, picks):
        if sem.ReportIfNavOpen() == 0:
            self._create_nav_file()
        _, _, default_stage_z_um = sem.ReportStageXYZ()
        pt_id = int(sem.GetUniqueNavID())
        for pick in picks:
            if pick['stage_z_um'] is not None:
                sem.AddStagePosAsNavPoint(pick['stage_x_um'], pick['stage_y_um'], pick['stage_z_um'], pt_id)
            else:
                sem.AddStagePosAsNavPoint(pick['stage_x_um'], pick['stage_y_um'], default_stage_z_um, pt_id)
            sem.ChangeItemNote(int(pt_id), pick['pick_id'])

    def _load_mrc_in_nav(self, mrc_file_name=None, buffer="0", site_id=None):
        if sem.ReportIfNavOpen() == 0:
            self._create_nav_file()
        
        if mrc_file_name is None:
            mrc_file_name = self.mrc_reader.identify_montage_file(site_id=site_id)

        if site_id is not None:
            sem.ReadOtherFile(0, buffer, os.path.join(self.output_root, site_id, mrc_file_name))
        else:
            sem.ReadOtherFile(0, buffer, os.path.join(self.output_root, mrc_file_name))
        sem.NewMap()
        
    def _save_buffer_image(self, site_id=None, acquisition_type=None, label=None):
        timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
        mag, *_ = sem.ReportMag()

        filename_parts = [site_id, acquisition_type, label, f"mag_{mag}", timestamp, '.mrc']
        filename = "_".join(str(part).strip("_") for part in filename_parts if part is not None and str(part) != "")

        if site_id is not None:
            output_dir = os.path.join(self.output_root, site_id)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = self.output_root

        sem.OpenNewFile(output_dir, filename)
        sem.Save()                 
        sem.CloseFile()

    def get_image_properties(self, mode, imaging_state=None):
        self.prepare_imaging_state(mode=mode,imaging_state=imaging_state)
        self.ACQUIRE[mode]()
        image_x_pxl, image_y_pxl, binning, exposure, pxl_size_nm, param_set = sem.ImageProperties("A")
        magnification, *_ = self.sem.ReportMag()
        return {
                'img_width_px': int(image_x_pxl),
                'img_height_px': int(image_y_pxl),
                'pixel_size_um': pxl_size_nm/1000,
                'magnification': magnification,
                }
    
    def get_low_dose_mode_properties(self, mode):
        if self.offline:
            return{'mode': mode, 'magnification': 2000, 'pixel_size_um': 0.007}
        
        sem.GoToLowDoseArea(mode)
        sem.Delay(1, 'sec')
        magnification = int(self.sem.ReportMag()[0])
        return magnification


    def acquire_image(self, mode, imaging_state=None, save=False, site_id=None, label=None):
        self.prepare_imaging_state(mode=mode, imaging_state=imaging_state,)

        self.ACQUIRE[mode]()
        if save:
            self._save_buffer_image(site_id=site_id, acquisition_type="image",label=label)

    # ---------------------------------------------------------------------------
    # fundamental microscope controls
    # ---------------------------------------------------------------------------

    def _reset_defocus(self):
        mag_Index = sem.ReportMagIndex()[0]
        if mag_Index > 5:
            sem.SetEucentricFocus()
            sem.Delay(2, 'sec')
            sem.ResetDefocus()
            sem.Delay(2, 'sec')
        else:
            print("[INFO] Defocus reset and eucentric focus skipped because the microscope is in LM mode.")

    def apply_specimen_shift(self, mode, specimen_shift):
        sem.prepare_imaging_state(mode=mode)
        sem.Delay(1, 'sec')
        sem.ImageShiftByMicrons(specimen_shift[0], specimen_shift[1])
 
    def set_eucentricity(self, level):
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
        low_dose_mode_state = sem.ReportLowDoseMode()[0]
        if not low_dose_mode_state:
            sem.SetLowDoseMode(1)
            sem.Delay(1, 'sec')
            sem.NormalizeLenses(7)
            print("[INFO] Switchted to Low Dose Mode.")
        else:
            print("[INFO] Already in Low Dose Mode.")

    def set_non_low_dose_imaging_state(self, imaging_state):
        sem.SetLowDoseMode(0)
        sem.Delay(1, 'sec')
        defocus = {'grid_square': -10.0, 'LMM': -50.0}
        
        if imaging_state in ['LMM', 'grid_square']:
            sem.GoToImagingState(imaging_state)
            sem.Delay(2, 'sec')
            sem.NormalizeLenses(7)
            sem.Delay(5, 'sec')
            sem.SetDefocus(float(defocus[imaging_state]))
        else:
            raise ValueError(f"imaging_state {imaging_state} is not in the list of pre-defined imaging states.")

    def precise_stage_move(self, stage_x_um=None, stage_y_um=None, stage_z_um=None, stage_tilt=0.0):
        backlashTilt = 2.0
        backlashXY = 5.0

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

        backlashXY = 5.0

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

    def acquire_montage(self, fov_um_x, fov_um_y, stage_tilt=0.0, site_id=None, imaging_state=None):
        
        self.prepare_imaging_state(mode="View", imaging_state=imaging_state)
        
        self.set_eucentricity(level='rough_fine')

        image_x_pxl, image_y_pxl, binning, exposure, pxl_size_nm, param_set = self._readout_image_properties(mode="View")

        tile_fov_um_x = image_x_pxl * pxl_size_nm/1000
        tile_fov_um_y = image_y_pxl * pxl_size_nm/1000

        overlap_fraction = 0.15
        overlap_pxl_x = int(overlap_fraction * image_x_pxl)
        overlap_pxl_y = int(overlap_fraction * image_y_pxl)

        step_um_x = tile_fov_um_x * (1.0 - overlap_fraction)
        step_um_y = tile_fov_um_y * (1.0 - overlap_fraction)

        nx = max(1, math.ceil((fov_um_x - tile_fov_um_x) / step_um_x) + 1)
        ny = max(1, math.ceil((fov_um_y - tile_fov_um_y) / step_um_y) + 1)

        timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
        mag, *_ = sem.ReportMag()
        if site_id is not None:
            filepath = os.path.join(self.output_root, site_id, site_id + '_montage_' +'mag_' + str(mag) + '_' + timestamp + '.mrc')
            os.makedirs(os.path.join(self.output_root, site_id), exist_ok=True)
        else:
            filepath = os.path.join(self.output_root, 'Montage_' +'mag_' + str(mag) + '_' + timestamp + '.mrc')

        self.precise_stage_move(stage_tilt=stage_tilt)
        sem.OpenNewMontage(nx, ny, filepath)
        sem.SetMontageParams(int(1),    # useStage = 1 (stage montage, required for hybrid usage of both image shift and stage shift)
                        overlap_pxl_x,    # overlap in x in pxl
                        overlap_pxl_y,    # overlap in y in pxl
                        int(image_x_pxl),     # set to image size currently in buffer
                        int(image_y_pxl),     # set to image size currently in buffer
                        0,              # skip correlation
                        int(binning),
                        -1.0)            # max image shift in micron, only relevant when first argument is 2, otherwise set to -1

        sem.Montage()
        
        if site_id is not None:
            sem.NewMap(0, site_id + '_montage_' + timestamp)
        else:
            sem.NewMap(0, 'Montage' + '_' + timestamp)
            
    # ---------------------------------------------------------------------------
    # Alignment routines
    # ---------------------------------------------------------------------------

    def run_serialem_alignment_routine(self, buffer, mode, pick_id, mag_compensation=True, debug_display=False, eucentric=False, site_id=None):

        if eucentric is True:
            self.set_eucentricity(level='fine')

        if debug_display is True:
            debug = int(1)
        else:
            debug = int(0)

        if mag_compensation is True:
            self.sem.AlignBetweenMags(buffer, -1, -1, -1, 0, -1, int(0)) # buffer, center X in the reference, center Y in the reference, max allowed shift (negative means field of view, positive microns), scale (default: 4%), rotation (default: 3 degrees), avoid_image_shift
            self._save_buffer_image(site_id=site_id, acquisition_type='mag_adjusted_image', label=pick_id)
            buffer = "Q"
            self.sem.Copy("B", buffer) 

        max_iter = 5
        iteration = 0
        stage_shift = np.array([np.inf, np.inf])
        
        while np.linalg.norm(stage_shift) > 0.1 and iteration < max_iter:
            self.acquire_image(mode=mode, save=False, site_id=site_id, label=f"pre_align_{pick_id}")
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
        self.acquire_image(mode=mode, save=True, site_id=site_id, label=f"{pick_id}_aligned_target_{mode}")
        stage_x_um, stage_y_um, stage_z_um = sem.ReportStageXYZ()
        return {"site_id": site_id, "pick_id": pick_id, "stage_x_um": stage_x_um, "stage_y_um": stage_y_um, "stage_z_um": stage_z_um, "mode": mode}     

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
    

    def run_picks_visualization(self, mrc_file_name, picks, site_id=None):
        self._load_mrc_in_nav(mrc_file_name=mrc_file_name,buffer='A', site_id=site_id)
        self.add_stage_pos_to_nav(picks=picks)

