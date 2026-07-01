import os
import sys
sys.path.append(r"C:\Program Files\SerialEM\PythonModules")
print(sys.executable)
print(sys.version)
import serialem as sem
from datetime import datetime
import numpy as np
import math

class TEMComm:

    ACQUIRE = {
                    "View":    sem.View,
                    "Record":  sem.Record,
                    "Search":  sem.Search,
                    "Preview": sem.Preview,
                }

    def __init__(self, path, rotation=True):
        self.path = path
        self.rotation = rotation

    # ---------------------------------------------------------------------------
    # control image acquisition and navigator handling
    # ---------------------------------------------------------------------------

    def _create_nav_file(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")

        if sem.ReportIfNavOpen() > 0:
            sem.CloseNavigator()
            
        nav_file = os.path.join(self.path, "nav_file" + "_" + timestamp + '.nav')
        sem.OpenNavigator(nav_file)

    def _load_mrc_in_nav(self, mrc_file, path=None, buffer="0"):
        if sem.ReportIfNavOpen() == 0:
            self._create_nav_file()
        sem.CloseFile()
        if path is not None:
            sem.ReadOtherFile(0, buffer, os.path.join(path, mrc_file))
        sem.NewMap()
        
    def _save_buffer_image(self, position=None, token=None):
        timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
        mag, *_ = sem.ReportMag()
        if position is not None:
            filepath = os.path.join(self.path, position, position + '_mag_' + str(mag) + '_' + timestamp + '.mrc')
        elif token is not None:
            filepath = os.path.join(self.path, position, position + token + '_mag' + str(mag) + '_' + timestamp + '.mrc')
        else:
            filepath = os.path.join(self.path, 'Mag_' + str(mag) + '_' + timestamp + '.mrc')
        sem.OpenNewFile(filepath)
        sem.Save()                 
        sem.CloseFile()

    def acquire_image(self, mode, lowdose=True, save=False, position=None, token=None):
        if lowdose:
            self.set_low_dose_imaging_state()
            self.GoToLowDoseArea(mode)
        else:
            self.set_non_low_dose_imaging_state()

        self.AQUIRE[mode]
        if save:
            self._save_buffer_image(position=position, token=token)


    def _readout_image_properties(self, mode):
        self.AQUIRE[mode]
        image_x_pxl, image_y_pxl, binning, exposure, pxl_size_nm, param_set = sem.ImageProperties("A")
        return image_x_pxl, image_y_pxl, binning, exposure, pxl_size_nm, param_set
    
    def report_stage_position(self):
        stage_x, stage_y, stage_z = sem.ReportStageXYZ()
        stage_tilt = sem.ReportTiltAngle()
        return (stage_x, stage_y, stage_z), stage_tilt

    # ---------------------------------------------------------------------------
    # fundamental microscope controls
    # ---------------------------------------------------------------------------

    def _reset_defocus(self):
        sem.SetEucentricFocus()
        sem.Delay(2, 'sec')
        sem.ResetDefocus()
        sem.Delay(2, 'sec')

    def apply_specimen_shift(self, mode, specimen_shift):
        sem.GoToLowDoseArea(mode)
        sem.Delay(1, 'sec')
        sem.ImageShiftByMicrons(specimen_shift[0], specimen_shift[1])
 
    def set_eucentricity(self, level):
        self._reset_defocus()
        eucentricity_settings = {'fine': 2, 'rough': 1, 'rough_fine': 3}
        sem.Eucentricity(eucentricity_settings[level])

    def _set_low_dose_imaging_state(self):
        sem.SetLowDoseMode(0)
        sem.Delay(1, 'sec')
        sem.NormalizeLenses(7)
        print("[INFO] Switchted to Low Dose Mode.")

    def _set_non_low_dose_imaging_state(self, imaging_state, defocus_multiplier):
        sem.SetLowDoseMode(0)
        sem.Delay(1, 'sec')

        image_shift = {
                   'grid_square': (2.688, -5.400),
                   'LMM': (0.0, 0.0)
                        }
        defocus = {
                   'grid_square': -10.0,
                   'LMM': -50.0
                        }
        
        if imaging_state in ['LMM', 'grid_square']:
            sem.GoToImagingState(imaging_state)
            sem.Delay(2, 'sec')
            sem.NormalizeLenses(7)
            sem.Delay(2, 'sec')
            sem.SetDefocus(defocus[imaging_state]*defocus_multiplier)
            sem.SetImageShift(image_shift[imaging_state][0], image_shift[imaging_state][1])
        else:
            raise ValueError(f"imaging_state {imaging_state} is not in the list of pre-defined imaging states.")

        print(f"[INFO] Set magnficiation to {sem.ReportMag()[0]}, defocus to {defocus[imaging_state]*defocus_multiplier}, image shift to {sem.ReportImageShift()[0:2]}", )


    def precise_stage_move(self, stage_position=None, tilt_angle=None):
        backlashTilt = 2.0

        if tilt_angle is not None:
            current_tilt = sem.ReportTiltAngle()
            if current_tilt != tilt_angle:
                if current_tilt > tilt_angle:
                    backlashTilt = backlashTilt * (-1)
                sem.TiltTo(tilt_angle) + backlashTilt
                sem.TiltTo(tilt_angle)
                sem.Delay(1, 'sec')
        if stage_position is not None:
            backlashX = 2.0
            backlashY = 2.0
            if len(stage_position) == 3:
                sem.MoveStageTo(stage_position[0], stage_position[1], stage_position[2], backlashX, backlashY)
            else:
                sem.MoveStageTo(stage_position[0], stage_position[1], backlashX, backlashY)
            sem.Delay(2, 'sec')

    # ---------------------------------------------------------------------------
    # Montage control
    # ---------------------------------------------------------------------------

    def acquire_montage(self, imaging_state, fov_um_x, fov_um_y, tilt_angle, low_dose=True, position=None):
        if low_dose:
            self._set_low_dose_imaging_state(imaging_state)
        else:
            self._set_non_low_dose_imaging_state(imaging_state)

        self.precise_stage_move(tilt_angle=tilt_angle)
        self.set_eucentricity(level='rough')
        image_x_pxl, image_y_pxl, binning, exposure, pxl_size_nm, param_set = self._readout_image_properties(mode="Record")

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
        if position is not None:
            filepath = os.path.join(self.path, position, position + '_mag_' + str(mag) + '_montage_' + timestamp + '.mrc')
        else:
            filepath = os.path.join(self.path, 'Mag_' + str(mag) + '_montage_' + timestamp + '.mrc')
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
        if position is not None:
            sem.NewMap(0, position + '_' + timestamp)
        else:
            sem.NewMap(0, 'montage' + '_' + timestamp)

    def show_nav_adjustment(self):
        rows = []
        for i, p in enumerate(self.picks, start=1):
            coord = p.get(self.rotation)
            if coord is None:
                continue
            sx, sy = float(coord[0]), float(coord[1])
            ax, ay = sem.AdjustStagePosForNav(sx, sy)
            ax, ay = float(ax), float(ay)
            rows.append({"n": i, "stage_z": p.get("stage_z"),
                        "raw": (sx, sy), "adjusted": (ax, ay),
                        "delta": (ax - sx, ay - sy)})

        if not rows:
            print(f"No picks with a '{self.rotation}' coordinate to test.")
            return rows

        print(f"\nAdjustStagePosForNav effect (coord = '{self.rotation}')")
        print(f"{'#':>3}  {'stageZ':>8}  "
            f"{'raw X':>10} {'raw Y':>10}   "
            f"{'adj X':>10} {'adj Y':>10}   "
            f"{'dX':>8} {'dY':>8}")
        print("-" * 86)
        for r in rows:
            zs = "n/a" if r["stage_z"] is None else f"{r['stage_z']:.3f}"
            print(f"{r['n']:>3}  {zs:>8}  "
                f"{r['raw'][0]:>10.3f} {r['raw'][1]:>10.3f}   "
                f"{r['adjusted'][0]:>10.3f} {r['adjusted'][1]:>10.3f}   "
                f"{r['delta'][0]:>8.4f} {r['delta'][1]:>8.4f}")

        dxs = [r["delta"][0] for r in rows]
        dys = [r["delta"][1] for r in rows]
        print("-" * 86)
        print(f"mean delta : ({sum(dxs)/len(dxs):.4f}, {sum(dys)/len(dys):.4f}) um")
        print(f"delta spread (max-min): "
            f"({max(dxs)-min(dxs):.4g}, {max(dys)-min(dys):.4g}) um")
        if (max(dxs)-min(dxs)) < 1e-3 and (max(dys)-min(dys)) < 1e-3:
            print("=> delta CONSTANT: rigid shift, relative geometry preserved.")
        else:
            print("=> delta VARIES: not a pure rigid shift (check mag/tilt/IS state).")

        zvals = [r["stage_z"] for r in rows if r["stage_z"] is not None]
        if not zvals:
            print("NOTE: stage_z is None for all picks - mdoc has no stage Z, "
                "supply a default before AddStagePosAsNavPoint.")
        return rows
    
    def add_stage_pos_to_nav(self, picks):
        if sem.ReportIfNavOpen() == 0:
            self._create_nav_file()
        _, _, default_z = sem.ReportStageXYZ()
        pt_id = int(sem.GetUniqueNavID())
        for p in picks:
            sx, sy = p["stage"]
            z = p["stage_z"] if p["stage_z"] is not None else default_z
            sem.AddStagePosAsNavPoint(sx, sy, z, pt_id)

    def run_picks_visualization(self, mrc_file, picks):
        self._load_mrc_in_nav(mrc_file=mrc_file, path=None)
        self.add_stage_pos_to_nav(picks=picks)



    def acquire_image(self, mode):

        if mode in ["View", "Search", "Record", "Preview"]:
            sem.GoToLowDoseArea(mode)
        else:
            raise ValueError("Selected acquisition mode not available.")

        if mode == 'View':
            sem.V()
        elif mode == 'Record':
            sem.R()
        elif mode == 'Preview':
            sem.Preview()
        elif mode == 'Search':
            sem.S()
        else:
            raise ValueError("Selected acquisition mode no available.")
        
        imgX, imgY, _, live_img_px = sem.ImageProperties()
        return (imgX, imgY, live_img_px)
        

    def convert_stage_img_pos(self, buf, direction, coords):
        if direction == 'to_stage':
            sem.BufImagePosToStagePos(buf, 1, coords[0], coords[1])[:2]
        elif direction == 'to_img':
            sem.StagePosToBufImagePos(buf, 1, coords[0], coords[1])

    def run_serialem_alignment_routine(self, buffer, mode, pick_id, mag_compensation=True, debug_display=False, eucentric=False):

        if eucentric is True:
            self.set_eucentricity(level='fine')

        if debug_display is True:
            debug = int(1)
        else:
            debug = int(0)

        if mag_compensation is True:
            sem.AlignBetweenMags(buffer, -1, -1, -1, 0, -1, int(0)) # buffer, center X in the reference, center Y in the reference, max allowed shift (negative means field of view, positive microns), scale (default: 4%), rotation (default: 3 degrees), avoid_image_shift
            self._save_buffer_image(token=f"mag_adjusted_{pick_id}")
            buffer = "Q"
            sem.Copy("B", buffer) 

        max_iter = 5
        iteration = 0
        ss_shift = np.array([np.inf, np.inf])
        
        while np.linalg.norm(ss_shift) > 0.5 and iteration < max_iter:
            self.acquire_image(mode=mode)
            sem.AlignTo(buffer, int(0), int(0), int(0), debug) #don't apply imageshift, #trimming, #correlation peak handling
            shift = sem.ReportAlignShift()
            ss_shift = np.array(shift[4:6]) / 1000
            ss2s = np.array(sem.SpecimenToStageMatrix(0)).reshape((2, 2))
            stage_shift = ss2s @ ss_shift     
            #sem.MoveStage(-stage_shift[0], -stage_shift[1])
            sem.TestRelaxingStage(-stage_shift[0], -stage_shift[1], 3.0) # last number is the backlash correction
            iteration += 1 
            if debug_display is True:
                sem.AddBufToStackWindow("A", 0, 0, 0, 0, f"CC_{iteration}")
                sem.Copy("B", "A")

        if iteration == 5:
            raise RuntimeError("Align routine didn't converge within 5 iterations.")
        
        self.acquire_image(mode=mode)
        sem.AlignTo(buffer, int(0), int(0), int(1), debug) #use image shift to compensate
        shift = sem.ReportAlignShift()
        print(f"[INFO] The final shift is {np.linalg.norm(shift[4:6])/1000} um.")
        if np.linalg.norm(shift[4:6]) > 0.5:
            raise RuntimeError("Alignment failed.")
        self.acquire_image(mode=mode)
        self._save_buffer_image(token=f"_pick_{pick_id}_{mode}")
        stage_x, stage_y, stage_z = sem.ReportStageXYZ()
        shift_x, shift_y = sem.ReportSpecimenShift()

        return {"stage": (stage_x, stage_y), "stageZ": stage_z, "specimen_shift": (shift_x, shift_y)}

