import os
import sys
sys.path.append(r"C:\Program Files\SerialEM\PythonModules")
print(sys.executable)
print(sys.version)
import serialem as sem
from datetime import datetime
import numpy as np

class TEMComm:
    def __init__(self, path, rotation=True):
        self.path = path
        self.rotation = rotation

    def _create_nav_file(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")

        if sem.ReportIfNavOpen() > 0:
            sem.CloseNavigator()
            
        nav_file = os.path.join(self.path, "nav_file" + "_" + timestamp + '.nav')
        sem.OpenNavigator(nav_file)

    def load_mrc_in_nav(self, mrc_file, buf="0"):
        if sem.ReportIfNavOpen() == 0:
            self._create_nav_file()
        print( os.path.join(self.path, mrc_file))
        sem.CloseFile()
        sem.ReadOtherFile(0, buf, os.path.join(self.path, mrc_file))
        sem.NewMap()

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

    def run_picks_visualization(self, mrc_file):
        self.load_mrc_in_nav(mrc_file=mrc_file)
        self.add_stage_pos_to_nav()

    def precise_stage_move(self, stage_position):
        backlashX = 2.0
        backlashY = 2.0
        if len(stage_position) == 3:
            sem.MoveStageTo(stage_position[0], stage_position[1], stage_position[2], backlashX, backlashY)
        else:
            sem.MoveStageTo(stage_position[0], stage_position[1], backlashX, backlashY)
        sem.Delay(2, 'sec')

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
        
    def get_calibration_matrices(self, key):
        read_out_dict = {"ss2s": np.array(sem.SpecimenToStageMatrix(0)).reshape((2, 2))}
        return read_out_dict

    def convert_stage_img_pos(self, buf, direction, coords):
        if direction == 'to_stage':
            sem.BufImagePosToStagePos(buf, 1, coords[0], coords[1])[:2]
        elif direction == 'to_img':
            sem.StagePosToBufImagePos(buf, 1, coords[0], coords[1])