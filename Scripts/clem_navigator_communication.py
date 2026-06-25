import os
import sys
sys.path.append(r"C:\Program Files\SerialEM\PythonModules")
print(sys.executable)
print(sys.version)
import serialem as sem
from datetime import datetime

class NavigatorComm:
    def __init__(self, path, picks=None, rotation=True):
        self.path = path
        self.picks = picks
        self.rotation = rotation

    def _create_nav_file(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")

        if sem.ReportIfNavOpen() > 0:
            sem.CloseNavigator()
            
        nav_file = os.path.join(self.path, "nav_file" + "_" + timestamp + '.nav')
        sem.OpenNavigator(nav_file)

    def load_mrc_in_nav(self, mrc_file):
        if sem.ReportIfNavOpen() == 0:
            self._create_nav_file()
        print( os.path.join(self.path, mrc_file))
        sem.CloseFile()
        sem.OpenOldFile(os.path.join(self.path, mrc_file))
        sem.ReadFile(0)
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
    
    def add_stage_pos_to_nav(self):
        if sem.ReportIfNavOpen() == 0:
            self._create_nav_file()
        _, _, default_z = sem.ReportStageXYZ()
        pt_id = int(sem.GetUniqueNavID())
        for p in self.picks:
            sx, sy = p["rot"]
            z = p["stage_z"] if p["stage_z"] is not None else default_z
            sem.AddStagePosAsNavPoint(sx, sy, z, pt_id)

    def run_picks_visualization(self, mrc_file):
        self.load_mrc_in_nav(mrc_file=mrc_file)
        self.add_stage_pos_to_nav()