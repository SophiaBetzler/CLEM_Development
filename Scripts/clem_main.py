
import os
from clem_mrc_mdoc_reader import MRCReader
from clem_tem_communication import TEMComm
from clem_target_picking import CLEMPicker
from clem_general import ExecutiveControls
from pathlib import Path


PATH = r"C:\Users\CZII\Documents\Data\s26jul01b"
SAMPLE_TYPE = 'airyscan'
OFFLINE = True
if SAMPLE_TYPE == 'airyscan':
    MILLING_ANGLE = 0.0
else:
    MILLING_ANGLE = 15.0


if __name__ == "__main__":
    mrc = MRCReader(coord_key="AlignedPieceCoordsVS", path=PATH, refine_alignment=False, section=0)

    temcom = TEMComm(rotation="rotation", mrc_reader=mrc, path=PATH, offline=OFFLINE)

    exc = ExecutiveControls(tem_communication=temcom, mrc_reader=mrc, sample_type=SAMPLE_TYPE, milling_angle=MILLING_ANGLE)

    #exc.run_experiment_setup()
    print(["[INFO] Finished experiment setup."])
    #exc.run_acquire_position_montages()
    print(["[INFO] Finished acquiring position montages."])
    exc.run_clem_alignment()
    if SAMPLE_TYPE == 'airyscan':
        exc.run_high_mag_montages()
        exec.run_clem_alignment()

    #picker = CLEMPicker(montage=mrc.run_montage_loader_and_create_summary(), navigator=temcom)
    

    #
    #navc.run_picks_visualization(mrc_file=FILENAME)
    


