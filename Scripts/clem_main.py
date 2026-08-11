
import os
from clem_mrc_mdoc_reader import MRCReader
from clem_tem_communication import TEMComm
from clem_target_picking import CLEMPicker
from clem_general import ExecutiveControls
from pathlib import Path
import traceback
import scipy
from clem_dataclasses import AllSitesDataCollection, SiteDataSummary


PATH = r"C:\\Users\\CZII\\Documents\\Data\\s26jul23a"
SAMPLE_TYPE = 'airyscan'
OFFLINE = True
if SAMPLE_TYPE == 'airyscan':
    MILLING_ANGLE = 0.0
else:
    MILLING_ANGLE = -15.0


if __name__ == "__main__":

    site_collection = AllSitesDataCollection()

    mrc = MRCReader(coord_key="AlignedPieceCoordsVS", section=0)

    temcom = TEMComm(mrc_reader=mrc, path=PATH, offline=OFFLINE)


    exc = ExecutiveControls(tem_communication=temcom, mrc_reader=mrc, sample_type=SAMPLE_TYPE, milling_angle=MILLING_ANGLE, site_collection=site_collection)

    #exc.run_experiment_setup()
    print(["[INFO] Finished experiment setup."])
    exc.run_acquire_position_montages()
    print(["[INFO] Finished acquiring position montages."])
    try:
        exc.run_clem_alignment()
    except Exception:
        traceback.print_exc()
        input("Crashed - press ENTER to close")

