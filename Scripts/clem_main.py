import os
from clem_mrc_mdoc_reader import MRCReader
from clem_tem_communication import TEMComm
from clem_target_picking import CLEMPicker
from clem_general import ExecutiveControls
from pathlib import Path


PATH = "/Users/sophia.betzler/Desktop/Test_Data"
SAMPLE_TYPE = 'airyscan'
FILENAME = "lamella-1-mmm-test-5x5-6500.mrc"

if __name__ == "__main__":
    mrc = MRCReader(coord_key="AlignedPieceCoordsVS", refine_alignment=True, section=0)

    temcom = TEMComm(rotation="rotation", path=PATH)

    exc = ExecutiveControls(tem_communication=temcom, mrc_reader=mrc, path=PATH, sample_type=SAMPLE_TYPE)

    exc.run_experiment_setup()
    exc.run_acquire_position_montages()
    exc.run_clem_alignment()

    picker = CLEMPicker(montage=mrc.run_montage_loader_and_create_summary(), navigator=temcom)
    

    #
    #navc.run_picks_visualization(mrc_file=FILENAME)
    
