import os
from clem_mrc_mdoc_reader import MRCReader
#from clem_navigator_communication import TEMComm
from clem_target_picking import CLEMPicker


PATH = "/Users/sophia.betzler/Desktop/Test_Data"
FILENAME = "lamella-1-mmm-test-5x5-6500.mrc"

if __name__ == "__main__":
    mrc = MRCReader(
        path=os.path.join(PATH, FILENAME),
        coord_key="AlignedPieceCoordsVS", refine_alignment=False, section=0)
    
    temcom = TEMComm(path=PATH, rotation="rotation")

    picker = CLEMPicker(montage=mrc.run_montage_loader_and_create_summary(), navigator=temcom)
    picks = picker.picker()
    picker.run_auto_picker(picks)

    #
    #navc.run_picks_visualization(mrc_file=FILENAME)
    
