import os

from clem_mrc_mdoc_reader import MRCReader
from clem_navigator_communication import NavigatorComm
from clem_target_picking import CLEMPicker


PATH = "C:\\Users\\CZII\\Desktop\\Test_Data\\"
FILENAME = "12-chief-dog_montage_20260616-07-47-11.mrc"

if __name__ == "__main__":
    mrc = MRCReader(
        path=os.path.join(PATH, FILENAME),
        coord_key="AlignedPieceCoordsVS", refine_alignment=False, section=0)

    
    picks = CLEMPicker(montage=mrc.run_montage_loader_and_create_summary()).picker()

    navc = NavigatorComm(path=PATH, picks=picks, rotation="rotation")
    navc.show_nav_adjustment()
    navc.run_picks_visualization(mrc_file=FILENAME)
