import os

from clem_mrc_mdoc_reader import MRCReader
from clem_navigator_communication import NavigatorComm


PATH = "C:\\Users\\CZII\\Desktop\\Test_Data\\"
FILENAME = "Airy-06_montage_20260612-16-34-09.mrc"

if __name__ == "__main__":
    mrc = MRCReader(
        path=os.path.join(PATH, FILENAME),
        coord_key="AlignedPieceCoordsVS", refine_alignment=False, section=0)

    mrc.run_mrc_reader_and_picker()
    navc = NavigatorComm(path=PATH, picks=mrc.picks, rotation="rotation")
    navc.run_picks_visualization(mrc_file=FILENAME)
