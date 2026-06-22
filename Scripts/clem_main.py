#from Scripts.clem_mrc_mdoc_reader.py import MRCReader
from clem_navigator_communication import NavigatorComm


if __name__ == "__main__":
    mrc = MRCReader(
        path='/Users/sophia.betzler/Desktop/12-chief-dog_montage_20260616-07-47-11.mrc',
        coord_key="PieceCoordinates", refine_alignment=False, section=0)
    navc = NavigatorComm(path='Users/sophia.betzler/Desktop')
    navc.load_mrc_in_nav(mrc_file='12-chief-dog_montage_20260616-07-47-11.mrc')
    #tile_summary = mrc.create_montage()
    #picks = mrc.pick_stage_positions()
    #print(picks)