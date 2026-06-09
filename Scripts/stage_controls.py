import pyserialem as pysem
import numpy as np
import math
import serialem as sem
import os

def _get_current_tilt() -> float:
    pysem.ReportTilt()
    tilt = float(pysem.ReportedValue())
    return tilt


def absolute_stage_movement(stage_position, tilt_angle):
    current_tilt = _get_current_tilt()
    print(f"[INFO] The current stage position is {pysem.ReportStageXYZ()} and the stage tilt is {current_tilt}.")
    if current_tilt != tilt_angle:
        sem.TiltTo(tilt_angle)
        pysem.Delay(1, 'sec')
    else:
        print('[INFO] Stage already tilted to the correct angle.')
    sem.MoveStageTo(stage_position[0], stage_position[1], stage_position[2])

def acquire_montage_for_state(imaging_state: str, fov_um_x: float, fov_um_y: float,
    tilt_angle: float = 0.0, path: str | None = None,
    file_prefix: str = "montage"):
    
    if imaging_state in ['Atlas', 'GridSquare', 'Lamella', 'HighMagOverview']:
        sem.GoToImagingState(imaging_state)
        sem.Delay(1)
    else:
        raise ValueError(f"imaging_state is not in the list of pre-defined imaging states.")
    
    sem.TiltTo(tilt_angle)
    sem.Record()
    sem.Delay()
    image_x_px, image_y_px = sem.ReportImageSize("A") #Image data for image in Buffer A (should be the last acquired one.)
    pixel_size_nm = sem.ReportPixelSize("A")
    
    tile_fov_um_x = image_x_px * pixel_size_nm / 1000
    tile_fov_um_y = image_y_px * pixel_size_nm / 1000

    overlap_fraction = 0.1

    step_um_x = tile_fov_um_x * (1.0 - overlap_fraction)
    step_um_y = tile_fov_um_y * (1.0 - overlap_fraction)

    nx = max(1, math.ceil((fov_um_x - tile_fov_um_x) / step_um_x) + 1)
    ny = max(1, math.ceil((fov_um_y - tile_fov_um_y) / step_um_y) + 1)

    filepath = os.path.join(path, file_prefix)
    sem.OpenNewMontage(nx, ny, filepath)
    #sem.SetMontageParams() 
    sem.Montage()
    sem.NewMap(file_prefix)


def run_experiment_setup(path):
    navigator_items = []
    input("[ToDO] Please load the experiment file and the settings file. ENTER")
    # Add with OK confirmation
    input('[ToDO] Switch to Low-Dose mode. ENTER')

    absolute_stage_movement([0.0e-6, 0.0e-6, 0.0e-6], tilt_anlge=0.0)

    sem.GoToImagingState("Atlas")
    pysem.Delay(1, 'sec')
    sem.View()

    input('[ToDO] Move feature of interest to the center of the stage for rough eucentric alignment. ENTER')

    sem.Eucentricity(1)

    acquire_montage_for_state(fov_um_x=3000, fov_um_y=3000, tilt_angle=0.0, path=path, file_prefix='LMM')

    # run rough eucentric at 0 degree
    # acquire atlas at 0 degree
    # Prompt to do transformation to stage coordinates
    # Run acquisition at milling angle or 0 degree
    # Do Correlation outside of this tool and write the txt file
    # Add txt file coordinates back to navigator file
def run_clem_alignment(position, type):

    if type not in ['lamella', 'airyscan']:
        raise ValueError(f"Type {type} is not predefined. Known types are lamella and airyscan.")
   
    # I think I should readin the 

    #I would like to add the ability to move to different positions, take a low mag mode, move the lamella to the center, set eucentricity and then do a montage
    #Then I should import the image into the viewer, do the correlation and return the points to the navigator



PATH = ''
run_experiment_setup()



    
