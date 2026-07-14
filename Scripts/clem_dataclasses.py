from __future__ import annotations   
import pickle
from datetime import datetime
from typing import Any, Optional
from dataclasses import dataclass, field
import os

@dataclass
class Tile:
    """One montage tile. Pixel coords are montage-pixel positions; stage coords
    are physical stage positions."""
    z_index: Optional[int] = None
    stage_z_um: Optional[float] = None
    pixel_x_um: Optional[float] = None
    pixel_y_um: Optional[float] = None
    stage_x_um: Optional[float] = None
    stage_y_um: Optional[float] = None


@dataclass
class Pick:
    pick_id: str
    pixel_x_um: Optional[float] = None
    pixel_y_um: Optional[float] = None
    stage_x_um: Optional[float] = None
    stage_y_um: Optional[float] = None
    stage_z_um: Optional[float] = None
    notes: Optional[str] = None

    refined_stage_x: Optional[float] = None
    refined_stage_y: Optional[float] = None
    refined_stage_z: Optional[float] = None

    record_img_path: Optional[str] = None
    view_crop_path: Optional[str] = None

    image_shift_x: Optional[float] = None
    image_shift_y: Optional[float] = None

    is_tracking_target: bool = False
    refinement_quality: Optional[str] = None

    def has_refinement(self) -> bool:
        return self.refined_stage_x is not None

    def get_stage_position(self) -> tuple:
        if self.refined_stage_x is not None:
            return (self.refined_stage_x, self.refined_stage_y, self.refined_stage_z)
        else:
            return (self.stage_x_um, self.stage_y_um, self.stage_z_um)


@dataclass
class TargetGroup:
    group_id: str
    tracking: Pick
    picks: list = field(default_factory=list)   # list[Pick]
 

# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

@dataclass
class MRCSummary:
    mrc_path: Optional[str] = None
    image: Optional[Any] = None                 # np.ndarray (assembled montage)
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    pixel_spacing_um: Optional[float] = None
    feather_pixels: Optional[int] = None
    section: Optional[int] = None
    alignment: Optional[str] = None
    coord_field: Optional[str] = None
    rotation_deg: Optional[float] = None
    min_x_pixels: Optional[float] = None
    min_y_pixels: Optional[float] = None
    tiles: list[Tile] = field(default_factory=list)
    stage_matrix: Optional[Any] = None          # np 3x3, or None
    stage_fit: Optional[dict] = None            # diagnostics leaf
    flip_x: Optional[bool] = None
    flip_y: Optional[bool] = None


@dataclass
class TiffSummary:
    ome_path: Optional[str] = None
    stack_czyx: Optional[Any] = None            # np.ndarray (C, Z, Y, X)
    num_channels: Optional[int] = None
    num_z_slices: Optional[int] = None
    stack_height: Optional[int] = None
    stack_width: Optional[int] = None
    info: str = ""


@dataclass
class RegistrationSummary:
    transform_type: Optional[str] = None
    transform_matrix: Optional[Any] = None
    fit_info: Optional[dict] = None             # diagnostics leaf
    num_point_pairs: Optional[int] = None
    flip_x: bool = False
    flip_y: bool = False
    warped_channels: Optional[list] = None       # list[np.ndarray]
    rotation_deg: Optional[float] = 0.0


@dataclass
class Acquisition:
    stage_x_um: Optional[float] = None
    stage_y_um: Optional[float] = None
    stage_z_um: Optional[float] = None
    stage_tilt: Optional[float] = None


# --------------------------------------------------------------------------- #
# The container (root of the object graph)
# --------------------------------------------------------------------------- #

@dataclass
class SiteDataSummary:
    site_id: str
    path: str                                   # the site folder
    mrc: Optional[MRCSummary] = None
    tiff: Optional[TiffSummary] = None
    registration: Optional[RegistrationSummary] = None
    acquisition: Optional[Acquisition] = None
    picks: list[Pick] = field(default_factory=list)
    groups: list[TargetGroup] = field(default_factory=list)

    # -------- MRC: reader builds the MRCSummary; we just assign it ---------- #
    def populate_mrc(self, mrc_reader, mrc_filepath):
        """No transcription: build_montage_summary already returns an
        MRCSummary, so this is a direct hand-off."""
        self.mrc = mrc_reader.build_montage_summary(mrc_filepath)
        return self

    # -------- TIFF: build the TiffSummary from the low-level loader --------- #
    def populate_tiff(self, mrc_reader, tiff_filepath):
        stack, info = mrc_reader.load_ome_tiff(tiff_filepath)
        c, z, y, x = stack.shape
        self.tiff = TiffSummary(
            ome_path=os.fspath(tiff_filepath),
            stack_czyx=stack,
            num_channels=int(c),
            num_z_slices=int(z),
            stack_height=int(y),
            stack_width=int(x),
            info=info,
        )
        return self

    # -------- Registration: from the correlator result dict ---------------- #
    def set_registration(self, correlator_result, transform_type, flip_x=False, flip_y=False):
        r = correlator_result
        fit_info = r.get("fit_info") or {}

        self.registration = RegistrationSummary(
            transform_type=transform_type,
            transform_matrix=r.get("transform"),
            fit_info=fit_info,
            num_point_pairs=r.get("n_pairs"),
            flip_x=bool(flip_x),
            flip_y=bool(flip_y),
            warped_channels=r.get("warped_channels"),
            rotation_deg=fit_info.get("rotation_deg", 0.0),
        )
        return self

    # -------- Acquisition / picks ------------------------------------------ #
    def set_acquisition_from_csv_row(self, row):
        self.acquisition = Acquisition(
            stage_x_um=row.get("stage_x_um"),
            stage_y_um=row.get("stage_y_um"),
            stage_z_um=row.get("stage_z_um"),
            stage_tilt=row.get("stage_tilt"),
        )
        return self

    def add_pick(self, pick_id, pixel_x_um=None, pixel_y_um=None,
                 stage_x_um=None, stage_y_um=None, stage_z_um=None, notes=None):
        self.picks.append(Pick(
            pick_id=pick_id,
            pixel_x_um=pixel_x_um, pixel_y_um=pixel_y_um,
            stage_x_um=stage_x_um, stage_y_um=stage_y_um,
            stage_z_um=stage_z_um, notes=notes,
        ))
        return self

    # -------- Convenience accessors ---------------------------------------- #
    @property
    def pixel_spacing_um(self):
        return self.mrc.pixel_spacing_um if self.mrc else None

    @property
    def montage_image(self):
        return self.mrc.image if self.mrc else None

    @property
    def warped_channels(self):
        return self.registration.warped_channels if self.registration else None

    # ======================================================================= #
    # Persistence
    # -----------------------------------------------------------------------
    # pickle walks the whole object graph from `self`, so a single save() writes
    # the SiteDataSummary AND its MRCSummary/TiffSummary/RegistrationSummary/
    # Acquisition, every Tile and Pick, and all numpy arrays, into ONE file.
    # load() rebuilds the entire tree; the returned object is ready to use with
    # no re-linking needed.
    # ======================================================================= #
    def make_pickle_path(self):
        """<site folder>/<site_id>_<YYYYmmdd-HH-MM-SS>.pkl -- new name per save,
        so nothing is overwritten."""
        timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
        return os.path.join(self.path, f"{self.site_id}_{timestamp}.pkl")

    def save(self, pickle_path=None):
        """Store the ENTIRE object (all sections, tiles, picks, arrays) to one
        pickle file. Returns the path written."""
        pickle_path = pickle_path or self.make_pickle_path()
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] SiteDataSummary saved: {pickle_path}")
        return pickle_path

    @classmethod
    def load(cls, pickle_path):
        """Read a pickle written by save() and return the reconstructed
        SiteDataSummary, with its full sub-tree intact."""
        with open(pickle_path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{pickle_path} did not contain a {cls.__name__}.")
        return obj

    @classmethod
    def load_latest(cls, site_folder, site_id):
        """Find the newest <site_id>_*.pkl in a folder and load it."""
        matches = sorted(Path(site_folder).glob(f"{site_id}_*.pkl"),
                         key=lambda p: p.stat().st_mtime)
        if not matches:
            raise FileNotFoundError(f"No {site_id}_*.pkl found in {site_folder}")
        return cls.load(matches[-1])

    def __repr__(self):
        stages = [n for n in ("mrc", "tiff", "registration", "acquisition")
                  if getattr(self, n) is not None]
        n_tiles = len(self.mrc.tiles) if self.mrc else 0
        return (f"SiteDataSummary(site_id={self.site_id!r}, loaded={stages}, "
                f"n_tiles={n_tiles}, n_picks={len(self.picks)})")
