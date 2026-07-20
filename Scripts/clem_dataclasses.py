from __future__ import annotations   
import pickle
from datetime import datetime
from typing import Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import os


class AllSitesDataCollection:
    def __init__(self):
        self.sites = {}
        self.active_site_id = None

    def add_site(self, site_data):
        if site_data.site_id is None:
            raise ValueError("site_data.site_id is required")
        self.sites[site_data.site_id] = site_data
        self.active_site_id = site_data.site_id
        return site_data

    def get_site(self, site_id):
        return self.sites.get(site_id)

    def set_active_site(self, site_id):
        self.active_site_id = site_id
        return self.get_site(site_id)

    def remove_site(self, site_id):
        self.sites.pop(site_id, None)
        if self.active_site_id == site_id:
            self.active_site_id = next(iter(self.sites), None)

    @property
    def active_site(self):
        return self.sites.get(self.active_site_id)


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
        
    def get_image_shift(self) -> tuple[float, float]:
        return (
            float(self.image_shift_x or 0.0),
            float(self.image_shift_y or 0.0),
        )


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
    pixel_spacing_um: Optional[float] = None
    czi_overview: Optional[Any] = None
    czi_path: Optional[str] = None
    czi_pixel_spacing_um: Optional[float] = None
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
        pixel_spacing_um = mrc_reader.read_tiff_pixel_spacing_um(tiff_filepath)

        prev = self.tiff.pixel_spacing_um if self.tiff is not None else None
        if (prev is not None and pixel_spacing_um is not None) and abs(prev - pixel_spacing_um) > 1e-9:
            print(f"[INFO] Overwritting TIFF pixel spacing for {self.site_id}: {prev:.6f} -> {pixel_spacing_um: .6f} um/px")

        self.tiff = TiffSummary(
            ome_path=os.fspath(tiff_filepath),
            stack_czyx=stack,
            num_channels=int(c),
            num_z_slices=int(z),
            stack_height=int(y),
            stack_width=int(x),
            pixel_spacing_um=pixel_spacing_um,
            info=info,
        )
        return self
    
    def populate_czi(self, mrc_reader, czi_filepath):
        stack, info = mrc_reader.load_czi(czi_filepath)
        pixel_spacing_um = mrc_reader.read_czi_pixel_spacing_um(czi_filepath)

        if self.tiff is None:
            self.tiff = TiffSummary()

        prev = self.tiff.czi_pixel_spacing_um

        if (prev is not None and pixel_spacing_um is not None
                and abs(prev - pixel_spacing_um) > 1e-9):
            print(f"[INFO] Overwriting CZI pixel spacing for {self.site_id}: "
                  f"{prev:.6f} -> {pixel_spacing_um:.6f} um/px")

        self.tiff.czi_overview = stack
        self.tiff.czi_path = os.fspath(czi_filepath)
        self.tiff.czi_pixel_spacing_um = pixel_spacing_um
        if not self.tiff.info:
            self.tiff.info = info
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
    def image(self):
        if self.mrc is not None and getattr(self.mrc, "image", None) is not None:
            return self.mrc.image
        return None

    @image.setter
    def image(self, value):
        if self.mrc is None:
            self.mrc = MRCSummary(image=value)
        else:
            self.mrc.image = value

    @property
    def mrc_full(self):
        return self.image

    @mrc_full.setter
    def mrc_full(self, value):
        self.image = value

    @property
    def montage_image(self):
        return self.image

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
    

# --------------------------------------------------------------------------- #
# Portable transform record (serialization-friendly)
# --------------------------------------------------------------------------- #
# Small parsing helpers so a record can be rebuilt from string cells (CSV) or
# already-typed values (YAML) without the caller caring which.

def _to_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() in ("none", "null", "na", "n/a"):
            return None
        return float(s)
    return float(value)


def _to_int(value):
    f = _to_float(value)
    return int(round(f)) if f is not None else None


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _to_shape(value):
    """Accept a list/tuple, a ';'/','-joined string, or None -> tuple[int] | None."""
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(round(float(v))) for v in value)
    parts = [p for p in str(value).replace(",", ";").split(";") if p.strip() != ""]
    return tuple(int(round(float(p))) for p in parts) if parts else None


@dataclass
class TransformRecord:

    matrix: Optional[Any] = None            # 3x3 homogeneous, row-major: list[list[float]]
    transform_type: Optional[str] = None
    flip_x: bool = False
    flip_y: bool = True                      # Y-flip is the permanent default
    scale_x: Optional[float] = None
    scale_y: Optional[float] = None
    rotation_deg: Optional[float] = None
    rmse_px: Optional[float] = None
    fixed_scale: Optional[float] = None      # scale bound used at fit time (if any)
    scale_tolerance: Optional[float] = None  # +/- fraction around fixed_scale
    n_pairs: Optional[int] = None
    mrc_shape: Optional[tuple] = None        # (H, W)
    tiff_shape: Optional[tuple] = None       # (C, Z, Y, X)
    pixel_spacing_um: Optional[float] = None
    created_at: Optional[str] = None
    source_path: Optional[str] = None        # file it was written to / read from

    @property
    def mean_scale(self) -> Optional[float]:
        """Best single scale estimate, preferring an explicit fixed_scale.
        Used when re-applying to a different image with a scale-limited fit."""
        if self.fixed_scale is not None:
            return float(self.fixed_scale)
        vals = [v for v in (self.scale_x, self.scale_y) if v is not None]
        return float(sum(vals) / len(vals)) if vals else None

    def to_dict(self) -> dict:
        """Plain-Python dict (JSON/YAML-safe); matrix as nested float lists."""
        return {
            "created_at": self.created_at,
            "transform_type": self.transform_type,
            "flip_x": bool(self.flip_x),
            "flip_y": bool(self.flip_y),
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "rotation_deg": self.rotation_deg,
            "rmse_px": self.rmse_px,
            "fixed_scale": self.fixed_scale,
            "scale_tolerance": self.scale_tolerance,
            "n_pairs": self.n_pairs,
            "mrc_shape": list(self.mrc_shape) if self.mrc_shape is not None else None,
            "tiff_shape": list(self.tiff_shape) if self.tiff_shape is not None else None,
            "pixel_spacing_um": self.pixel_spacing_um,
            "matrix": ([[float(v) for v in row] for row in self.matrix]
                       if self.matrix is not None else None),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TransformRecord":
        """Rebuild from a dict whose values may be typed (YAML) or strings (CSV)."""
        matrix = d.get("matrix")
        if matrix is not None:
            matrix = [[float(v) for v in row] for row in matrix]
        return cls(
            matrix=matrix,
            transform_type=(d.get("transform_type") or None),
            flip_x=_to_bool(d.get("flip_x", False)),
            flip_y=_to_bool(d.get("flip_y", True)),
            scale_x=_to_float(d.get("scale_x")),
            scale_y=_to_float(d.get("scale_y")),
            rotation_deg=_to_float(d.get("rotation_deg")),
            rmse_px=_to_float(d.get("rmse_px")),
            fixed_scale=_to_float(d.get("fixed_scale")),
            scale_tolerance=_to_float(d.get("scale_tolerance")),
            n_pairs=_to_int(d.get("n_pairs")),
            mrc_shape=_to_shape(d.get("mrc_shape")),
            tiff_shape=_to_shape(d.get("tiff_shape")),
            pixel_spacing_um=_to_float(d.get("pixel_spacing_um")),
            created_at=(d.get("created_at") or None),
            source_path=(d.get("source_path") or None),
        )

    @classmethod
    def from_registration(cls, reg: "RegistrationSummary", matrix,
                          pixel_spacing_um=None, created_at=None,
                          n_pairs=None) -> "TransformRecord":
        """Build a portable record from an existing RegistrationSummary.

        ``matrix`` must be supplied as a nested float list (the correlator
        extracts it from the fitted transform) so this module stays free of
        numpy/skimage dependencies.
        """
        fit = reg.fit_info or {}
        return cls(
            matrix=[[float(v) for v in row] for row in matrix] if matrix is not None else None,
            transform_type=reg.transform_type,
            flip_x=bool(reg.flip_x),
            flip_y=bool(reg.flip_y),
            scale_x=fit.get("scale_x"),
            scale_y=fit.get("scale_y"),
            rotation_deg=fit.get("rotation_deg", reg.rotation_deg),
            rmse_px=fit.get("rmse_px"),
            fixed_scale=fit.get("expected_scale"),
            scale_tolerance=fit.get("scale_tolerance"),
            n_pairs=n_pairs if n_pairs is not None else reg.num_point_pairs,
            pixel_spacing_um=pixel_spacing_um,
            created_at=created_at,
        )
