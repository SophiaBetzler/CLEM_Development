"""Session progress tracking for the CLEM workflow.

Holds a small JSON file at <output_root>/clem_session.json recording which
stages and which sites are finished, so a crashed or deliberately interrupted
run can be resumed instead of restarted.

Deliberately JSON and deliberately separate from the SiteDataSummary pickles:

    pickles hold DATA          (montages, stacks, registrations)
    this file holds PROGRESS   (what is done, what is not)

You can open it in a text editor, see where a run stopped, and force a redo by
changing one value to "pending".
"""

import json
import os
from datetime import datetime


# Status values used for both stages and per-site steps.
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"


class SessionState:

    FILENAME = "clem_session.json"
    VERSION = 1

    # Per-site steps, matching the workflow stages that act on a single site.
    SITE_STEPS = ("setup", "montage", "alignment")

    def __init__(self, output_root, sample_type=None, milling_angle=None,
                 offline=False, data=None):
        self.output_root = os.fspath(output_root)
        data = dict(data or {})

        self.version = data.get("version", self.VERSION)
        self.created_at = data.get("created_at") or self._now()
        self.updated_at = data.get("updated_at")
        self.sample_type = data.get("sample_type", sample_type)
        self.milling_angle = data.get("milling_angle", milling_angle)
        # True if ANY run that wrote this file was offline. Offline runs record
        # dummy stage coordinates (report_stage_position returns zeros), so a
        # file touched by one must not be trusted to drive a live session.
        self.offline_run = bool(data.get("offline_run", False))
        self.nav_file = data.get("nav_file")
        self.stages = dict(data.get("stages") or {})
        self.sites = {k: dict(v) for k, v in (data.get("sites") or {}).items()}

        # Values for the CURRENT process, kept apart from what was loaded so
        # check_compatible() can compare the two.
        self.current_sample_type = sample_type if sample_type is not None else self.sample_type
        self.current_milling_angle = milling_angle if milling_angle is not None else self.milling_angle
        self.current_offline = bool(offline)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    @staticmethod
    def _now():
        return datetime.now().isoformat(timespec="seconds")

    @property
    def path(self):
        return os.path.join(self.output_root, self.FILENAME)

    @classmethod
    def load(cls, output_root, sample_type=None, milling_angle=None, offline=False):
        """Load the state file if present, else return a fresh empty state."""
        path = os.path.join(os.fspath(output_root), cls.FILENAME)
        data = None
        if os.path.isfile(path):
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except (OSError, ValueError) as e:
                # A corrupt state file must never block a run: warn, move it
                # aside and start clean.
                backup = path + ".corrupt-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                print(f"[WARN] Could not read {path} ({e}); moving it to {backup}.")
                try:
                    os.replace(path, backup)
                except OSError:
                    pass
                data = None
        return cls(output_root, sample_type=sample_type,
                   milling_angle=milling_angle, offline=offline, data=data)

    def save(self):
        """Write atomically: a crash mid-write cannot corrupt the state."""
        self.updated_at = self._now()
        payload = {
            "version": self.VERSION,
            "path": self.output_root,
            "sample_type": self.current_sample_type,
            "milling_angle": self.current_milling_angle,
            "offline_run": bool(self.offline_run or self.current_offline),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "nav_file": self.nav_file,
            "stages": self.stages,
            "sites": self.sites,
        }
        os.makedirs(self.output_root, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        return self.path

    def exists(self):
        return os.path.isfile(self.path)

    # ------------------------------------------------------------------ #
    # Stage progress
    # ------------------------------------------------------------------ #

    def stage_status(self, stage):
        return self.stages.get(stage, PENDING)

    def mark_stage(self, stage, status):
        self.stages[stage] = status
        self.save()

    def is_stage_done(self, stage):
        return self.stage_status(stage) == DONE

    def last_completed_stage(self, stages):
        done = [s for s in stages if self.is_stage_done(s)]
        return done[-1] if done else None

    def next_stage(self, stages):
        """First stage not yet done, or None if the whole run is complete."""
        for stage in stages:
            if not self.is_stage_done(stage):
                return stage
        return None

    # ------------------------------------------------------------------ #
    # Per-site progress
    # ------------------------------------------------------------------ #

    def site(self, site_id):
        return self.sites.setdefault(site_id, {})

    def site_status(self, site_id, step):
        return self.sites.get(site_id, {}).get(step, PENDING)

    def is_site_done(self, site_id, step):
        return self.site_status(site_id, step) == DONE

    def mark_site(self, site_id, step, status, pickle_path=None):
        entry = self.site(site_id)
        entry[step] = status
        if pickle_path is not None:
            # Store relative to output_root so the folder stays portable.
            try:
                entry["pickle"] = os.path.relpath(pickle_path, self.output_root)
            except ValueError:
                entry["pickle"] = os.fspath(pickle_path)
        self.save()

    def site_pickle(self, site_id):
        """Absolute path to the last checkpoint written for a site, or None."""
        rel = self.sites.get(site_id, {}).get("pickle")
        if not rel:
            return None
        path = rel if os.path.isabs(rel) else os.path.join(self.output_root, rel)
        return path if os.path.isfile(path) else None

    def known_sites(self):
        return sorted(self.sites)

    # ------------------------------------------------------------------ #
    # Skipping sites
    # ------------------------------------------------------------------ #

    def mark_site_skipped(self, site_id, skipped=True, reason=None):
        """Exclude a site from montages and every later stage, persistently.

        Recorded on the site rather than on one step, so the decision carries
        across stages and across restarts until it is cleared.
        """
        entry = self.site(site_id)
        if skipped:
            entry["skipped"] = True
            if reason:
                entry["skip_reason"] = reason
        else:
            entry.pop("skipped", None)
            entry.pop("skip_reason", None)
        self.save()

    def is_site_skipped(self, site_id):
        return bool(self.sites.get(site_id, {}).get("skipped"))

    def skipped_sites(self):
        return [s for s in sorted(self.sites) if self.is_site_skipped(s)]

    # ------------------------------------------------------------------ #
    # Resume decisions
    # ------------------------------------------------------------------ #

    def has_progress(self):
        return any(v == DONE for v in self.stages.values()) or bool(self.sites)

    def reset(self):
        """Forget all progress, keeping the file's identity fields."""
        self.stages = {}
        self.sites = {}
        self.nav_file = None
        self.offline_run = self.current_offline
        self.save()

    def check_compatible(self):
        """Warnings about resuming this session with the current settings.

        Returns a list of human-readable strings; empty means consistent.
        """
        problems = []
        if self.sample_type is not None and self.current_sample_type != self.sample_type:
            problems.append(
                f"sample_type differs: session was '{self.sample_type}', "
                f"this run is '{self.current_sample_type}'")
        if (self.milling_angle is not None
                and self.current_milling_angle is not None
                and abs(float(self.milling_angle) - float(self.current_milling_angle)) > 1e-9):
            problems.append(
                f"milling_angle differs: session was {self.milling_angle}, "
                f"this run is {self.current_milling_angle}")
        if self.offline_run and not self.current_offline:
            problems.append(
                "this session was written by an OFFLINE run, so its stage "
                "coordinates are placeholders (0, 0, 0, 0) -- do not resume it "
                "on the microscope")
        return problems

    def summary(self, stages):
        """One-line-per-stage progress report for the operator."""
        lines = [f"Session: {self.path}",
                 f"  started {self.created_at}, last updated {self.updated_at}"]
        if self.offline_run:
            lines.append("  NOTE: touched by an offline run (placeholder coordinates)")
        for stage in stages:
            lines.append(f"  {stage:20s} {self.stage_status(stage)}")
        if self.sites:
            lines.append(f"  sites: {len(self.sites)}")
            for site_id in self.known_sites():
                steps = self.sites[site_id]
                if self.is_site_skipped(site_id):
                    reason = steps.get("skip_reason")
                    lines.append(f"    {site_id:16s} SKIPPED"
                                 + (f"  ({reason})" if reason else ""))
                    continue
                shown = " ".join(f"{s}={steps.get(s, PENDING)}" for s in self.SITE_STEPS)
                lines.append(f"    {site_id:16s} {shown}")
        return "\n".join(lines)
