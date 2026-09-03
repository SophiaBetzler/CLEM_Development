import os
import csv
from pathlib import Path


from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import numpy as np
from clem_dataclasses import SiteDataSummary, AllSitesDataCollection
from clem_session import SessionState, PENDING, RUNNING, DONE, FAILED


class ExecutiveControls:

    NA_VALUES = {"", "na", "NA", "n/a", "N/A", "none", "None"}

    # Ordered pipeline. run() walks this list; each name maps to a method and,
    # for the per-site stages, to a step key in the session state.
    STAGES = ("experiment_setup", "site_setup", "site_montages", "clem_alignment")

    # Which per-site step in the state file each stage completes.
    STAGE_SITE_STEP = {"site_setup": "setup",
                       "site_montages": "montage",
                       "clem_alignment": "alignment"}

    def __init__(self, tem_communication, mrc_reader, sample_type, milling_angle, site_collection,
                 session_state=None):

        self.tem = tem_communication
        self.sample_type = sample_type
        self.milling_angle = milling_angle
        self.mrc_reader = mrc_reader
        self.site_collection = site_collection
        self.site_summaries = {}

        if self.site_collection is not None:
            self.tem.site_collection = self.site_collection
            self.mrc_reader.site_collection = self.site_collection

        # Progress tracking. Loaded from <output_root>/clem_session.json if it
        # exists, so a new process picks up where the last one stopped.
        self.state = session_state if session_state is not None else SessionState.load(
            self.tem.output_root,
            sample_type=sample_type,
            milling_angle=milling_angle,
            offline=getattr(self.tem, "offline", False))

        self.montage_settings = {'lamella': {"stage_tilt": -self.milling_angle, "fov_um_x": 20.0, "fov_um_y": 45.0},
                                    'airyscan': {"stage_tilt": 0.0, "fov_um_x": 105.0, "fov_um_y": 105.0, "fov_um_x_high_mag": 38.0, "fov_um_y_high_mag": 38.0},}

    # ---------------------------------------------------------------------------
    # Data loading (CSV file management)
    # ---------------------------------------------------------------------------

    def _csv_value_or_none(self, value):
        if value is None:
            return None
        value = str(value).strip()
        if value in self.NA_VALUES:
            return None
        return value


    def _csv_float_or_none(self, value):
        value = self._csv_value_or_none(value)
        if value is None:
            return None
        return float(value)

    def _import_csv_file(self, filename):
        csv_path = os.path.join(self.tem.output_root, filename)
        print(f"[INFO] Importing csv file {csv_path}.")

        with open(csv_path, newline="") as f:
            rows = (line for line in f if not line.lstrip().startswith("#"))
            reader = csv.DictReader(rows)

            sites = []
            for row in reader:
                site = dict(row)

                site["name"] = row.get("name") or row.get("label")
                site["stage_x_um"] = self._csv_float_or_none(row.get("stage_x_um"))
                site["stage_y_um"] = self._csv_float_or_none(row.get("stage_y_um"))
                site["stage_z_um"] = self._csv_float_or_none(row.get("stage_z_um"))

                if "stage_tilt" in row:
                    site["stage_tilt"] = self._csv_float_or_none(row.get("stage_tilt"))

                sites.append(site)

        return sites
    
    def _write_sites_csv(self, sites, filename):
        csv_path = os.path.join(self.tem.output_root, filename)

        fieldnames = list(sites[0].keys())

        if "stage_z_um" not in fieldnames:
            fieldnames.append("stage_z_um")

        if "stage_tilt" not in fieldnames:
            fieldnames.append("stage_tilt")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for site in sites:
                writer.writerow({
                    key: ("na" if site.get(key) is None else site.get(key)) for key in fieldnames})

    # ---------------------------------------------------------------------------
    # Restart support: rehydration, navigator, stage driver
    # ---------------------------------------------------------------------------

    def _site_folders(self):
        """Site folders on disk, sorted by name.

        Disk is the source of truth for which sites exist once setup has run;
        tem_stage_positions.csv only drives the initial site_setup stage. That
        way a restart still works if the CSV is edited or lost, and sites you
        removed from disk drop out on their own.
        """
        root = self.tem.output_root
        if not os.path.isdir(root):
            return []
        folders = []
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            if not os.path.isdir(full) or name.startswith(".") or name == "transforms":
                continue
            folders.append((name, full))
        return folders

    def load_site_collection(self, sites=None, verbose=True):
        """Rebuild site_collection from disk so a stage can run in a fresh
        process.

        Prefers the newest pickle for each site; falls back to a bare
        SiteDataSummary pointing at the folder, which is enough for the UI and
        the MRC reader to find the montage and TIFF themselves.
        """
        if self.site_collection is None:
            self.site_collection = AllSitesDataCollection()
            self.tem.site_collection = self.site_collection
            self.mrc_reader.site_collection = self.site_collection

        wanted = set(sites) if sites else None
        loaded, created = [], []

        for site_id, folder in self._site_folders():
            if wanted is not None and site_id not in wanted:
                continue
            if site_id in self.site_collection.sites:
                continue        # already in memory from an earlier stage

            site_data = None
            recorded = self.state.site_pickle(site_id)
            try:
                if recorded:
                    site_data = SiteDataSummary.load(recorded)
                else:
                    site_data = SiteDataSummary.load_latest(folder, site_id)
                loaded.append(site_id)
            except (FileNotFoundError, TypeError, EOFError, AttributeError) as e:
                if recorded:
                    print(f"[WARN] Could not load checkpoint for {site_id} ({e}); "
                          f"falling back to the folder.")
                site_data = SiteDataSummary(site_id=site_id, path=folder,
                                            milling_angle=self.milling_angle)
                created.append(site_id)

            # The folder may have moved (copied session, different drive
            # letter): trust the current location over the pickled one.
            site_data.path = folder
            self.register_site_data(site_data, active=False)

        if verbose:
            if loaded:
                print(f"[INFO] Restored {len(loaded)} site(s) from checkpoints: {', '.join(loaded)}")
            if created:
                print(f"[INFO] Built {len(created)} site(s) from folder contents: {', '.join(created)}")
            if not loaded and not created:
                print("[INFO] No existing site folders found.")
        return self.site_collection

    def _ensure_nav_open(self):
        """Open the session's navigator, reusing the recorded one if possible.

        Without this a restarted run creates a fresh nav file and every map
        registered by the previous run becomes unreachable to
        find_nav_item_with_note.
        """
        if getattr(self.tem, "offline", False):
            return None
        nav_file = self.tem.open_or_create_nav_file(nav_file=self.state.nav_file, reuse=True)
        if nav_file and nav_file != self.state.nav_file:
            self.state.nav_file = nav_file
            self.state.save()
        return nav_file

    def _selected_sites(self, sites=None, include_skipped=False):
        """(site_id, site_data) pairs to act on, in a stable sorted order.

        Sites marked skipped in the session state are left out of every
        per-site stage until they are un-skipped.
        """
        if not self.site_collection or not self.site_collection.sites:
            return []
        wanted = set(sites) if sites else None
        out = []
        for sid in sorted(self.site_collection.sites):
            if wanted is not None and sid not in wanted:
                continue
            if not include_skipped and self.state.is_site_skipped(sid):
                print(f"[INFO] {sid} is marked skipped; not processing it.")
                continue
            out.append((sid, self.site_collection.sites[sid]))
        return out

    def _all_sites_done(self, stage):
        """True when every known site has finished this stage's per-site step.

        Checked across ALL known sites, not just the ones a --sites filter
        selected, so a partial run never marks the whole stage complete.
        """
        step = self.STAGE_SITE_STEP.get(stage)
        if step is None:
            return True
        site_ids = set(self.state.sites)
        if self.site_collection is not None:
            site_ids |= set(self.site_collection.sites)
        if not site_ids:
            return False
        # A skipped site counts as settled -- otherwise the stage could never
        # complete and every later run would re-enter it.
        return all(self.state.is_site_done(sid, step) or self.state.is_site_skipped(sid)
                   for sid in site_ids)

    def _checkpoint(self, site_data, step, status=DONE):
        """Persist a site and record the checkpoint in the session state."""
        pickle_path = None
        try:
            pickle_path = site_data.save()
        except Exception as e:
            print(f"[WARN] Could not checkpoint {site_data.site_id}: {e}")
        self.state.mark_site(site_data.site_id, step, status, pickle_path=pickle_path)
        return pickle_path

    def run(self, start_at=None, only=None, sites=None, force=False):
        """Walk the pipeline, skipping whatever the session state says is done.

        start_at -- begin at this stage (default: first stage not done)
        only     -- run just this one stage
        sites    -- restrict per-site stages to these site ids
        force    -- redo stages and sites already marked done
        """
        if only:
            if only not in self.STAGES:
                raise ValueError(f"Unknown stage {only!r}; expected one of {self.STAGES}.")
            todo = [only]
        else:
            if start_at is None:
                start_at = self.state.next_stage(self.STAGES) or self.STAGES[0]
            if start_at not in self.STAGES:
                raise ValueError(f"Unknown stage {start_at!r}; expected one of {self.STAGES}.")
            todo = list(self.STAGES[self.STAGES.index(start_at):])

        methods = {"experiment_setup": self.run_experiment_setup,
                   "site_setup": self.run_site_setup,
                   "site_montages": self.run_site_montages,
                   "clem_alignment": self.run_clem_alignment}

        # Anything past the first stage needs the sites back in memory.
        if todo and todo[0] != "experiment_setup":
            self.load_site_collection(sites=sites)

        for stage in todo:
            if self.state.is_stage_done(stage) and not force:
                # The per-site records are the truth; the stage flag is a
                # summary of them. If they disagree -- a site was un-skipped,
                # or the JSON was hand-edited -- re-enter the stage and let
                # the per-site checks decide what actually needs doing.
                if self._all_sites_done(stage):
                    print(f"[INFO] Skipping {stage} (already done). Use --force to redo it.")
                    continue
                print(f"[INFO] {stage} was marked done but some sites are "
                      f"outstanding; re-entering it.")

            print(f"\n[INFO] ===== stage: {stage} =====")
            self.state.mark_stage(stage, RUNNING)
            try:
                if stage == "experiment_setup":
                    methods[stage]()
                else:
                    methods[stage](sites=sites, force=force)
            except Exception:
                self.state.mark_stage(stage, FAILED)
                print(f"[ERROR] Stage {stage} failed; progress up to this point is saved in "
                      f"{self.state.path}.")
                raise
            # Only call a per-site stage complete when every site finished it.
            # A partial run (--sites), a dry run that could not acquire, or a
            # site left pending must not mark the stage done -- otherwise the
            # next run skips it entirely.
            if self._all_sites_done(stage):
                self.state.mark_stage(stage, DONE)
                print(f"[INFO] Finished {stage}.")
            else:
                self.state.mark_stage(stage, PENDING)
                step = self.STAGE_SITE_STEP.get(stage)
                # Include sites known only from a folder on disk, not just the
                # ones already in the state file -- those are exactly the ones
                # holding the stage back, so listing only state.sites reported
                # "none recorded" while the stage was plainly incomplete.
                known = set(self.state.sites)
                if self.site_collection is not None:
                    known |= set(self.site_collection.sites)
                outstanding = [sid for sid in sorted(known)
                               if not self.state.is_site_done(sid, step)
                               and not self.state.is_site_skipped(sid)]
                print(f"[INFO] {stage} incomplete; still pending: "
                      f"{', '.join(outstanding) or 'none recorded'}")

        return self.site_summaries

    def run_experiment_setup(self):

        nav_file = self.tem.open_or_create_nav_file(nav_file=None, reuse=False)
        if nav_file:
            self.state.nav_file = nav_file
            self.state.save()
        input("[ToDO] Please load the experiment file and the settings file. ENTER")
             
        self.tem.reset_defocus()
        self.tem.precise_stage_move(stage_x_um=0.0, stage_y_um=0.0, stage_z_um=0.0, stage_tilt=0.0)

        self.tem.acquire_image(mode='View', imaging_state='LMM')
        input('[ToDO] Move feature suitable for eucentricity alignment to the center of the stage (shift + right click + drag). ENTER')
        self.tem.set_eucentricity(level='rough')
        self.tem.acquire_montage(site_data=None, mode='Search', imaging_state='LMM', fov_um_x=3000.0, fov_um_y=3000.0, eucentricity=False)
        from clem_tem_stage_coordinates_tool import App
        app = App()
        app.mainloop()
        self.tem.acquire_image(mode='View', imaging_state='grid_square') 
        self.tem.return_control_to_serialem(message="[ToDo] Run 'Shift to Marker' alignment. ENTER.")

    def run_acquire_position_montages(self, sites=None, force=False):
        """Backwards-compatible wrapper: site setup followed by montages.

        Prefer calling the two stages separately (or via run()), so a crash in
        the montage loop does not cost you the manual centring work again.
        """
        self.run_site_setup(sites=sites, force=force)
        self.run_site_montages(sites=sites, force=force)

    def _center_or_skip(self, site_id):
        """Second centring prompt, which doubles as the skip decision.

        This is the last look you get at the site before eucentricity and the
        overview map, so it is where a bad lamella is cheapest to drop. Folded
        into the existing prompt rather than added as a separate question, so a
        good site still costs exactly one ENTER.

        Returns True if the site was skipped.
        """
        reply = input("Please move the center of the grid square / lamella to the "
                      "center of the field of view  (shift + right click + drag). "
                      f"[ENTER = continue, s = skip {site_id}] ").strip().lower()
        if reply not in ("s", "skip"):
            return False
        reason = input("Optional reason (ENTER for none): ").strip() or None
        self.state.mark_site_skipped(site_id, True, reason=reason)
        print(f"[INFO] {site_id} skipped; it will be left out of the montage "
              f"and alignment stages. Use --unskip-sites {site_id} to undo.")
        return True

    def run_site_setup(self, sites=None, force=False):
        """Per-site eucentricity + overview map, then the Shift-to-Marker handoff.

        This is the operator-expensive half: every site needs manual centring.
        It is checkpointed per site so a restart never repeats it.
        """
        self._ensure_nav_open()
        # Bring back anything a previous run already set up, so we can skip it.
        self.load_site_collection(sites=sites, verbose=False)

        csv_sites = self._import_csv_file(filename='tem_stage_positions.csv')
        wanted = set(sites) if sites else None
        did_any = False

        for site_number, site in enumerate(csv_sites):
            site_id = site.get("name") or f"site_{site_number+1:02d}"
            if wanted is not None and site_id not in wanted:
                continue
            if self.state.is_site_skipped(site_id):
                print(f"[INFO] {site_id} is marked skipped; not processing it.")
                continue
            if self.state.is_site_done(site_id, "setup") and not force:
                print(f"[INFO] Skipping setup for {site_id} (already done).")
                continue

            did_any = True
            site_dir = os.path.join(self.tem.output_root, site_id)
            os.makedirs(site_dir, exist_ok=True)
            print(f"[INFO] Setting up {site_id}.")
            self.state.mark_site(site_id, "setup", RUNNING)
            self.tem.precise_stage_move(stage_x_um=site["stage_x_um"], stage_y_um=site["stage_y_um"], stage_z_um=site["stage_z_um"])
            self.tem.acquire_image(mode='View', imaging_state='grid_square', save=False)
            input("Please move the center of the grid square / lamella to the center of the field of view  (shift + right click + drag). ENTER")
            self.tem.acquire_image(mode='View', imaging_state='grid_square')
            if self._center_or_skip(site_id):
                continue
            self.tem.set_eucentricity(level='rough_fine')
            stage_x_um, stage_y_um, stage_z_um, stage_tilt = self.tem.report_stage_position()
            site_data = SiteDataSummary(site_id=site_id, path=site_dir, stage_position=[stage_x_um, stage_y_um, stage_z_um, stage_tilt], milling_angle=self.milling_angle)
            self.tem.acquire_image(mode='View', imaging_state='grid_square', site_data=site_data, label=f"overview", save=True, simple_note=True)
            self.register_site_data(site_data, active=False)
            self._checkpoint(site_data, "setup")

        if not did_any:
            print("[INFO] All sites were already set up; skipping the Shift to Marker handoff.")
            return self.site_summaries

        self.tem.acquire_image(mode='View', imaging_state='grid_square')
        self.tem.acquire_image(mode='Search', imaging_state=None) # HERE THE SWITCH TO LOWDOSE MODE
        self.tem.return_control_to_serialem(message="Run 'Shift to Marker' alignment. Press Continue to resume.")
        return self.site_summaries

    def run_site_montages(self, sites=None, force=False):
        """Acquire the montage at each site's overview nav item."""
        self._ensure_nav_open()
        if not self.site_collection or not self.site_collection.sites:
            self.load_site_collection(sites=sites)

        for site_id, site_data in self._selected_sites(sites):
            if self.state.is_site_done(site_id, "montage") and not force:
                print(f"[INFO] Skipping montage for {site_id} (already done).")
                continue
            if not site_data.stage_position:
                print(f"[WARN] No stage position recorded for {site_id}; run site_setup first. Skipping.")
                continue

            print(f"[INFO] Acquiring montage for {site_id}.")
            self.state.mark_site(site_id, "montage", RUNNING)
            self.tem.precise_stage_move(stage_x_um = site_data.stage_position[0], stage_y_um = site_data.stage_position[1], stage_z_um = site_data.stage_position[2])
            nav_idx = self.tem.find_nav_item_with_note(f"{site_id}_overview")
            if self.tem.offline is False:
                if nav_idx is None:
                    # The nav item is gone (new nav file, or the map was
                    # deleted). Re-register it from the overview image saved
                    # during setup rather than failing the whole site.
                    nav_idx = self._reregister_overview(site_data)
                if nav_idx is None:
                    print(f"[WARN] No overview nav item for {site_id}; skipping.")
                    self.state.mark_site(site_id, "montage", FAILED)
                    continue
                if self.sample_type == 'airyscan':
                    self.tem.acquire_montage_at_nav_item(site_data=site_data, mode='Search', nav_idx=nav_idx, fov_um_x=self.montage_settings[self.sample_type]['fov_um_x'], fov_um_y=self.montage_settings[self.sample_type]['fov_um_y'], eucentricity=True)
                elif self.sample_type == 'lamella':
                    self.tem.acquire_montage_at_nav_item(site_data=site_data, mode='View', nav_idx=nav_idx, fov_um_x=self.montage_settings[self.sample_type]['fov_um_x'], fov_um_y=self.montage_settings[self.sample_type]['fov_um_y'], eucentricity=True)
                stage_x_um, stage_y_um, stage_z_um, stage_tilt = self.tem.report_stage_position()
                site_data.stage_position = [stage_x_um, stage_y_um, stage_z_um, stage_tilt]
            else:
                # Offline: attach whatever montage is already in the folder.
                attached = self.tem.acquire_montage(site_data=site_data,
                                         fov_um_x=self.montage_settings[self.sample_type]['fov_um_x'],
                                         fov_um_y=self.montage_settings[self.sample_type]['fov_um_y'])
                if attached is None:
                    # Nothing was acquired and nothing exists to attach. Leave
                    # the site pending so a later LIVE run still collects it --
                    # a dry run must never make real work look finished.
                    print(f"[WARN] {site_id}: no montage on disk and offline cannot "
                          f"acquire one; leaving it pending.")
                    self.state.mark_site(site_id, "montage", PENDING)
                    continue

            self._checkpoint(site_data, "montage")
        return self.site_summaries

    def _reregister_overview(self, site_data):
        """Recreate the overview nav map from the image saved during setup.

        Fallback for a restart where the navigator no longer holds the item.
        Returns a nav index, or None if there is nothing to register.
        """
        overview = (site_data.images or {}).get("overview")
        path = getattr(overview, "image_path", None)
        if not path or not os.path.isfile(path):
            return None
        print(f"[INFO] Re-registering overview map for {site_data.site_id} from {path}.")
        try:
            import serialem as sem
            while sem.ReportFileNumber() > 0:
                sem.CloseFile()
            sem.OpenOldFile(path)
            sem.ReadFile(0)
            nav_idx = int(sem.NewMap(0, f"{site_data.site_id}_overview"))
            sem.CloseFile()
            return nav_idx if nav_idx > 0 else None
        except Exception as e:
            print(f"[WARN] Could not re-register overview for {site_data.site_id}: {e}")
            return None

    def register_site_data(self, site_data, active=True):
        if self.site_collection is not None:
            self.site_collection.add_site(site_data)
            if active:
                self.site_collection.set_active_site(site_data.site_id)
        if site_data.site_id is not None:
            self.site_summaries[site_data.site_id] = site_data
        return site_data

    def _site_stage_z(self, site_data):
        """Stage Z for a site, or None if it cannot be determined.

        Prefers the position measured during site_setup. A site rehydrated from
        a bare folder has an empty stage_position, so fall back to the Z the
        montage itself recorded -- otherwise indexing stage_position[2] raises
        IndexError midway through the alignment stage.
        """
        pos = site_data.stage_position or []
        if len(pos) >= 3 and pos[2] is not None:
            return float(pos[2])

        mrc = self.mrc_reader._find_latest_mrc_dataclass(site_data)
        if mrc is not None:
            if mrc.stage_z_um is not None:
                return float(mrc.stage_z_um)
            meta = getattr(mrc, "metadata", None)
            if meta is not None and meta.stage_z_um is not None:
                return float(meta.stage_z_um)
        return None

    def run_clem_alignment(self, sites=None, force=False):
        from clem_ui import RegistrationApp
        if not self.site_collection or not self.site_collection.sites:
            self.load_site_collection(sites=sites)

        for site_id, site_data in self._selected_sites(sites):
            if self.state.is_site_done(site_id, "alignment") and not force:
                print(f"[INFO] Skipping alignment for {site_id} (already done).")
                continue

            # Re-running a site that was already fitted: offer its stored
            # transform back instead of starting from scratch.
            reuse = self.state.is_site_done(site_id, "alignment")
            self.state.mark_site(site_id, "alignment", RUNNING)

            ui = RegistrationApp(mrc_reader=self.mrc_reader, site_data=site_data,
                                 tem_communication=self.tem, reuse_transform=reuse)
            ui.mainloop()
            try:
                ui.destroy()
            except Exception:
                pass

            if site_data.registration is None:
                # UI closed without a fit: leave the site pending so the next
                # run picks it up again rather than silently marking it done.
                print(f"[WARN] No registration produced for {site_id}; leaving it pending.")
                self.state.mark_site(site_id, "alignment", PENDING)
                continue

            if self.sample_type == 'airyscan':
                stage_z = self._site_stage_z(site_data)
                if stage_z is None:
                    print(f"[WARN] No stage Z for {site_id} (no setup position and "
                          f"none in the montage); skipping the high-mag pass. "
                          f"Run site_setup for this site to enable it.")
                    self._checkpoint(site_data, "alignment")
                    continue
                cx, cy = site_data.registration.overlay_center_px
                idx = self.tem.add_nav_point_with_note(cx, cy, stage_z_um=stage_z, buffer="S", note='center')
                self.tem.move_stage_to_nav_item(nav_idx=idx)
                self.tem.acquire_montage_at_nav_item(site_data=site_data, mode='View', nav_idx=idx, fov_um_x=self.montage_settings[self.sample_type]['fov_um_x_high_mag'], fov_um_y=self.montage_settings[self.sample_type]['fov_um_y_high_mag'], eucentricity=True, realign=True)
                self.tem.delete_nav_item(nav_idx=idx)
                ui = RegistrationApp(mrc_reader=self.mrc_reader, site_data=site_data,
                                     tem_communication=self.tem)
                ui.mainloop()
                try:
                    ui.destroy()
                except Exception:
                    pass
            elif self.sample_type == 'lamella':
                print('Lamella alignment complete.')

            # Persist the registration -- without this the fit is lost on exit.
            self._checkpoint(site_data, "alignment")
        return self.site_summaries


