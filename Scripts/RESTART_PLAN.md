# Making the CLEM workflow restartable

Implementation plan for adding stage-level restart to the workflow driven by
`clem_main.py` → `ExecutiveControls` (`clem_general.py`).

Status: **implemented.** The bug fixes in §2 and work items 1-10 in §4 are done
and covered by `tests/test_restart_e2e.py`. Item 11 (pickle slimming) is still
deferred by choice — see §3.6.

Quick reference:

```
python clem_main.py                            # run, or resume after asking
python clem_main.py --resume                   # resume without the prompt
python clem_main.py --restart                  # clear progress, start over
python clem_main.py --status                   # show progress, change nothing
python clem_main.py --start-at clem_alignment  # from this stage onward
python clem_main.py --only site_montages --sites site_03,site_07
python clem_main.py --only site_setup --sites site_03 --force
python clem_main.py --skip-sites site_02,site_05   # drop bad sites for good
python clem_main.py --unskip-sites site_02         # bring one back
python clem_main.py --offline ...              # dry run, no hardware commands
```

### Skipping sites

`--skip-sites` excludes a site from montages and every later stage. Unlike
`--sites`, which filters one invocation, a skip is **recorded in
`clem_session.json` and persists** until `--unskip-sites` clears it — so a
lamella you judge bad after setup stays out of the run without you having to
remember it each time. Skipped sites show as `SKIPPED` in `--status`, and can
carry a reason (`mark_site_skipped(site_id, True, reason="broken lamella")`).

**At the microscope**, the second centring prompt in `site_setup` doubles as the
skip decision:

```
Please move the center of the grid square / lamella to the center of the field
of view  (shift + right click + drag). [ENTER = continue, s = skip site_02]
```

Typing `s` (then an optional reason) marks the site skipped and moves to the
next one, before eucentricity and the overview map — the expensive part. It is
folded into the existing prompt rather than added as a separate question, so a
good site still costs exactly one ENTER. This is the only interactive skip
point; every other stage honours the recorded flag silently.

Two consequences worth knowing:

- A skipped site counts as *settled* for stage completion, so a stage still
  reaches `done` with sites skipped. Without that it could never complete and
  every later run would re-enter it.
- Un-skipping a site whose stage is already `done` still works: `run()` treats
  the per-site records as the truth and the stage flag as a summary of them, so
  it re-enters any stage whose sites disagree with it and processes just the
  outstanding ones. The same holds if you hand-edit the JSON.

---

## 1. The problem

`ExecutiveControls` exposes three entry points — `run_experiment_setup`,
`run_acquire_position_montages`, `run_clem_alignment` — but only the first is
genuinely independent. Four couplings prevent starting anywhere but the top:

| # | Coupling | Where |
|---|---|---|
| 1 | `site_collection` is populated *only* by the montage stage, so a fresh process entering at alignment iterates an empty dict and silently does nothing | `clem_general.py:129` writes; `clem_general.py:156` reads |
| 2 | Every run creates a **new** nav file and closes the old one, orphaning all prior nav items — so `find_nav_item_with_note` can never rediscover them | `clem_tem_communication.py:38-45`, consumed at `clem_general.py:135` |
| 3 | Checkpoints are written but never read; `SiteDataSummary.load()` / `load_latest()` have **zero callers**. Alignment results are never saved at all | written `clem_general.py:143`; unused `clem_dataclasses.py:311-328` |
| 4 | `run_acquire_position_montages` is really two stages in one function, separated by an operator "Shift to Marker" step | `clem_general.py:113-143`, boundary at `:132` |

The current restart mechanism is commenting out line 33 of `clem_main.py`.

---

## 2. Prerequisite bug fixes (DONE)

These were dead or broken code sitting directly in the restart path. All are
fixed and verified — including an offline dry run asserting that no `sem.*`
command executes (§2.5), which is the harness the rest of this plan is tested
against.

### 2.1 `MRCReader._fit_latest_transfer` → `_find_latest_transform`
`clem_mrc_mdoc_reader.py:206`

Three defects in one method: the name did not match its only caller
(`clem_ui.py`, which looked up `_find_latest_transform`); the glob pattern was
`transform_*.haml`, a typo for `.yaml`, which is exactly what
`CLEMCorrelator.save_transform` writes (`clem_correlation.py:309`); and it took
`site_data.path` directly, so it raised `AttributeError` on a plain folder
string.

Now: correct name, routed through `_site_folder()` so it accepts a
`SiteDataSummary`, a `Path` or a folder string, correct `.yaml` pattern, and it
returns `None` instead of raising when nothing is found.

### 2.2 Transform reuse wired up, opt-in — `clem_ui.py:1211`
The startup "re-apply stored transform" block was doubly dead: the finder name
did not resolve, and it passed `self.site_id` (a string) where a `SiteDataSummary`
was expected. It also confused a *path* with a *record* — the finder returns a
file path, but the block used it as if it had `.flip_x` / `.flip_y`.

Now: passes `self.site_data`, parses the path via
`CLEMCorrelator.load_transform()` (handles yaml / csv / legacy txt), and is
gated behind a new `RegistrationApp(..., reuse_transform=False)` flag so
behaviour is unchanged unless asked for. `clem_ui.py --reuse-transform` exposes
it standalone.

### 2.3 `TEMComm.load_mrc_in_nav` — `clem_tem_communication.py:83`
Called `self.mrc_reader.identify_latest_montage_file(site_data)`, which does not
exist on `MRCReader`, outside the `try`, so it raised `AttributeError` rather
than being caught. `idx` could also be returned unbound.

Now: a new `TEMComm.resolve_latest_mrc_dataclass(site_data)` prefers an
`MRCSummary` already in `site_data.mrcs`, then falls back to
`_find_latest_montage_mrc` + `populate_mrc` — **this disk fallback is what lets
a restarted process load a montage acquired by an earlier run.** Returns `None`
with a warning on every degenerate path; `idx` is initialised.

### 2.4 `CLEMPicker` site lookup — `clem_target_picking.py:36`
Read `site_collection.site_data`; the attribute is `.sites`. Now uses
`sc.get_site(self.site_id)`.

### 2.5 Offline mode is now a true no-op
`clem_tem_communication.py`

Offline previously printed a notice and then executed the `sem.*` calls anyway
— `precise_stage_move` (`:283`) and `acquire_montage` (`:365`) both lacked a
`return`, and `report_stage_position`, `create_nav_file`,
`find_nav_item_with_note`, `add_nav_point_with_note`, `delete_nav_item`,
`set_montage_to_buffer` and `return_control_to_serialem` had no offline branch
at all. Offline dry runs of the workflow therefore crashed or drove hardware.

Now every one of them prints its intent and returns without touching SerialEM.
Return values were chosen so downstream guards keep working:

- `report_stage_position` → `(0.0, 0.0, 0.0, 0.0)` with a `[WARN]`. **Site data
  built during an offline run holds dummy coordinates — never save an offline
  run over a real session's pickles.**
- `add_nav_point_with_note` → `0`, not `None`, so `delete_nav_item`'s
  `if nav_idx > 0` guard stays valid (it is also `None`-safe now).
- `find_nav_item_with_note` → `None`, as in the live "not found" case.

`sem.NoMessageBoxOnError()` in `__init__` is left in place — it is deliberate,
and "offline" here means *SerialEM is importable but must not be driven*, not
*SerialEM is absent*.

**`acquire_montage` offline** cannot acquire, so it simulates *"the montage is
already there"*: it skips all `sem.*` and attaches the newest montage already
in the site folder via `resolve_latest_mrc_dataclass` (§2.3), returning `None`
if the folder is empty. This is what makes an offline dry run of a restarted
stage behave like the real thing. `acquire_montage` and
`acquire_montage_at_nav_item` now also *return* the `MRCSummary` in both live
and offline paths, where before they returned `None` implicitly.

### 2.6 `MRCSummary.timestamp` populated from the filename
`clem_dataclasses.py`, `clem_mrc_mdoc_reader.py`

`MRCSummary` had no `timestamp` field, yet three call sites read one via
`getattr(m, "timestamp", None)` — `_find_latest_mrc_dataclass`
(`clem_mrc_mdoc_reader.py:161`) and `_resolve_latest_mrc` in both UIs. So
"newest montage" silently fell back to `st_mtime`, which cloud sync and file
copies rewrite.

Added the field and a `_timestamp_from_filename()` helper that parses the
`YYYYmmdd-HH-MM-SS` already stamped into the name by `acquire_montage`.
`build_montage_summary` sets it, and `_find_latest_montage_mrc` now prefers it,
falling back to mtime for files without one. Both comparators key on a
`(has_timestamp, value)` tuple so string and float keys are never compared.

Verified against the case that motivated it: two montages whose mtimes are
deliberately inverted relative to their filenames — the filename-newest one is
correctly selected.

---

## 3. Design

Four pieces, independently useful, in dependency order.

### 3.1 Split the stages so they are addressable

Break `run_acquire_position_montages` at the `return_control_to_serialem` call
(`clem_general.py:132`) into two methods:

- `run_site_setup()` — the per-site loop (`:115-129`): move, operator centring,
  eucentricity, overview image, register site. Ends with the low-dose switch and
  the Shift-to-Marker handoff.
- `run_site_montages()` — the second loop (`:133-143`): montage at each site's
  overview nav item.

This matters because the first loop is where all the operator time goes. A
crash in the montage loop currently forces you to redo every manual centring.

Then declare the pipeline explicitly on `ExecutiveControls`:

```python
STAGES = ("experiment_setup", "site_setup", "site_montages", "clem_alignment")
```

and add a driver:

```python
def run(self, start_at=None, only=None, sites=None, force=False):
    """Walk STAGES from start_at (or the first not-done stage), skipping
    stages already recorded as complete unless force=True."""
```

`clem_main.py` becomes argparse over that, replacing the module-level constants
and the commented-out line:

```
python clem_main.py --path Z:\tomo\s26aug18a --start-at clem_alignment
python clem_main.py --only site_montages --sites site_03,site_07
python clem_main.py --only site_setup --sites site_03 --force
```

### 3.2 Session state file

`<output_root>/clem_session.json`, written after each stage and after each site
within a stage:

```json
{
  "path": "Z:\\tomo\\s26aug18a",
  "sample_type": "lamella",
  "milling_angle": -15.0,
  "nav_file": "nav_file_20260819-14-02-11.nav",
  "stages": {"experiment_setup": "done", "site_setup": "done",
             "site_montages": "running", "clem_alignment": "pending"},
  "sites": {
    "site_01": {"setup": "done", "montage": "done", "alignment": "pending",
                "pickle": "site_01/site_01_20260819-14-31-02.pkl"}
  }
}
```

Deliberately JSON, not pickle: you can open it, see where the run stopped, and
force a redo by flipping one value. Keep the split clean — **pickles hold data,
the JSON holds progress.** A small `SessionState` class (load / save / mark /
is_done) in a new `clem_session.py`, ~60 lines, with an atomic write (temp file
+ `os.replace`) so a crash mid-write cannot corrupt it.

Guard on mismatch: if `sample_type` or `milling_angle` in the file disagrees
with the current run, refuse to resume and say why.

### 3.3 Rehydrate `site_collection` on entry

The keystone. Add to `ExecutiveControls`:

```python
def load_site_collection(self, sites=None):
    """Rebuild site_collection from disk so a stage can run in a fresh
    process. Prefers the newest pickle per site; falls back to a bare
    SiteDataSummary built from the folder."""
```

For each site folder under `output_root` (filtered by `sites=` if given):

1. `SiteDataSummary.load_latest(folder, site_id)` — `clem_dataclasses.py:321`.
2. On `FileNotFoundError`, construct
   `SiteDataSummary(site_id=basename(folder), path=folder, milling_angle=self.milling_angle)`.
   `clem_ui.py:1745` already does exactly this in its standalone `__main__`, so
   the pattern is proven — the UI then auto-loads the newest MRC and TIFF from
   the folder via `_display_loaded_site_data` (`clem_ui.py:907`).
3. `self.register_site_data(site_data, active=False)`.

Call it at the top of `run()` for any `start_at` other than `experiment_setup`.
Site ordering should be sorted by name so montage and alignment loops are
deterministic across restarts (`dict` insertion order otherwise depends on
directory scan order).

### 3.4 Nav file reuse

Replace `create_nav_file()` (`clem_tem_communication.py:38`) with:

```python
def open_or_create_nav_file(self, nav_file=None, reuse=True):
    """Reopen nav_file if it exists, else create a new timestamped one.
    Returns the path actually opened."""
```

If the state file names a nav and it is on disk, `sem.ReadNavFile` it rather
than `sem.OpenNavigator` on a fresh path; record the path in the state file
either way. Without this, `find_nav_item_with_note(f"{site_id}_overview")`
(`clem_general.py:135`) returns `None` after every restart and the montage stage
cannot find its maps. Keep `create_nav_file` as a thin wrapper so nothing else
breaks.

Fallback worth building: if the note lookup still misses, re-register the map
from the saved overview image path in `site_data.images["overview"]` rather than
failing the site.

### 3.5 More checkpoints

- After `register_site_data` in `run_site_setup` — currently the operator
  centring work is unsaved until the montage completes.
- After each site in `run_clem_alignment` — **`site_data.save()` is never called
  there today**, so `site_data.registration` (set at `clem_ui.py:1464`) and
  `overlay_center_px` (`:1479`, consumed at `clem_general.py:166`) are lost on
  exit.
- Record the written pickle path into the state file so rehydration does not
  have to glob.

### 3.6 Pickle size (plan only — not to build yet)

A `SiteDataSummary` pickle currently embeds the assembled montage array
(`MRCSummary.image`), the full `(C,Z,Y,X)` TIFF stack (`TiffSummary.stack_czyx`)
and `RegistrationSummary.warped_channels`. With a checkpoint per site per stage
this becomes GB-scale fast.

Proposed when it starts to hurt: `__getstate__` / `__setstate__` on
`MRCSummary` and `TiffSummary` that drop the arrays on save and reload them
lazily from `mrc_path` / `ome_path` on first access. The paths are already
stored, and `MRCReader.load_mrc_into_data_class` / `load_tiff_into_data_class`
already do the reloading — but note both **reset `site_data.registration = None`**
as a side effect (`clem_mrc_mdoc_reader.py:273-281`), which must not happen
during a lazy reload. Fix that first.

Interim mitigation: prune old pickles, keeping the newest N per site.

---

## 4. Ordered work items

Each item is independently testable offline (`OFFLINE=True`, no SerialEM).

| # | Item | Files | Status |
|---|---|---|---|
| 1 | `load_site_collection()` | `clem_general.py` | **done** — test 5, 8 |
| 2 | `SessionState` class | new `clem_session.py` | **done** — atomic write, corrupt-file recovery |
| 3 | Split into `run_site_setup` / `run_site_montages` | `clem_general.py` | **done** — `run_acquire_position_montages` kept as a wrapper |
| 4 | `STAGES` + `run(start_at, only, sites, force)` | `clem_general.py` | **done** |
| 5 | argparse front end | `clem_main.py` | **done** — old constants are now defaults |
| 6 | State writes at stage and site boundaries | `clem_general.py` | **done** — test 1 |
| 7 | Per-site skip on `is_done` unless `--force` | `clem_general.py` | **done** — tests 2, 3 |
| 8 | `open_or_create_nav_file` + nav path in state | `clem_tem_communication.py` | **done** — on by default; see §6 |
| 9 | `site_data.save()` after alignment; pickle path into state | `clem_general.py` | **done** — test 8 |
| 10 | `reuse_transform=True` when re-aligning a fitted site | `clem_general.py` | **done** — test 6 |
| 11 | Pickle slimming (§3.6) | `clem_dataclasses.py`, `clem_mrc_mdoc_reader.py` | deferred by choice |

Regression suite: `tests/test_restart_e2e.py`. It stubs SerialEM so that *any*
hardware command raises, stubs the GUI and operator input, then runs the
pipeline, interrupts it mid-stage, and resumes it. Run it with
`python tests/test_restart_e2e.py`.

---

## 5. Risks and things to decide

- **Nav reuse is the riskiest item.** Reopening a stale nav file on the
  microscope could put maps and the current stage state out of sync. Worth a dry
  run with `OFFLINE=True` and a copied nav before trusting it live. Also
  consider whether reopening should be explicitly opt-in (`--reuse-nav`) rather
  than automatic.
- **Site identity depends on folder name.** `site_id` comes from the CSV `name`
  column (`clem_general.py:116`) but rehydration infers it from the folder name.
  These agree today because the folder is created as
  `os.path.join(output_root, site_id)` (`:117`) — worth an assertion so it stays
  true.
- **`output_root` has no session subfolder.** `TEMComm.__init__` takes `PATH`
  verbatim, so two experiments sharing a path would share state. Good for
  restart, bad for isolation. If that ever changes, the state file must move
  with it.
- ~~Offline mode is not a true no-op.~~ **Fixed — see §2.5.** Offline dry runs
  are now a supported way to test the restart logic. Remaining caveat: stage
  positions read back as zeros offline, so do not let an offline run write
  pickles into a real session folder.
- ~~`MRCSummary.timestamp` is never set.~~ **Fixed — see §2.6.** Ordering is now
  by the filename timestamp, so Google Drive sync no longer perturbs "newest".
  Caveat: OME-TIFF and CZI discovery (`_find_latest_ome_tiff`,
  `_find_latest_czi`) still order by mtime, since light-microscope filenames
  carry no known timestamp convention.
- `MRCSummary.groups` is never populated (`clem_target_picking.py:118-160`
  returns groups without assigning them), so `TEMComm.add_nav_point` always
  takes the ungrouped branch. Not a restart blocker, but it means picks/groups
  are not currently part of the persisted state and cannot be resumed.

---

## 6. Behaviour notes and what still needs your attention

Two rules emerged while building this; both are enforced and tested.

**A stage is only "done" when every site finished it.** A partial run
(`--sites site_03`), a site left pending, or a dry run that could not acquire
leaves the stage `pending`, so the next run picks up the rest. Without this the
stage-level flag masks unfinished sites and the next run skips them entirely.

**A dry run never makes real work look finished.** Offline, if there is no
montage on disk to attach, the site is left `pending` rather than marked done,
so a later live run still collects it. Separately, resuming a session that was
touched by an offline run on the microscope is refused outright (its stage
coordinates are placeholder zeros) unless you pass `--force`.

An interrupted site is left `running`, which records which site was in flight.
`is_site_done()` is false for it, so a resume redoes that site.

### Needs your attention

1. **Nav reuse on the microscope (§3.4) has not been exercised live.** It is on
   by default, per your call. The logic is: reopen the nav recorded in
   `clem_session.json` via `sem.ReadNavFile`, else create a new one. Two things
   to confirm on the scope before trusting it on a real session:
   - `sem.ReportNavFile()` — used to check whether the *already open* nav is the
     one we want. If that command does not exist in your SerialEM build, the
     code falls back to closing and rereading, which is correct but noisier.
     Worth confirming the name.
   - Reopening a nav from an earlier session while the stage has moved. Suggest
     a dry run against a **copy** of a finished session folder first.
2. **`_reregister_overview` is a new fallback** for the case where the nav item
   is missing after a restart: it reopens the saved overview image and calls
   `NewMap` to recreate the map. It only triggers when
   `find_nav_item_with_note` returns `None`. This has not been run live either.
3. **Stage ordering assumption.** `run()` walks
   `experiment_setup -> site_setup -> site_montages -> clem_alignment`. If you
   ever want alignment interleaved per site rather than as a final sweep, the
   driver would need a per-site outer loop instead.
4. **`experiment_setup` is not per-site checkpointed** — it is all-or-nothing,
   since it ends with the operator-driven stage-coordinates tool. Re-running it
   makes a new nav file.
5. **Pickle size (§3.6) is still unaddressed.** Checkpoints now happen at three
   points per site instead of one, so this will bite sooner than before. Prune
   old `*.pkl` per site, or do item 11, when it starts to hurt.
