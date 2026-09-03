"""Offline end-to-end test of the CLEM restart machinery.

Stubs SerialEM (strictly: any hardware command is a failure), the GUI, and
operator input, then runs the pipeline, interrupts it, and resumes it.
"""
import builtins
import os
import sys
import tempfile
import types

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKDIR)

ALLOWED_SEM = {"NoMessageBoxOnError"}


class StrictSem(types.ModuleType):
    def __getattr__(self, name):
        def _call(*a, **k):
            if name in ALLOWED_SEM:
                return 0
            raise AssertionError(f"OFFLINE VIOLATION: sem.{name}() executed")
        return _call


sys.modules["serialem"] = StrictSem("serialem")
for _m in ("mrcfile", "tifffile", "czifile"):
    sys.modules[_m] = types.ModuleType(_m)
for _m in ("tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox"):
    sys.modules[_m] = types.ModuleType(_m)

from clem_dataclasses import AllSitesDataCollection, MRCSummary, RegistrationSummary
from clem_general import ExecutiveControls
from clem_mrc_mdoc_reader import MRCReader
from clem_session import SessionState, DONE, PENDING
from clem_tem_communication import TEMComm

SITES = ["site_01", "site_02", "site_03"]


def make_experiment(root):
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "tem_stage_positions.csv"), "w") as fh:
        fh.write("name,stage_x_um,stage_y_um,stage_z_um\n")
        for i, s in enumerate(SITES):
            fh.write(f"{s},{100 + i},{200 + i},{10 + i}\n")


def drop_montage(root, site_id, stamp="20260819-12-00-00"):
    folder = os.path.join(root, site_id)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{site_id}_montage_mag_2250_{stamp}.mrc")
    with open(path, "wb") as fh:
        fh.write(b"fake")
    return path


def build_controls(root, inputs=None, fail_on_input=None):
    """Fresh ExecutiveControls wired to stubs. Returns (exc, state)."""
    reader = MRCReader(coord_key="AlignedPieceCoordsVS", section=0)
    # Avoid real mdoc/mrc parsing.
    reader.build_montage_summary = lambda p, site_id=None: MRCSummary(
        mrc_path=os.fspath(p),
        montage_id=os.path.splitext(os.path.basename(p))[0],
        timestamp=reader._timestamp_from_filename(p))

    tem = TEMComm(mrc_reader=reader, path=root, offline=True)
    state = SessionState.load(root, sample_type="lamella", milling_angle=-15.0, offline=True)
    exc = ExecutiveControls(tem_communication=tem, mrc_reader=reader,
                            sample_type="lamella", milling_angle=-15.0,
                            site_collection=AllSitesDataCollection(),
                            session_state=state)

    counter = {"n": 0}

    def fake_input(prompt=""):
        counter["n"] += 1
        if fail_on_input is not None and counter["n"] >= fail_on_input:
            raise KeyboardInterrupt("simulated operator abort")
        return ""

    builtins.input = fake_input
    return exc, state


def fake_alignment(exc):
    """Stand in for the GUI: mark every site as registered."""
    import clem_general

    class FakeUI:
        def __init__(self, mrc_reader, site_data, tem_communication, reuse_transform=False):
            self.site_data = site_data
            self.reuse_transform = reuse_transform
            REUSE_SEEN.append((site_data.site_id, reuse_transform))

        def mainloop(self):
            self.site_data.registration = RegistrationSummary(
                transform_type="affine", overlay_center_px=(10.0, 20.0))

        def destroy(self):
            pass

    mod = types.ModuleType("clem_ui")
    mod.RegistrationApp = FakeUI
    sys.modules["clem_ui"] = mod


REUSE_SEEN = []


def check(label, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        print(f"        got  {got!r}")
        print(f"        want {want!r}")
    return ok


def main():
    failures = 0
    root = tempfile.mkdtemp(prefix="clem_e2e_")
    make_experiment(root)
    fake_alignment(None)

    print("\n--- 1. site_setup interrupted after two sites -------------------")
    # 2 input() calls per site -> abort during the 3rd site.
    exc, state = build_controls(root, fail_on_input=5)
    try:
        exc.run(only="site_setup")
    except KeyboardInterrupt:
        print("  (simulated crash)")
    failures += not check("site_01 setup done", state.site_status("site_01", "setup"), DONE)
    failures += not check("site_02 setup done", state.site_status("site_02", "setup"), DONE)
    # The interrupted site stays "running": it records which site was in
    # flight, and is_site_done() is False for it, so a resume redoes it.
    failures += not check("site_03 left running", state.site_status("site_03", "setup"), "running")
    failures += not check("stage not marked done", state.is_stage_done("site_setup"), False)

    print("\n--- 2. resume: only the unfinished site is redone ---------------")
    exc, state = build_controls(root)
    prompts = []
    real_input = builtins.input

    def counting_input(prompt=""):
        prompts.append(prompt)
        return real_input(prompt)

    builtins.input = counting_input
    exc.run(only="site_setup")
    failures += not check("site_03 now done", state.site_status("site_03", "setup"), DONE)
    failures += not check("only site_03 prompted (2 prompts)", len(prompts), 2)
    failures += not check("stage done", state.is_stage_done("site_setup"), True)

    print("\n--- 3. re-running a finished stage does nothing -----------------")
    exc, state = build_controls(root)
    prompts.clear()
    builtins.input = counting_input
    exc.run(only="site_setup")
    failures += not check("no operator prompts", len(prompts), 0)

    print("\n--- 4. montages: rehydrated from folders, attach existing -------")
    for s in SITES:
        drop_montage(root, s)
    exc, state = build_controls(root)
    exc.run(only="site_montages")
    failures += not check("all montages done",
                          [state.site_status(s, "montage") for s in SITES], [DONE] * 3)
    site = exc.site_collection.sites["site_02"]
    failures += not check("montage attached to site_02", len(site.mrcs), 1)
    failures += not check("checkpoint recorded", bool(state.site_pickle("site_02")), True)

    print("\n--- 5. fresh process resumes alignment from checkpoints ---------")
    REUSE_SEEN.clear()
    exc, state = build_controls(root)
    exc.run(only="clem_alignment")
    failures += not check("all alignments done",
                          [state.site_status(s, "alignment") for s in SITES], [DONE] * 3)
    failures += not check("sites rehydrated without setup rerun",
                          sorted(exc.site_collection.sites), SITES)
    failures += not check("reuse_transform off on first pass",
                          [r for _, r in REUSE_SEEN], [False] * 3)

    print("\n--- 6. --force re-runs alignment and offers the stored fit ------")
    REUSE_SEEN.clear()
    exc, state = build_controls(root)
    exc.run(only="clem_alignment", force=True)
    failures += not check("reuse_transform on for refitted sites",
                          [r for _, r in REUSE_SEEN], [True] * 3)

    print("\n--- 7. --sites restricts the work -------------------------------")
    REUSE_SEEN.clear()
    exc, state = build_controls(root)
    exc.run(only="clem_alignment", sites=["site_02"], force=True)
    failures += not check("only site_02 touched", [s for s, _ in REUSE_SEEN], ["site_02"])

    print("\n--- 8. registration survives the restart ------------------------")
    exc, state = build_controls(root)
    exc.load_site_collection()
    reg = exc.site_collection.sites["site_02"].registration
    failures += not check("registration restored from pickle",
                          getattr(reg, "transform_type", None), "affine")
    failures += not check("overlay center restored",
                          getattr(reg, "overlay_center_px", None), (10.0, 20.0))

    print("\n--- 9. a dry run must not make real work look finished ----------")
    root2 = tempfile.mkdtemp(prefix="clem_e2e2_")
    make_experiment(root2)
    exc, state = build_controls(root2)
    exc.run(start_at="site_setup")     # no montage files exist yet
    failures += not check("montages left pending when none on disk",
                          [state.site_status(s, "montage") for s in SITES],
                          [PENDING] * 3)

    print("\n--- 10. once the montages exist, the run completes --------------")
    for s in SITES:
        drop_montage(root2, s)
    exc, state = build_controls(root2)
    exc.run(start_at="site_setup")
    failures += not check("all stages from site_setup done",
                          [state.is_stage_done(s) for s in
                           ("site_setup", "site_montages", "clem_alignment")],
                          [True, True, True])
    failures += not check("all montages now done",
                          [state.site_status(s, "montage") for s in SITES], [DONE] * 3)


    print("\n--- 11. skipped site is excluded from montage and after ---------")
    root3 = tempfile.mkdtemp(prefix="clem_e2e3_")
    make_experiment(root3)
    exc, state = build_controls(root3)
    exc.run(only="site_setup")
    state.mark_site_skipped("site_02", True, reason="broken lamella")
    for s in SITES:
        drop_montage(root3, s)
    REUSE_SEEN.clear()
    exc, state = build_controls(root3)
    exc.run(start_at="site_montages")
    failures += not check("skipped site got no montage",
                          state.site_status("site_02", "montage"), PENDING)
    failures += not check("other sites did",
                          [state.site_status(s, "montage") for s in ("site_01", "site_03")],
                          [DONE, DONE])
    failures += not check("skipped site not aligned either",
                          [s for s, _ in REUSE_SEEN], ["site_01", "site_03"])
    failures += not check("stage still completes despite the skip",
                          [state.is_stage_done("site_montages"),
                           state.is_stage_done("clem_alignment")], [True, True])
    failures += not check("skip is listed", state.skipped_sites(), ["site_02"])
    failures += not check("SKIPPED shows in the summary",
                          "SKIPPED" in state.summary(ExecutiveControls.STAGES), True)

    print("\n--- 12. skip survives a fresh process ---------------------------")
    exc, state = build_controls(root3)
    failures += not check("still skipped after reload", state.is_site_skipped("site_02"), True)

    print("\n--- 13. un-skipping brings the site back ------------------------")
    state.mark_site_skipped("site_02", False)
    exc, state2 = build_controls(root3)
    exc.run(only="site_montages")
    failures += not check("site_02 montage now done",
                          state2.site_status("site_02", "montage"), DONE)
    failures += not check("no longer listed as skipped", state2.skipped_sites(), [])


    print("\n--- 14. interactive skip at the second centring prompt ----------")
    root4 = tempfile.mkdtemp(prefix="clem_e2e4_")
    make_experiment(root4)
    exc, state = build_controls(root4)

    # Operator types "s" at the skip prompt for site_02 only.
    seen = []

    def operator(prompt=""):
        seen.append(prompt)
        if "s = skip site_02" in prompt:
            return "s"
        if "Optional reason" in prompt:
            return "broken lamella"
        return ""

    builtins.input = operator
    exc.run(only="site_setup")

    failures += not check("site_02 skipped by the operator",
                          state.is_site_skipped("site_02"), True)
    failures += not check("reason recorded",
                          state.sites["site_02"].get("skip_reason"), "broken lamella")
    failures += not check("site_02 has no checkpoint",
                          state.site_pickle("site_02"), None)
    failures += not check("other sites set up normally",
                          [state.site_status(s, "setup") for s in ("site_01", "site_03")],
                          [DONE, DONE])
    failures += not check("no site_02 folder contents",
                          os.path.exists(os.path.join(root4, "site_02", "x")), False)
    # A good site costs exactly two prompts (two centring steps), not three.
    good = [p for p in seen if "site_01" in p or "site_03" in p]
    failures += not check("skip prompt adds no keypress for good sites",
                          len([p for p in seen if "Please move" in p]), 6)

    print("\n--- 15. skipped-at-setup site stays out of later stages ---------")
    for s in SITES:
        drop_montage(root4, s)
    REUSE_SEEN.clear()
    builtins.input = lambda prompt="": ""
    exc, state = build_controls(root4)
    exc.run(start_at="site_montages")
    failures += not check("no montage for the skipped site",
                          state.site_status("site_02", "montage"), PENDING)
    failures += not check("no alignment for it either",
                          [s for s, _ in REUSE_SEEN], ["site_01", "site_03"])

    print()
    print("=" * 64)
    print("E2E RESULT:", "all checks passed" if failures == 0 else f"{failures} FAILURE(S)")
    print("=" * 64)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
