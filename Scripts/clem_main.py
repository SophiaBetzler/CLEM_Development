"""CLEM workflow entry point.

Runs the pipeline:  experiment_setup -> site_setup -> site_montages -> clem_alignment

Progress is recorded in <path>/clem_session.json, so an interrupted run can be
resumed instead of restarted. With no arguments the script finds a previous
session and asks whether to resume it.

Examples
--------
    python clem_main.py                                  # run / resume, asks
    python clem_main.py --resume                         # resume, no prompt
    python clem_main.py --restart                        # start over, no prompt
    python clem_main.py --status                         # show progress, do nothing
    python clem_main.py --start-at clem_alignment        # from this stage on
    python clem_main.py --only site_montages --sites site_03,site_07
    python clem_main.py --only site_setup --sites site_03 --force
    python clem_main.py --skip-sites site_02,site_05      # drop bad sites
    python clem_main.py --unskip-sites site_02            # bring one back
"""

import argparse
import os
import sys
import traceback

from clem_mrc_mdoc_reader import MRCReader
from clem_tem_communication import TEMComm
from clem_general import ExecutiveControls
from clem_dataclasses import AllSitesDataCollection
from clem_session import SessionState


# Defaults; override on the command line.
PATH = r"Z:\tomo\s26aug20a"
SAMPLE_TYPE = 'lamella'
OFFLINE = False


def milling_angle_for(sample_type):
    return 0.0 if sample_type == 'airyscan' else -15.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run or resume the CLEM workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("--path", default=PATH,
                   help=f"Experiment output root (default: {PATH})")
    p.add_argument("--sample-type", default=SAMPLE_TYPE, choices=("lamella", "airyscan"),
                   help=f"Sample type (default: {SAMPLE_TYPE})")
    p.add_argument("--milling-angle", type=float, default=None,
                   help="Milling angle in degrees (default: 0 for airyscan, -15 otherwise)")
    p.add_argument("--offline", action="store_true", default=OFFLINE,
                   help="Dry run: no commands are sent to the microscope")

    p.add_argument("--start-at", choices=ExecutiveControls.STAGES,
                   help="Begin at this stage (default: first stage not yet done)")
    p.add_argument("--only", choices=ExecutiveControls.STAGES,
                   help="Run only this stage")
    p.add_argument("--sites", default=None,
                   help="Comma-separated site ids to restrict per-site stages to")
    p.add_argument("--skip-sites", default=None,
                   help="Comma-separated site ids to exclude from montages and "
                        "all later stages (persists until --unskip-sites)")
    p.add_argument("--unskip-sites", default=None,
                   help="Comma-separated site ids to bring back into the run")
    p.add_argument("--force", action="store_true",
                   help="Redo stages and sites already marked done")

    resume = p.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true",
                        help="Resume a previous session without asking")
    resume.add_argument("--restart", action="store_true",
                        help="Discard previous progress and start from the top")

    p.add_argument("--status", action="store_true",
                   help="Print the session progress and exit")
    return p.parse_args(argv)


def ask_resume_or_restart(state):
    """Ask the operator what to do with an existing session.

    Returns "resume", "restart" or "quit".
    """
    print()
    print(state.summary(ExecutiveControls.STAGES))
    for problem in state.check_compatible():
        print(f"  [WARN] {problem}")
    print()
    prompt = ("A previous session was found in this folder.\n"
              "  [R] resume where it stopped   (default)\n"
              "  [S] start over from the top   (previous progress is cleared)\n"
              "  [Q] quit\n"
              "Choice: ")
    while True:
        try:
            choice = input(prompt).strip().lower()
        except EOFError:
            return "resume"
        if choice in ("", "r", "resume"):
            return "resume"
        if choice in ("s", "start", "restart"):
            confirm = input("This clears the recorded progress (your data files are "
                            "untouched). Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                return "restart"
            continue
        if choice in ("q", "quit", "exit"):
            return "quit"
        print("Please answer R, S or Q.")


def main(argv=None):
    args = parse_args(argv)

    milling_angle = (args.milling_angle if args.milling_angle is not None
                     else milling_angle_for(args.sample_type))

    def _split(value):
        return [s.strip() for s in value.split(",") if s.strip()] if value else []

    sites = _split(args.sites) or None

    if not os.path.isdir(args.path):
        print(f"[ERROR] Experiment folder does not exist: {args.path}")
        return 2

    state = SessionState.load(args.path, sample_type=args.sample_type,
                              milling_angle=milling_angle, offline=args.offline)

    # Applied before anything else, so --skip-sites can be combined with
    # --status to mark sites and immediately see the result.
    for site_id in _split(args.skip_sites):
        state.mark_site_skipped(site_id, True)
        print(f"[INFO] {site_id} marked skipped.")
    for site_id in _split(args.unskip_sites):
        state.mark_site_skipped(site_id, False)
        print(f"[INFO] {site_id} un-skipped.")

    if args.status:
        print(state.summary(ExecutiveControls.STAGES) if state.exists()
              else f"No session file in {args.path}.")
        return 0

    # Decide what to do about prior progress.
    if state.has_progress():
        if args.restart:
            decision = "restart"
        elif args.resume or args.start_at or args.only:
            decision = "resume"     # an explicit stage selection implies resuming
        else:
            decision = ask_resume_or_restart(state)

        if decision == "quit":
            print("Nothing done.")
            return 0
        if decision == "restart":
            state.reset()
            print("[INFO] Previous progress cleared.")
        else:
            problems = state.check_compatible()
            if problems and not args.force:
                print("[ERROR] Refusing to resume:")
                for problem in problems:
                    print(f"  - {problem}")
                print("Re-run with --force to override, or --restart to start over.")
                return 2

    site_collection = AllSitesDataCollection()
    mrc = MRCReader(coord_key="AlignedPieceCoordsVS", section=0)
    temcom = TEMComm(mrc_reader=mrc, path=args.path, offline=args.offline)
    exc = ExecutiveControls(tem_communication=temcom, mrc_reader=mrc,
                            sample_type=args.sample_type, milling_angle=milling_angle,
                            site_collection=site_collection, session_state=state)

    try:
        exc.run(start_at=args.start_at, only=args.only, sites=sites, force=args.force)
    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted. Progress is saved in {state.path}; "
              f"re-run to resume.")
        return 130
    except Exception:
        traceback.print_exc()
        print(f"\n[INFO] Progress is saved in {state.path}; re-run to resume.")
        input("Crashed - press ENTER to close")
        return 1

    print()
    print(state.summary(ExecutiveControls.STAGES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
