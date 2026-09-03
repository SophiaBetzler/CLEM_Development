"""Channel-role logic in clem_ui: colours, composite membership, panel plans.

Imports only the pure helpers out of clem_ui (no Tk, no display needed), so it
runs anywhere. Run with:  python tests/test_channel_roles.py
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)


def _load_helpers(module):
    """Exec the helper block of a UI module without importing the GUI."""
    src = open(os.path.join(SCRIPTS, module)).read()
    start = src.index("CHANNEL_HEX")
    # The downsample helper is named _fast_ds in some modules, fast_ds in others.
    ends = [src.index(m) for m in ("def _fast_ds", "def fast_ds") if m in src]
    if not ends:
        raise AssertionError(f"{module}: no fast_ds marker to bound the helper block")
    # The block may contain dataclasses or type hints; supply the names their
    # definitions need. Nothing here is called, only defined.
    import dataclasses
    import math
    import typing
    ns = {"np": np, "math": math, "os": os, "re": re,
          "dataclass": dataclasses.dataclass, "field": dataclasses.field,
          "Optional": typing.Optional, "Any": typing.Any, "List": typing.List,
          "Tuple": typing.Tuple, "Dict": typing.Dict}
    exec(compile(src[start:min(ends)], f"<{module} helpers>", "exec"), ns)
    return ns


# Both the pipeline UI and the standalone single-file tool carry the same
# channel-role logic; test them identically.
MODULES = ("clem_ui.py", "test.py", "clem_correlation_tool_standalone.py")

# Panel titles differ slightly between the pipeline UI ("MRC (reference)",
# "TEM + reflection (green)") and the standalone tool ("TEM map",
# "map + reflection (green)"), so the expected titles are per module.
TITLES = {
    "default": {"base": "MRC (reference)", "refl": "TEM + reflection (green)",
                "ch": "MRC + Ch {i} ({role})", "full": "Full composite"},
    "clem_correlation_tool_standalone.py": {
        "base": "TEM map", "refl": "map + reflection (green)",
        "ch": "map + Ch {i} ({role})", "full": "composite"},
}
H = None
FAILURES = []


def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
        return
    FAILURES.append(f"[{CURRENT}] {label}")
    print(f"  FAIL  {label}\n        got  {got!r}\n        want {want!r}")


def titles(roles, n):
    return [t for t, _c, _g in H["overlay_panel_plan"](roles, n)]


def T():
    return TITLES.get(CURRENT, TITLES["default"])


def expect(*parts):
    """Build the expected title list from tokens: 'base', 'refl', 'full',
    or (index, role) for a per-channel panel."""
    t = T()
    out = []
    for p in parts:
        if isinstance(p, tuple):
            out.append(t["ch"].format(i=p[0], role=p[1]))
        else:
            out.append(t[p])
    return out


def run_for_module(module):
    global H, CURRENT
    CURRENT = module
    H = _load_helpers(module)
    print(f"\n{'=' * 60}\n  {module}\n{'=' * 60}")
    checks()


def checks():
    print("\n--- roles and colours ------------------------------------------")
    check("positional defaults",
          [H["default_role"](i) for i in range(5)],
          ["reflection", "red", "green", "blue", "off"])
    check("reflection is green", H["role_rgb"]("reflection"), (0.0, 1.0, 0.0))
    check("red/green/blue", [H["role_rgb"](r) for r in ("red", "green", "blue")],
          [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
    check("off has no colour", H["role_rgb"]("off"), None)
    check("reflection is not a composite role",
          [H["is_composite_role"](r) for r in H["CHANNEL_ROLES"]],
          [False, True, True, True, False])

    print("\n--- full 4-channel stack ---------------------------------------")
    full = ["reflection", "red", "green", "blue"]
    check("panel order", titles(full, 4),
          expect("base", "refl", (1, "red"), (2, "green"), (3, "blue"), "full"))
    plan = H["overlay_panel_plan"](full, 4)
    refl_colors = plan[1][1]
    check("reflection panel shows only the reflection channel",
          refl_colors, [(0.0, 1.0, 0.0), None, None, None])
    comp_colors = plan[-1][1]
    check("composite excludes reflection",
          comp_colors, [None, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])

    print("\n--- partial stacks ---------------------------------------------")
    check("no reflection: no reflection panel",
          titles(["red", "green", "blue"], 3),
          expect("base", (0, "red"), (1, "green"), (2, "blue"), "full"))
    check("reflection + one fluorophore: no redundant composite",
          titles(["reflection", "red"], 2),
          expect("base", "refl", (1, "red")))
    check("reflection only",
          titles(["reflection"], 1),
          expect("base", "refl"))
    check("roles list shorter than the stack is padded off",
          titles(["reflection", "red"], 4),
          expect("base", "refl", (1, "red")))
    check("a channel switched off drops out",
          titles(["reflection", "red", "off", "blue"], 4),
          expect("base", "refl", (1, "red"), (3, "blue"), "full"))
    check("reassigned order is honoured, not position",
          titles(["blue", "reflection", "red"], 3),
          expect("base", "refl", (0, "blue"), (2, "red"), "full"))
    check("two reflection channels share one panel",
          titles(["reflection", "reflection", "red", "green"], 4),
          expect("base", "refl", (2, "red"), (3, "green"), "full"))

    print("\n--- rendering --------------------------------------------------")
    co = H["composite_overlay"]
    mrc = np.full((3, 3), 0.5, np.float32)
    chans = [np.ones((3, 3), np.float32) for _ in range(4)]

    plan = H["overlay_panel_plan"](full, 4)
    refl_img = co(mrc, chans, colors=plan[1][1])
    check("TEM+reflection renders green", list(refl_img[0, 0]), [0.0, 1.0, 0.0])

    red_img = co(mrc, chans, colors=plan[2][1])
    check("single red channel renders red", list(red_img[0, 0]), [1.0, 0.0, 0.0])

    dark = [np.zeros((3, 3), np.float32) for _ in range(4)]
    base = co(mrc, dark, colors=plan[-1][1])
    check("empty channels leave the MRC at alpha 0.6",
          [round(float(v), 3) for v in base[0, 0]], [0.3, 0.3, 0.3])

    check("all-None colours is a no-op",
          np.allclose(co(mrc, chans, colors=[None] * 4), co(mrc, [])), True)
    check("legacy positional call still works", co(mrc, chans).shape, (3, 3, 3))

    print("\n--- filename slugs ---------------------------------------------")
    slugs = [H["_panel_slug"](t) for t in titles(full, 4)]
    check("slugs are filename-safe",
          all(re.fullmatch(r"[a-z0-9_]+", s) for s in slugs), True)
    check("slugs are unique", len(set(slugs)), len(slugs))
    check("reflection slug names the channel",
          any("reflection" in s for s in slugs), True)


def main():
    for module in MODULES:
        run_for_module(module)
    print()
    print("=" * 60)
    print("CHANNEL ROLES:", "all checks passed" if not FAILURES
          else f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
    print("=" * 60)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
