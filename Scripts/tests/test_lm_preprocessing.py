"""LMPreprocessor: weak-fluorescence enhancement in the standalone tool.

scipy may not be installed where this runs, so a numpy implementation of the
two ndimage functions the pipeline uses is injected. The checks are therefore
about behaviour -- what the pipeline does to known synthetic data -- not about
matching scipy bit for bit.

Run with:  python tests/test_lm_preprocessing.py
"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)


# --------------------------------------------------------------------------- #
# numpy stand-ins for scipy.ndimage
# --------------------------------------------------------------------------- #

def _gaussian_filter(img, sigma, mode="nearest", truncate=4.0):
    if sigma <= 0:
        return img.astype(np.float64, copy=True)
    r = int(truncate * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()
    out = img.astype(np.float64, copy=True)
    for axis in (0, 1):                       # separable
        out = np.apply_along_axis(
            lambda v: np.convolve(np.pad(v, r, mode="edge"), k, mode="valid"),
            axis, out)
    return out


def _median_filter(img, size=3, mode="nearest"):
    r = size // 2
    pad = np.pad(img, r, mode="edge")
    H, W = img.shape
    win = np.empty((size * size, H, W), dtype=img.dtype)
    n = 0
    for dy in range(size):
        for dx in range(size):
            win[n] = pad[dy:dy + H, dx:dx + W]
            n += 1
    return np.median(win, axis=0)


ndimage = types.ModuleType("scipy.ndimage")
ndimage.gaussian_filter = _gaussian_filter
ndimage.median_filter = _median_filter
scipy_mod = types.ModuleType("scipy")
scipy_mod.ndimage = ndimage
sys.modules.setdefault("scipy", scipy_mod)
sys.modules.setdefault("scipy.ndimage", ndimage)

# The tool imports GUI/IO packages at module scope; stub what's missing.
for name in ("mrcfile", "tifffile", "skimage", "skimage.transform",
             "tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox",
             "matplotlib", "matplotlib.pyplot", "matplotlib.figure",
             "matplotlib.patheffects", "matplotlib.backends",
             "matplotlib.backends.backend_tkagg"):
    sys.modules.setdefault(name, types.ModuleType(name))


def _load_preprocessor():
    """Exec just the preprocessing block, avoiding the GUI import chain."""
    src = open(os.path.join(SCRIPTS, "clem_correlation_tool_standalone.py")).read()
    start = src.index("@dataclass\nclass LMPreprocessSettings:")
    end = src.index("def make_record(")
    from dataclasses import dataclass, field
    from typing import Any, Optional
    ns = {"np": np, "dataclass": dataclass, "field": field,
          "Any": Any, "Optional": Optional}
    exec(compile(src[start:end], "<lm preprocessing>", "exec"), ns)
    return ns["LMPreprocessSettings"], ns["LMPreprocessor"]


LMPreprocessSettings, LMPreprocessor = _load_preprocessor()
FAILURES = []


def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
        return
    FAILURES.append(label)
    print(f"  FAIL  {label}\n        got  {got!r}\n        want {want!r}")


def check_true(label, cond, detail=""):
    check(label + (f"  [{detail}]" if detail and not cond else ""), bool(cond), True)


# --------------------------------------------------------------------------- #
# synthetic data
# --------------------------------------------------------------------------- #

# The image must be comfortably larger than the background kernel (which
# reaches ~4 sigma), otherwise the background estimate degenerates towards a
# constant. A real 2k camera frame at sigma 35 is fine; this smaller fixture
# uses a proportionally smaller sigma.
Z, H, W = 5, 256, 288
BG_SIGMA = 12.0
SPOT = (150, 190)            # (y, x) of the weak fluorescent spot
HOT = (40, 60)               # persistent detector defect
TRANSIENT = (200, 240)       # a real, bright, single-plane event


def make_stack(seed=0):
    """(C, Z, Y, X): ch0 reflection, ch1 weak fluorescence."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]

    # slowly varying background, a broad ramp plus a smooth blob
    bg = (0.30 + 0.0009 * xx + 0.0007 * yy
          + 0.10 * np.exp(-((yy - 50) ** 2 + (xx - 240) ** 2) / (2 * 40.0 ** 2)))

    # weak spot, sigma 2 px, only a little above the noise
    spot = 0.045 * np.exp(-(((yy - SPOT[0]) ** 2 + (xx - SPOT[1]) ** 2)
                            / (2 * 2.0 ** 2)))

    fluo = np.empty((Z, H, W), np.float32)
    for z in range(Z):
        fluo[z] = bg + spot + rng.normal(0, 0.004, (H, W))
    fluo[:, HOT[0], HOT[1]] = 3.0                  # every plane -> defect
    fluo[2, TRANSIENT[0], TRANSIENT[1]] = 3.0      # one plane -> real event

    refl = np.stack([rng.random((H, W)).astype(np.float32) for _ in range(Z)])
    refl[:, HOT[0], HOT[1]] = 3.0                  # same defect, must survive

    return np.stack([refl, fluo]).astype(np.float32)


def main():
    stack = make_stack()
    settings = LMPreprocessSettings(enabled=True, background_sigma=BG_SIGMA)
    pre = LMPreprocessor(settings)

    print("\n--- bad pixels: persistent removed, one-off kept ---------------")
    fluo = stack[1]
    mask = pre.find_bad_pixels(fluo)
    check("persistent hot pixel flagged", bool(mask[HOT]), True)
    check("single-plane event not flagged", bool(mask[TRANSIENT]), False)
    check("the weak spot is not flagged", bool(mask[SPOT]), False)
    check("hardly anything else flagged", int(mask.sum()) <= 2, True)

    corrected = pre.correct_bad_pixels(fluo, mask)
    check("hot pixel replaced by a sane value", bool(corrected[0][HOT] < 1.0), True)
    check("one-off event survives correction",
          float(corrected[2][TRANSIENT]), 3.0)
    untouched = np.ones((H, W), bool); untouched[HOT] = False
    check("no other pixel altered",
          float(np.abs(corrected - fluo)[:, untouched].max()), 0.0)

    print("\n--- background subtraction -------------------------------------")
    sm = pre.presmooth(corrected)
    bgsub = pre.subtract_background(sm)
    corner_before = float(np.median(sm[:, :20, :20]) - np.median(sm[:, -20:, -20:]))
    corner_after = float(np.median(bgsub[:, :20, :20]) - np.median(bgsub[:, -20:, -20:]))
    check("corner-to-corner ramp suppressed",
          abs(corner_after) < 0.1 * abs(corner_before), True)
    check("signed values kept (not clipped at zero)", bool((bgsub < 0).any()), True)

    print("\n--- matched filter ---------------------------------------------")
    matched = pre.apply_matched_filter(bgsub)

    def snr(arr):
        # Noise patch taken well inside the frame: the corners carry
        # background-subtraction edge residuals, not noise, and a low-frequency
        # residual is not what a matched filter is meant to suppress.
        z = arr[0]
        peak = z[SPOT[0] - 1:SPOT[0] + 2, SPOT[1] - 1:SPOT[1] + 2].max()
        far = z[100:140, 40:80]
        return float((peak - far.mean()) / (far.std() + 1e-12))

    snr_raw, snr_out = snr(bgsub), snr(matched)
    check("matched filter raises SNR at the spot", snr_out > snr_raw, True)
    print(f"        SNR {snr_raw:.1f} -> {snr_out:.1f}")

    peak = np.unravel_index(np.argmax(matched[0]), matched[0].shape)
    check("peak stays within 1 px of the true spot",
          max(abs(peak[0] - SPOT[0]), abs(peak[1] - SPOT[1])) <= 1, True)

    print("\n--- reflection channel is never touched ------------------------")
    roles = ["reflection", "red"]
    res = pre.process_stack(stack, roles=roles)
    check("only the fluorescence channel processed", res["processed"], [False, True])
    check("reflection returned unchanged",
          float(np.abs(res["matched"][0] - stack[0]).max()), 0.0)
    check("reflection keeps its own defect (not repaired)",
          float(res["matched"][0][0][HOT]), 3.0)
    check("fluorescence channel did change",
          bool(np.abs(res["matched"][1] - stack[1]).max() > 0), True)

    print("\n--- channels with no role are treated as fluorescence ----------")
    res_norole = pre.process_stack(stack, roles=[])
    check("both processed when no roles given", res_norole["processed"], [True, True])

    print("\n--- shapes and the two display modes ---------------------------")
    check("matched keeps (C, Z, Y, X)", res["matched"].shape, (2, Z, H, W))
    check("maxproj is (C, Y, X)", res["maxproj"].shape, (2, H, W))
    m_stack = LMPreprocessor.as_stack(res, "matched")
    p_stack = LMPreprocessor.as_stack(res, "maxproj")
    check("matched mode is the per-plane response", m_stack.shape, (2, Z, H, W))
    check("maxproj mode keeps the stack shape", p_stack.shape, (2, Z, H, W))
    check("maxproj repeats one image across Z",
          float(np.abs(p_stack[1, 0] - p_stack[1, -1]).max()), 0.0)
    check("maxproj equals the Z maximum",
          float(np.abs(p_stack[1, 0] - res["matched"][1].max(axis=0)).max()), 0.0)
    check("maxproj >= any single plane",
          bool((p_stack[1, 0] + 1e-6 >= res["matched"][1]).all()), True)

    print("\n--- steps can be switched off ----------------------------------")
    off = LMPreprocessor(LMPreprocessSettings(
        enabled=True, background_sigma=BG_SIGMA, remove_bad_pixels=False,
        pre_smooth=False, subtract_background=False, matched_filter=False))
    passthrough = off.process_channel(stack[1])
    check("all steps off is a pass-through",
          float(np.abs(passthrough["matched"] - stack[1]).max()), 0.0)

    bgonly = LMPreprocessor(LMPreprocessSettings(
        enabled=True, background_sigma=BG_SIGMA, remove_bad_pixels=False,
        pre_smooth=False, matched_filter=False))
    r = bgonly.process_channel(stack[1])
    check("background-only run leaves matched == bgsub",
          float(np.abs(r["matched"] - r["bgsub"]).max()), 0.0)

    print("\n--- parameters actually take effect ----------------------------")
    wide = LMPreprocessor(LMPreprocessSettings(enabled=True, background_sigma=BG_SIGMA, matched_filter_sigma=6.0))
    narrow = LMPreprocessor(LMPreprocessSettings(enabled=True, background_sigma=BG_SIGMA, matched_filter_sigma=1.0))
    w = wide.process_channel(stack[1])["matched"][0]
    n = narrow.process_channel(stack[1])["matched"][0]
    check("a wider matched filter gives a flatter peak",
          float(n.max()) > float(w.max()), True)

    print()
    print("=" * 64)
    print("LM PREPROCESSING:", "all checks passed" if not FAILURES
          else f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
    print("=" * 64)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
