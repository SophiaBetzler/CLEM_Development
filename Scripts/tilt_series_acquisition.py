



def run_grouped_dose_symmetric_tilt_series(
    track_state="TS_Track",
    focus_state="TS_Focus",
    record_state="TS_Record",
    start_tilt=15,
    start_defocus=-3.0,
    step=3,
    max_tilt=60,
    group=4,
    tilt_backlash=-1,
    wait=3,
):
    import serialem as sem

    state = {
                "stage_x": None,
                "stage_y": None,
                "plus_tilt": start_tilt,
                "minus_tilt": start_tilt,
                "focus_plus": start_defocus,
                "focus_minus": start_defocus,
                "isx_plus": 0,
                "isy_plus": 0,
                "isx_minus": 0,
                "isy_minus": 0,
            }

    def echo(msg):
        sem.Echo(str(msg))

    def first(vals):
        return vals[0] if isinstance(vals, tuple) else vals

    def check_auto_fill():
        for _ in range(10):
            filling = first(sem.AreDewarsFilling())
            if int(filling) == 0:
                return
            echo("dewars are filling")
            sem.Delay(60, "sec")

    def report_is_xy():
        vals = sem.ReportImageShift()
        if isinstance(vals[0], tuple):
            return float(vals[0][0]), float(vals[0][1])
        return float(vals[0]), float(vals[1])

    def acquire_track(ref_buf=None, save_buf=None):
        check_auto_fill()
        sem.GoToImagingState(track_state)
        sem.T()

        if ref_buf is not None:
            sem.AlignTo(ref_buf)

        if save_buf is not None:
            sem.Copy("A", save_buf)

    def autofocus():
        check_auto_fill()
        sem.GoToImagingState(focus_state)
        sem.G()
        return float(first(sem.ReportDefocus()))

    def acquire_record(ref_buf=None, save_buf=None):
        check_auto_fill()
        sem.GoToImagingState(record_state)
        sem.R()
        sem.S()

        if ref_buf is not None:
            sem.AlignTo(ref_buf)

        if save_buf is not None:
            sem.Copy("A", save_buf)

        return report_is_xy()

    def set_record_prediction(defocus, isx, isy):
        sem.GoToImagingState(record_state)
        sem.SetDefocus(float(defocus))
        sem.SetImageShift(float(isx), float(isy))

    def tilt_zero():
        sem.TiltTo(float(start_tilt))
        sem.Delay(wait, "sec")

        stage = sem.ReportStageXYZ()
        state["stage_x"] = float(stage[0])
        state["stage_y"] = float(stage[1])

        set_record_prediction(
            start_defocus,
            state["isx_plus"],
            state["isy_plus"],
        )

        acquire_track(save_buf="K")
        sem.Copy("A", "L")

        focus = autofocus()
        state["focus_plus"] = focus
        state["focus_minus"] = focus

        isx, isy = acquire_record(save_buf="M")
        sem.Copy("A", "N")

        state["isx_plus"] = isx
        state["isy_plus"] = isy
        state["isx_minus"] = isx
        state["isy_minus"] = isy

        acquire_track(save_buf="K")
        sem.Copy("A", "L")

    def tilt_plus():
        sem.TiltTo(float(state["plus_tilt"]))
        sem.MoveStageTo(state["stage_x"], state["stage_y"])

        set_record_prediction(
            state["focus_plus"],
            state["isx_plus"],
            state["isy_plus"],
        )

        sem.Delay(wait, "sec")

        acquire_track(ref_buf="K")
        state["focus_plus"] = autofocus()

        isx, isy = acquire_record(ref_buf="M", save_buf="M")
        state["isx_plus"] = isx
        state["isy_plus"] = isy

        acquire_track(save_buf="K")

    def tilt_minus():
        sem.TiltTo(float(state["minus_tilt"]))
        sem.TiltBy(float(tilt_backlash))
        sem.TiltTo(float(state["minus_tilt"]))
        sem.MoveStageTo(state["stage_x"], state["stage_y"])

        set_record_prediction(
            state["focus_minus"],
            state["isx_minus"],
            state["isy_minus"],
        )

        sem.Delay(wait, "sec")

        acquire_track(ref_buf="L")
        state["focus_minus"] = autofocus()

        isx, isy = acquire_record(ref_buf="N", save_buf="N")
        state["isx_minus"] = isx
        state["isy_minus"] = isy

        acquire_track(save_buf="L")

    echo("Starting grouped dose-symmetric tilt series with imaging states")

    sem.SetLowDoseMode(0)

    if max_tilt % (group * step) != 0:
        raise RuntimeError("max_tilt must be divisible by group * step")

    branch_steps = int(max_tilt / (group * step))

    tilt_zero()

    for _ in range(branch_steps):
        for _ in range(group):
            state["plus_tilt"] += step
            tilt_plus()

        for _ in range(group):
            state["minus_tilt"] -= step
            tilt_minus()

    sem.TiltTo(0)
    sem.SetDefocus(0)
    sem.ResetImageShift()
    sem.CloseFile()

    echo("Finished grouped dose-symmetric tilt series")