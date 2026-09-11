"""George Sterbinsky's analysis functions. DO NOT EDIT.

SOURCE   : y2026m08d18_in-situ_valence_analysis.ipynb
CELL     : 4 (the "Analysis functions" cell)
SHA256   : 7020ec0a9cd964b1 (first 16 hex of the cell source)
DATE : 2026-09-10

This is a byte-for-byte copy of cell 4 of the notebook, so the oxidation states
the controller acts on are exactly the numbers you would get by running the
notebook by hand.

If George reissues the notebook, replace this file with the new cell 4 and run
`python controller.py --selftest` to see whether any number moved.

The only addition is the import block below, which the notebook gets from its
own cell 2.
"""

from pathlib import Path
from datetime import datetime, timedelta
import re
import warnings

import larch.xafs as xafs
from larch import Group
from larch.io import read_ascii, merge_groups

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# BEGIN VERBATIM COPY OF CELL 4
# ---------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def load_xanes(data_dir, data_prefix, start=1, stop=1, x='mono_energy', 
               y='xmap8_mnka_sum', mon='xmap8_dt_corr_i0', extensions=("", ".csv")):
    xanes_data = []
    for index in range(start, stop+1):
        stem = f"{data_prefix}.{index:04d}"
        for ext in extensions:
            path = Path(data_dir) / f"{stem}{ext}"
            if path.is_file():
                break
        else:
            raise FileNotFoundError(f"{stem}[{'|'.join(extensions)}] not in {data_dir}")
        scan = read_ascii(str(path))

        if isinstance(x, str) and hasattr(scan, x):
            scan.energy = getattr(scan, x)
        if not hasattr(scan, 'energy'):
            raise ValueError(f"{path.name}: no energy column {x!r}; has {scan.array_labels}")

        monitor = getattr(scan, mon, None) if isinstance(mon, str) else mon
        if isinstance(y, str) and hasattr(scan, y):
            signal = getattr(scan, y)
        elif len(scan.array_labels) == 2:   # (energy, mu) export: no monitor to divide by
            signal, monitor = getattr(scan, scan.array_labels[1]), None
        else:
            raise ValueError(f"{path.name}: no column {y!r}; has {scan.array_labels}")
        scan.mu = signal if monitor is None else signal / monitor

        xanes_data.append(scan)
    return xanes_data, merge_groups(xanes_data)
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def make_spectrum(energy, intensity):
    """Return a validated 2 x N floating-point spectrum array."""
    x = np.asarray(energy, dtype=float).reshape(-1)
    y = np.asarray(intensity, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("energy and intensity must be equal-length 1-D arrays")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("spectrum contains non-finite values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("energy must be strictly increasing")
    return np.vstack((x, y))
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def find_edge_points(spec, levels, edge_min=None, edge_max=None, *, strict=False, show=False, ax=None):
    """Find energies for absorption levels using legacy-compatible NumPy interpolation.

    NumPy's interpolation clamps a level to an energy-window boundary when the
    level is outside the observed intensity range. That behavior matches the
    former pyDXAS helper. This function makes the clamp visible through both a
    warning and the returned validity mask. Set strict=True to reject it.
    """
    x, y = make_spectrum(spec[0], spec[1])
    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    mask = np.ones(x.size, dtype=bool)
    if edge_min is not None:
        mask &= x >= edge_min
    if edge_max is not None:
        mask &= x <= edge_max
    x_window = x[mask]
    y_window = y[mask]
    if x_window.size < 2:
        raise ValueError("edge window contains fewer than two data points")

    valid = (levels >= y_window.min()) & (levels <= y_window.max())
    if not valid.all():
        missing = ", ".join(f"{level:g}" for level in levels[~valid])
        message = (
            f"Requested mu level(s) {missing} are outside the edge-window range "
            f"[{y_window.min():.3f}, {y_window.max():.3f}]."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message + " Legacy boundary clamping is being used.", stacklevel=2)

    points = np.interp(levels, y_window, x_window)
    if show:
        ax = plt.gca() if ax is None else ax
        ax.plot(x, y)
        ax.scatter(points, levels, color="black", zorder=3, label="thresholds")
    return points, valid
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def trapezoid_numpy(y, x):
    """Integrate y(x) with the composite trapezoidal rule using NumPy only."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.size != y.size or x.size < 2:
        raise ValueError("integration arrays must have equal length >= 2")
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)))
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def white_line_max_index(norm, level=1.0):
    """Index of the largest value reached before the spectrum first falls back below `level`.

    The normalized spectrum starts near zero, rises through `level` on the absorption
    edge, peaks at the white line, then decays. This returns the argmax of the first
    excursion above `level`.

    Ties resolve to the lower index. Raises ValueError if `level` is never reached.
    """
    y = np.asarray(norm, dtype=float).reshape(-1)
    if y.size < 2:
        raise ValueError("spectrum must contain at least two points")
    if not np.isfinite(y).all():
        raise ValueError("spectrum contains non-finite values")

    above = y >= level
    if not above.any():
        raise ValueError(f"spectrum never rises to {level:g} (max {y.max():.4f})")

    i_up = int(np.argmax(above))                 # first point at or above `level`
    tail_below = ~above[i_up:]                   # where it drops back under
    i_stop = i_up + int(np.argmax(tail_below)) if tail_below.any() else y.size
    return i_up + int(np.argmax(y[i_up:i_stop]))
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def dau_edge_energy(spec, mu1, mu2, edge_min=None, edge_max=None, *, strict=False,
                    subgrid=True, show=False, ax=None):
    """Calculate the historical integral (Dau) edge energy.

    `spec` is either a 2 x N array or a normalized Larch Group, whose `energy` and `norm` are
    read in its place.

    Left to their defaults the integration window runs from the first measured point to the white
    line, which is where white_line_max_index puts it, and the integral is taken over exactly
    that interval. The standards, the samples and the tape reference all reach this function that
    way, so an edge energy taken from one is comparable with an edge energy taken from another.
    Pass the window explicitly only to depart from that deliberately.

    With subgrid on the trapezoid runs over exactly the interval the method defines,
    [e1, e2], by inserting those two endpoints into the abscissa. Without it the integral is
    truncated at the nearest enclosed grid points and the two partial trapezoids at the ends are
    silently dropped. That quantizes the edge energy in steps of one grid point: when e1 crosses
    a grid point the whole trapezoid beyond it leaves the sum at once, and since the integrand at
    e1 is mu2 - mu1 -- the same quantity the area is divided by -- the step is the grid spacing
    itself, whatever the thresholds are. Here that is 0.3 eV, about three times the scan-to-scan
    reproducibility and 0.08 in oxidation state, which is the same argument derivative_edge_energy
    makes for its own sub-grid refinement.

    The quantization is a step function of anything that moves e1, so it also makes the edge
    energy a sawtooth in the self-absorption parameter C. Pass subgrid=False for the pre-2026-08
    behaviour.

    Returns (edge, (E_mu1, E_mu2), thresholds_bracketed).
    """
    if hasattr(spec, "norm"):                    # a normalized Larch Group, not a 2 x N array
        x, y = make_spectrum(spec.energy, spec.norm)
    else:
        x, y = make_spectrum(spec[0], spec[1])
    edge_min = x[0] if edge_min is None else edge_min
    edge_max = x[white_line_max_index(y)] if edge_max is None else edge_max
    (e1, e2), threshold_valid = find_edge_points(
        (x, y), [mu1, mu2], edge_min, edge_max, strict=strict
    )
    if not e2 > e1:                              # thresholds crossed or coincident: no area
        area = 0.0
    elif subgrid:
        abscissa = np.unique(np.concatenate([x[(x > e1) & (x < e2)], [e1, e2]]))
        area = trapezoid_numpy(mu2 - np.interp(abscissa, x, y), abscissa)
    else:
        integration_mask = (x >= e1) & (x <= e2)
        area = trapezoid_numpy(mu2 - y[integration_mask], x[integration_mask])
    edge = float(e1 + area / (mu2 - mu1))

    if show:
        ax = plt.gca() if ax is None else ax
        ax.plot(x, y)
        ax.scatter([e1, e2], [mu1, mu2], color="black", s=18, zorder=3)
        ax.axvline(edge, color="red", linestyle="-.", linewidth=1, alpha=0.7)
    return edge, (float(e1), float(e2)), bool(threshold_valid.all())
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def linear_calibration(edge_energies, oxidation_states):
    """Return slope, intercept, and Pearson correlation using NumPy only."""
    x = np.asarray(edge_energies, dtype=float)
    y = np.asarray(oxidation_states, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    correlation = np.corrcoef(x, y)[0, 1]
    return float(slope), float(intercept), float(correlation)
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def safe_label(label, keep_suffix=False):
    """Create a portable filename stem from a spectrum label.

    Scan-counter suffixes such as ".0002" look like file extensions to Path.stem
    and are removed by default. Pass keep_suffix=True to retain them.
    """
    path = Path(str(label).strip())
    stem = path.name if keep_suffix else path.stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def derivative_edge_energy(spec, edge_min, edge_max, *, smooth_points=5, subgrid=True,
                           show=False, ax=None):
    """Return the maximum-first-derivative edge energy using NumPy only.

    With subgrid=True (default) the energy is refined off the measurement grid. The first
    derivative peaks where the second derivative crosses zero, so that crossing is
    interpolated linearly between the two samples straddling the peak. Without it the
    result can only be one of the measured energies, and on a 0.3 eV grid that quantization
    is roughly twenty times coarser than the scan-to-scan reproducibility. Pass
    subgrid=False for the raw grid point.

    The crossing is bracketed at the peak rather than searched for across the window: the
    second derivative is zero at every local extremum of the first, including noise wiggles
    and the pre-edge feature, so a search can lock onto the wrong one.

    Sub-grid refinement does not make the peak *choice* more stable. Where two derivative
    maxima are close in height, `smooth_points` can still swing the result by several eV;
    it will simply report the wrong peak more precisely.
    """
    x, y = make_spectrum(spec[0], spec[1])
    mask = (x >= edge_min) & (x <= edge_max)
    x_window = x[mask]
    y_window = y[mask]
    if x_window.size < 3:
        raise ValueError("derivative edge window contains fewer than three points")
    derivative = np.gradient(y_window, x_window)
    if smooth_points > 1:
        smooth_points = int(smooth_points)
        if smooth_points % 2 == 0:
            smooth_points += 1
        pad = smooth_points // 2
        kernel = np.ones(smooth_points, dtype=float) / smooth_points
        derivative = np.convolve(
            np.pad(derivative, (pad, pad), mode="edge"), kernel, mode="valid"
        )
    index = int(np.argmax(derivative))
    edge = float(x_window[index])

    if subgrid and 0 < index < x_window.size - 1:
        second = np.gradient(derivative, x_window)
        low, high = (index, index + 1) if second[index] > 0 else (index - 1, index)
        if second[low] > 0 > second[high]:      # a clean sign change; otherwise keep the grid point
            edge = float(x_window[low] + (x_window[high] - x_window[low])
                         * second[low] / (second[low] - second[high]))

    if show:
        ax = plt.gca() if ax is None else ax
        ax.plot(x_window, y_window, label="spectrum")
        scale = 0.5 / np.max(np.abs(derivative))
        ax.plot(x_window, derivative * scale, alpha=0.7, label="scaled derivative")
        ax.axvline(edge, color="red", linestyle="-.", linewidth=1)
    return edge
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def scan_timestamp(scan):
    """Return the acquisition start time parsed from a 9-BM scan header.

    The first header line reads, e.g.

        # 1-D Scan File created by LabVIEW Control Panel  7/25/2026  7:05:31 PM; Scan time ...

    The date is present, so the 12-hour clock resolves unambiguously across midnight.
    Raises if no timestamp is found rather than returning None: a silently missing time
    would place a scan at the wrong point on a time axis.
    """
    for line in getattr(scan, "header", []):
        match = re.search(
            r"created by .*?\s(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2}:\d{2})\s*([AP]M)", line
        )
        if match:
            date, clock, meridiem = match.groups()
            return datetime.strptime(f"{date} {clock}{meridiem}", "%m/%d/%Y %I:%M:%S%p")
    raise ValueError(f"{getattr(scan, 'filename', '?')}: no creation timestamp in header")
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def scan_duration(scan):
    """Return how long a scan took to acquire, from the header's "Scan time" clause.

    The clause sits on the same first header line as the creation time:

        # 1-D Scan File created by ... 7:05:31 PM; Scan time 0 hrs 5 min 10 sec.

    Needed because the furnace moves up to 50 C during a single 5-minute scan on the ramp
    and the cooldown, so a scan samples a range of temperatures rather than one value.
    """
    for line in getattr(scan, "header", []):
        match = re.search(r"Scan time\s+(\d+)\s*hrs?\s+(\d+)\s*min\s+(\d+)\s*sec", line)
        if match:
            hours, minutes, seconds = (int(value) for value in match.groups())
            return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    raise ValueError(f"{getattr(scan, 'filename', '?')}: no scan duration in header")
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def load_temperature_log(data_dir, pattern="*.temperature_log"):
    """Read every furnace log in `data_dir` into one time-ordered frame.

    Each file carries two comment lines then four whitespace-separated fields: a LabVIEW
    timestamp (seconds since 1904-01-01 UTC), the temperature in C, and the local date and
    time. The local text columns are the ones used here, because the scan headers are stamped
    by the same clock -- joining on them keeps a UTC offset out of the notebook entirely.

    Rows reading exactly 0 C are dropped as logger dropouts, not measurements. The lowest
    genuine reading in this dataset is 15.9 C, so the value is a sentinel; left in, a single
    one puts a full-scale spike through the plotted trace and drags the mean of whichever scan
    happens to span it. The count is reported so a future run full of dropouts is not cleaned
    up silently.
    """
    frames = []
    for path in sorted(Path(data_dir).glob(pattern)):
        frames.append(pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                                  names=["labview_s", "temperature_C", "date", "time", "meridiem"]))
    if not frames:
        raise FileNotFoundError(f"no files matching {pattern!r} in {data_dir}")

    log = pd.concat(frames, ignore_index=True)
    log["timestamp"] = pd.to_datetime(log["date"] + " " + log["time"] + log["meridiem"],
                                      format="%m/%d/%Y %I:%M:%S%p")

    dropouts = log["temperature_C"] == 0.0
    if dropouts.any():
        print(f"temperature log: discarded {dropouts.sum()} of {len(log)} rows reading exactly 0 C "
              f"({log.loc[dropouts, 'timestamp'].min()} to {log.loc[dropouts, 'timestamp'].max()})")

    return (log.loc[~dropouts, ["timestamp", "temperature_C"]]
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True))
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def temperature_window(log, start, duration):
    """Return the (timestamps, temperatures) the furnace log recorded during one scan.

    The whole trace is kept rather than a mean, so whichever statistic a later question needs
    can be taken from it. A scan on the ramp spans tens of degrees, and which end of that
    spread is the right answer depends on what is being asked: the value at the scan start
    matches a point plotted at the scan start, while the mean matches the scan as a whole.

    Returns empty arrays when the log does not cover the window, so an uncovered scan stays
    visibly empty instead of borrowing a reading from elsewhere in the run.

    `log` must be sorted by timestamp, which load_temperature_log guarantees; the slice is
    found by binary search rather than by masking every row for every scan.
    """
    stamps = log["timestamp"].values
    first = np.searchsorted(stamps, np.datetime64(start), side="left")
    last = np.searchsorted(stamps, np.datetime64(start + duration), side="right")
    return stamps[first:last], log["temperature_C"].values[first:last]
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def transmission_mu(scan, numerator="i0", denominator="it"):
    """Return ln(numerator/denominator) for a pair of ion-chamber columns.

    Two absorptions are read this way at 9-BM and both are the same arithmetic on a different
    pair of columns: the sample in transmission, ln(I0/IT), and the reference foil sitting
    downstream of it, ln(IT/IRef).

    Raises rather than returning NaN where a channel is non-positive. A dropped chamber reads
    zero or slightly negative after dark subtraction, and np.log would turn that into a silent
    -inf that a least-squares fit then propagates into every fitted parameter.
    """
    channels = []
    for name in (numerator, denominator):
        if not hasattr(scan, name):
            raise ValueError(f"{getattr(scan, 'filename', '?')}: no column {name!r}; "
                             f"has {scan.array_labels}")
        channels.append(np.asarray(getattr(scan, name), dtype=float))
    num, den = channels
    bad = (num <= 0) | (den <= 0)
    if bad.any():
        raise ValueError(f"{getattr(scan, 'filename', '?')}: {bad.sum()} point(s) with "
                         f"non-positive {numerator!r} or {denominator!r}")
    return np.log(num / den)
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def self_absorption_fit(energy, fluor, tape_energy, tape_mu, *, shift=0.0,
                        shift_range=(-1.0, 1.0), shift_step=0.01,
                        c_max_factor=6.0, c_points=600, refine=200):
    """Fit the self-absorption parameters G and C against a true-absorption reference.

    Fluorescence from a thick sample is damped by self absorption. Writing the undamped
    absorption as mu_f, the saturation relation and the reference it is fitted to are

        mu_f(E) = G / ((C / I_f(E)) - 1)                  (1)
        mu_f(E) = p * I_t(E) + m * E + b                  (2)

    where I_t is the same material measured in transmission and the linear term absorbs the
    difference in background slope between the two measurements. G and p are exactly degenerate
    -- scaling (G, p, m, b) by any alpha leaves both equations unchanged -- so p is fixed at 1
    and G carries the scale. G then cancels again downstream, because pre_edge normalization
    divides it out; only C changes a reported edge energy.

    With C held fixed, equating (1) and (2) is *linear* in (G, m, b), so this is a
    one-dimensional search over C with an exact least-squares solve inside rather than a
    four-parameter nonlinear fit. There is no initial guess to get wrong, and the returned rms
    profile shows how sharply C is determined instead of leaving that to a covariance estimate.

    `shift` is added to `tape_energy` to place the reference on the sample's energy axis. It is
    an input rather than a parameter because the mono position is measurable from the reference
    foil far more precisely than this fit can determine it. Passing shift=None searches
    `shift_range` instead, which fits better and predicts worse: the free parameter absorbs
    lineshape differences between two physically different samples and drags C with it.

    C is the fluorescence level reached at infinite absorption, so C > max(I_f) is required for
    (1) to stay finite and positive. The search starts just above that bound and raises if the
    minimum lands on either end of it, where the reported C would be a search limit rather than
    a measurement.
    """
    energy = np.asarray(energy, dtype=float).reshape(-1)
    fluor = np.asarray(fluor, dtype=float).reshape(-1)
    if energy.size != fluor.size:
        raise ValueError("energy and fluor must be equal-length 1-D arrays")
    f_max = float(fluor.max())
    if not f_max > 0:
        raise ValueError("fluorescence must be positive somewhere")

    def solve(c_value, shift_value):
        """Exact least squares in (G, m, b) at fixed C and shift. Returns (sse, coefficients).

        The design columns are (u, -E, -1) rather than (u, E, 1) so the coefficients come back as
        (G, m, b) with the signs eq. (2) gives them, and callers can substitute them straight
        into eq. (3) without a sign flip in between.
        """
        target = np.interp(energy, np.asarray(tape_energy, dtype=float) + shift_value,
                           np.asarray(tape_mu, dtype=float))
        design = np.column_stack([fluor / (c_value - fluor), -energy, -np.ones_like(energy)])
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        return float(np.sum((design @ coefficients - target) ** 2)), coefficients

    c_grid = f_max * np.linspace(1.02, c_max_factor, c_points)
    shifts = (np.array([float(shift)]) if shift is not None
              else np.arange(shift_range[0], shift_range[1] + shift_step / 2, shift_step))

    best = None
    for shift_value in shifts:
        sse = np.array([solve(c, shift_value)[0] for c in c_grid])
        index = int(np.argmin(sse))
        if best is None or sse[index] < best[0]:
            best = (float(sse[index]), float(c_grid[index]), float(shift_value), sse)
    _, c_best, shift_best, profile = best

    if c_best <= c_grid[0] or c_best >= c_grid[-1]:
        raise ValueError(f"C minimum sits at the search boundary (C = {c_best:.6g}, searched "
                         f"{c_grid[0]:.6g} to {c_grid[-1]:.6g}); widen c_max_factor")

    step = c_grid[1] - c_grid[0]
    low, high = c_best - step, c_best + step
    for _ in range(refine):                      # golden-section refine off the grid
        trial_low = low + 0.382 * (high - low)
        trial_high = low + 0.618 * (high - low)
        if solve(trial_low, shift_best)[0] < solve(trial_high, shift_best)[0]:
            high = trial_high
        else:
            low = trial_low
    c_best = 0.5 * (low + high)
    sse, (g_best, m_best, b_best) = solve(c_best, shift_best)

    return {
        "C": float(c_best), "G": float(g_best), "m": float(m_best), "b": float(b_best),
        "shift": float(shift_best), "rms": float(np.sqrt(sse / energy.size)),
        "f_max": f_max, "c_grid": c_grid, "rms_profile": np.sqrt(profile / energy.size),
    }
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def apply_self_absorption(fluor, C, G):
    """Undo self-absorption damping: mu_f = G * I_f / (C - I_f), which is eq. (1) rearranged.

    C is the fluorescence level the signal would reach at infinite absorption, so a point at or
    above it is outside the model rather than merely poorly fitted. Raising here is the
    difference between a run that stops and one that silently flips sign through the divide and
    reports a physically impossible edge energy several eV away.
    """
    fluor = np.asarray(fluor, dtype=float)
    f_max = float(fluor.max())
    if not C > f_max:
        raise ValueError(f"C = {C:.6g} must exceed max(I_f) = {f_max:.6g}")
    return G * fluor / (C - fluor)
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def normalized_group(energy, mu, npre=1, pre1=-145, pre2=-50, nnorm=2, norm1=75.0, norm2=721.58, nvict=0):
    """Return a Larch Group carrying `mu` normalized with the notebook's one pre_edge setting.

    Standards, samples, and the self-absorption reference all pass through here, so an edge
    energy taken from one is comparable with an edge energy taken from another.
    """
    group = Group(energy=np.asarray(energy, dtype=float).copy(),
                  mu=np.asarray(mu, dtype=float).copy())
    xafs.pre_edge(group, npre=npre, pre1=pre1, pre2=pre2, nnorm=nnorm, norm1=norm1, norm2=norm2, nvict=nvict)
    return group
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
def self_absorption_anchor_fit(anchors, slope, intercept, mu1, mu2, edge_min=None, edge_max=None,
                               *, strict=False, subgrid=True, c_bounds=(1.02, 8.0),
                               c_points=200, refine=80, npre=1, pre1=-145, pre2=-50,
                               nnorm=2, norm1=75.0, norm2=721.58, nvict=0):
    """Choose the self-absorption parameter C so known oxidation states come back correct.

    The alternative to fitting C against a transmission reference. Where a scan's oxidation
    state is known independently -- from the stoichiometry of the starting material, say -- C
    can be set by demanding that the integral method plus the standards calibration return that
    number. `anchors` is a sequence of (energy, fluor, target_valence); with one anchor this is
    a root find, with several it is a least-squares compromise across them.

    G plays no part. Eq. (1) scales linearly with it and `pre_edge` normalization divides it out
    again, so it is fixed at 1 here and the corrected mu carries arbitrary units.

    Apparent oxidation state falls monotonically with C -- larger C means a flatter correction,
    in the limit no correction at all -- so the minimum is unique. That holds only with subgrid
    on: truncating the integral at grid points quantizes the edge energy in steps of one grid
    point, 0.3 eV here, and the sawtooth that produces gives this search three roots within 0.08
    in valence of each other. Elsewhere subgrid=False is a comparison setting; here it breaks the
    search, and it is forwarded from this signature rather than read from dau_edge_energy's.

    mu1, mu2 and everything after them are handed straight to `dau_edge_energy`. Leave the window
    at its default: the white line moves as C changes, so it has to be re-derived at every trial
    rather than fixed once from the uncorrected spectrum.

    Grid then golden-section, the same shape as `self_absorption_fit`, and the grid is kept in
    the return value because the profile of valence against C shows how sharply the anchor
    determines it far better than a curvature at the minimum does.
    """
    anchors = [(np.asarray(energy, dtype=float).reshape(-1),
                np.asarray(fluor, dtype=float).reshape(-1), float(target))
               for energy, fluor, target in anchors]
    if not anchors:
        raise ValueError("at least one anchor is required")
    for energy, fluor, _ in anchors:
        if energy.size != fluor.size:
            raise ValueError("energy and fluor must be equal-length 1-D arrays")
    f_max = max(float(fluor.max()) for _, fluor, _ in anchors)
    if not f_max > 0:
        raise ValueError("fluorescence must be positive somewhere")
    targets = np.array([target for _, _, target in anchors])

    def valences(c_value):
        """Apparent oxidation state of every anchor at this C, or NaN where the model breaks."""
        out = np.full(targets.size, np.nan)
        for index, (energy, fluor, _) in enumerate(anchors):
            try:
                group = normalized_group(energy, apply_self_absorption(fluor, c_value, 1.0),
                                         npre, pre1, pre2, nnorm, norm1, norm2, nvict)
                out[index] = intercept + slope * dau_edge_energy(
                    group, mu1, mu2, edge_min, edge_max, strict=strict, subgrid=subgrid)[0]
            except (ValueError, IndexError, ZeroDivisionError):
                pass                      # a C the model cannot reach scores as inf below
        return out

    def score(c_value):
        row = valences(c_value)
        return np.inf if not np.isfinite(row).all() else float(np.sum((row - targets) ** 2))

    c_grid = f_max * np.linspace(c_bounds[0], c_bounds[1], c_points)
    profiles = np.array([valences(c_value) for c_value in c_grid])
    scores = np.where(np.isfinite(profiles).all(axis=1),
                      np.sum((profiles - targets) ** 2, axis=1), np.inf)
    if not np.isfinite(scores).any():
        raise ValueError("no C in the search range gives a usable spectrum")
    index = int(np.argmin(scores))
    if index in (0, c_grid.size - 1):
        raise ValueError(f"the anchor minimum sits at a search bound (C = {c_grid[index]:.6g}, "
                         f"searched {c_grid[0]:.6g} to {c_grid[-1]:.6g}); widen c_bounds")

    low, high = c_grid[index - 1], c_grid[index + 1]
    for _ in range(refine):                      # golden-section refine off the grid
        trial_low = low + 0.382 * (high - low)
        trial_high = low + 0.618 * (high - low)
        if score(trial_low) < score(trial_high):
            high = trial_high
        else:
            low = trial_low
    c_best = 0.5 * (low + high)
    achieved = valences(c_best)

    # How hard the answer leans on the assumed oxidation state. dV/dC is negative, so a valence
    # assumed 0.05 too high pulls C down by 0.05 * |dC/dvalence|.
    step = 1e-4 * c_best
    slope_vc = float(np.mean((valences(c_best + step) - valences(c_best - step)) / (2 * step)))

    return {
        "C": float(c_best), "achieved": achieved, "targets": targets,
        "residuals": achieved - targets,
        "rms_valence": float(np.sqrt(np.mean((achieved - targets) ** 2))),
        "dC_dvalence": float(1.0 / slope_vc) if slope_vc else np.nan,
        "f_max": f_max, "c_grid": c_grid, "valence_profiles": profiles,
    }
#---------------------------------------------------------------------------------------------


# --- END VERBATIM COPY OF CELL 4 ---
