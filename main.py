"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's 55 PS5 trophies, five quantiles
(5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at the case's target,
from the house's reads of the trophy page in the case's history.

Model (per trophy, in logit space). Exophase refreshes its counts in
steps, so the history is collapsed to its distinct states first: long
flat runs are one stale page, not many independent reads.

The median is an equal-weight ensemble of three views, each capped to a
move of at most 1 logit from the last rate (a 4.5-day horizon cannot be
trusted further than that, and the house curve baseline's log-time line
overshoots exactly by extrapolating launch-night growth):
  * persistence: the last rate;
  * a saturating-exponential fit of the rate against time since launch,
    which bends toward an asymptote instead of running away like a line;
  * an earners/players decomposition: saturating growth fitted separately
    to the players tracked and to the trophy's earners (rate x players),
    then divided at the target. This is the only member that can express
    the dilution shape — rates overshooting their eventual share while
    earners grow faster than the player base, then settling back to it.

The spread is measured from the case itself, not assumed: half the
ensemble's own move (trend uncertainty) plus half the trophy's median
refresh jump (one snapshot's staleness) plus a small random-walk term,
floored at 0.15 and capped at 1.2 logit. With a flat history (the
practice cases: one stale page, no movement) there is nothing to fit, so
the median is the last rate and the spread is a wide prior — the
practice refresh moved rates by ~1.5 logit, and a flat page gives no
evidence about where the refreshed rates land.

Backtested on synthetic final-like worlds (2.5 days of 10-minute reads,
refresh intervals 2–12 h, 24 worlds each): the ensemble beats
persistence by ~4–6% in pinball loss at 2–8 h refresh intervals and ties
it at 12 h, while the log-time line the house curve baseline uses loses
badly everywhere. Standard library only, deterministic (no randomness,
no dict-order dependence), well inside 8 s / 1 CPU / 1024 MB.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch epoch the house curve baseline measures hours from.
LAUNCH_EPOCH = "2026-11-19T00:00:00Z"

# Random-walk spread term, logit per sqrt-day (small: the measured terms
# dominate, and the backtest says the starter's 0.5 is far too wide).
RW_SIGMA = 0.2
# Widest logit move any ensemble member may make from the last rate.
MAX_MOVE = 1.0
# Floor / cap on the total spread (logit).
SIGMA_FLOOR = 0.15
SIGMA_CAP = 1.2
# Spread used when the history is flat: the practice refresh moved rates
# by ~1.5 logit, so a flat page earns a wide prior, not a narrow band.
FLAT_SIGMA = 1.0


def parse_time(s):
    """ISO-8601 UTC timestamp -> seconds since the Unix epoch."""
    import datetime

    dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt.timestamp()


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def distinct_states(history, idx):
    """Collapse consecutive reads with an identical rate for trophy idx.

    Returns [(hours_since_launch, rate)] with one entry per distinct
    page state, oldest first. Null rates are skipped.
    """
    out = []
    last = None
    for read in history:
        rates = read.get("rates") or []
        if idx >= len(rates):
            continue
        r = rates[idx]
        if r is None:
            continue
        if last is not None and r == last:
            continue
        last = r
        t = (parse_time(read["at"]) - parse_time(LAUNCH_EPOCH)) / 3600.0
        out.append((t, float(r)))
    return out


def sat_fit(ts, ys, ws):
    """Saturating-exponential fit: y(t) = Linf - D * exp(-k * (t - t0)).

    k is profiled over a grid; given k the model is linear in (Linf, D).
    Returns (Linf, D, k, sse, resid_std) of the best fit, or None.
    """
    t0 = ts[0]
    best = None
    k = 0.002
    while k <= 0.5:
        us = [math.exp(-k * (t - t0)) for t in ts]
        # Skip k whose basis barely varies over the history: the fit is
        # then ill-conditioned and its asymptote is meaningless.
        if max(us) - min(us) < 0.05:
            k *= 1.25
            continue
        sw = sum(ws)
        um = sum(w * u for w, u in zip(ws, us)) / sw
        ym = sum(w * y for w, y in zip(ws, ys)) / sw
        suu = sum(w * (u - um) ** 2 for w, u in zip(ws, us))
        if suu <= 1e-12:
            k *= 1.25
            continue
        suy = sum(w * (u - um) * (y - ym) for w, u, y in zip(ws, us, ys))
        coef = suy / suu  # y = Linf + coef * u, coef = -D
        linf = ym - coef * um
        sse = sum(w * (y - (linf + coef * u)) ** 2 for w, u, y in zip(ws, us, ys))
        if best is None or sse < best[3]:
            best = (linf, -coef, k, sse)
        k *= 1.25
    if best is None:
        return None
    linf, d, k, sse = best
    resid = math.sqrt(sse / sum(ws))
    return linf, d, k, sse, resid


def decomp_median(history, idx, target_t, last_lg):
    """Earners/players decomposition: fit saturating growth to the players
    tracked and to this trophy's earners separately, then divide at the
    target. Falls back to the last rate when there is too little to fit.
    """
    # Distinct player counts (dedup consecutive equal).
    p_states = []
    last_p = None
    for read in history:
        p = read.get("players_tracked")
        if p is None or p == last_p:
            continue
        last_p = p
        t = (parse_time(read["at"]) - parse_time(LAUNCH_EPOCH)) / 3600.0
        p_states.append((t, float(p)))
    if len(p_states) < 3:
        return last_lg
    # Fit P(t) = Pinf + C * exp(-t/tau) over a tau grid (linear in Pinf, C).
    best_p = None
    tau = 6.0
    while tau <= 240.0:
        us = [math.exp(-max(t, 0.0) / tau) for t, _ in p_states]
        ys = [p for _, p in p_states]
        n = len(ys)
        um, ym = sum(us) / n, sum(ys) / n
        suu = sum((u - um) ** 2 for u in us)
        if suu > 1e-12:
            c = sum((u - um) * (y - ym) for u, y in zip(us, ys)) / suu
            pinf = ym - c * um
            sse = sum((y - (pinf + c * u)) ** 2 for u, y in zip(us, ys))
            if best_p is None or sse < best_p[0]:
                best_p = (sse, pinf, c, tau)
        tau *= 1.3
    if best_p is None:
        return last_lg
    _, pinf, c_p, tau_p = best_p
    p_target = pinf + c_p * math.exp(-max(target_t, 0.0) / tau_p)
    if p_target <= 0:
        return last_lg
    # Earners series for this trophy: E = rate * P / 100 at each state.
    e_states = []
    last_e = None
    for read in history:
        rates = read.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        p = read.get("players_tracked")
        if p is None:
            continue
        e = rates[idx] * p / 100.0
        if last_e is not None and e == last_e:
            continue
        last_e = e
        t = (parse_time(read["at"]) - parse_time(LAUNCH_EPOCH)) / 3600.0
        e_states.append((t, e))
    if len(e_states) < 3:
        return last_lg
    # Fit E(t) = Einf * (1 - exp(-(t+2)/tau_e)) over a tau grid.
    best_e = None
    tau = 4.0
    while tau <= 400.0:
        us = [1.0 - math.exp(-(max(t, 0.0) + 2.0) / tau) for t, _ in e_states]
        ys = [e for _, e in e_states]
        suu = sum(u * u for u in us)
        if suu > 1e-12:
            einf = sum(u * y for u, y in zip(us, ys)) / suu
            if einf > 0:
                sse = sum((y - einf * u) ** 2 for u, y in zip(us, ys))
                if best_e is None or sse < best_e[0]:
                    best_e = (sse, einf, tau)
        tau *= 1.3
    if best_e is None:
        return last_lg
    _, einf, tau_e = best_e
    e_target = einf * (1.0 - math.exp(-(max(target_t, 0.0) + 2.0) / tau_e))
    rate_target = 100.0 * e_target / p_target
    med = logit(rate_target)
    return min(last_lg + MAX_MOVE, max(last_lg - MAX_MOVE, med))


def forecast_trophy(history, idx, horizon_days, target_t):
    """Return (median_logit, sigma_logit) for one trophy."""
    states = distinct_states(history, idx)
    if not states:
        # No read ever listed this trophy: centre on 1% with a wide band.
        return logit(1.0), FLAT_SIGMA
    last_rate = states[-1][1]
    last_lg = logit(last_rate)
    if len(states) < 3:
        # Flat history: no trend to fit. Median is the last rate, spread
        # is the wide flat-page prior, growing slowly with the horizon.
        return last_lg, FLAT_SIGMA + 0.3 * math.sqrt(horizon_days)
    ts = [t for t, _ in states]
    ys = [logit(r) for _, r in states]
    t_last = ts[-1]
    ws = [0.5 ** ((t_last - t) / 12.0) for t in ts]
    # Member 1: saturating-exponential fit of the rate.
    med_sat = last_lg
    sat = sat_fit(ts, ys, ws)
    if sat is not None:
        linf, d, k, _sse, _resid = sat
        med_sat = linf - d * math.exp(-k * (target_t - ts[0]))
        med_sat = min(last_lg + MAX_MOVE, max(last_lg - MAX_MOVE, med_sat))
    # Member 2: earners/players decomposition.
    med_dec = decomp_median(history, idx, target_t, last_lg)
    # Ensemble median: saturating fit + decomposition + persistence.
    med = (med_sat + med_dec + last_lg) / 3.0
    # Spread, measured from this trophy's own history: half the ensemble's
    # move plus half its median refresh jump plus a small random-walk term.
    jumps = [abs(ys[j] - ys[j - 1]) for j in range(1, len(ys))]
    jmed = sorted(jumps)[len(jumps) // 2] if jumps else 0.0
    move = abs(med - last_lg)
    sigma = math.sqrt((0.5 * move) ** 2 + (0.5 * jmed) ** 2
                      + (RW_SIGMA * math.sqrt(horizon_days)) ** 2)
    return med, min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target_t = (parse_time(case["target"]) - parse_time(LAUNCH_EPOCH)) / 3600.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        med, sigma = forecast_trophy(history, i, horizon_days, target_t)
        q = [round4(clamp_pct(inv_logit(med + z * sigma))) for z in Z]
        # Keep the row non-decreasing after rounding.
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
