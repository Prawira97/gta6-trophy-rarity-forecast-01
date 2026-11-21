# GTA VI Trophy Rarity Forecaster

Forecasts, for each of the game's 55 PlayStation 5 trophies, five quantiles
(5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at the case's target
(2026-11-26 00:00 UTC for the final), from the house's reads of the
trophy page in the case's history.

## How it works

Exophase refreshes its counts in steps, so the history is collapsed to its
distinct states first: long flat runs are one stale page, not many reads.

The median is an equal-weight ensemble of three views, each capped to a
move of at most 1 logit from the last rate:
- **persistence** — the last rate;
- **a saturating-exponential fit** of the rate against time since launch,
  which bends toward an asymptote instead of running away like a line;
- **an earners/players decomposition** — saturating growth fitted
  separately to the players tracked and to the trophy's earners
  (rate × players), then divided at the target. This is the member that
  expresses dilution: rates overshooting their eventual share while
  earners grow faster than the player base, then settling back to it.

The spread is measured from the case itself: half the ensemble's own move
plus half the trophy's median refresh jump plus a small random-walk term
(floor 0.15, cap 1.2 logit). With a flat history (the practice cases) the
median is the last rate and the spread is a wide prior (1.0 logit), since
the practice refresh moved rates by ~1.5 logit.

## Local verification

There is no local grade for program/v1 kits (the scorer is server-only),
so the entry was verified against synthetic final-like worlds: 2.5 days
of 10-minute reads with step refreshes, scored with the exact pinball
loss on the logit scale. Over 24 worlds at each refresh interval it beat
the house persistence baseline in 24/24, 24/24, 24/24 and 22/24 worlds
(2 h, 5 h, 8 h, 12 h refresh intervals) and the house curve baseline
(log-time line) in 24/24, 24/24, 24/24 and 22/24 — mean loss ~0.36–0.40
vs ~0.48–0.51 for persistence at 2–8 h intervals. The log-time line
loses badly everywhere, which is why this entry does not use one.

Practice cases (flat history): valid output, 55 rows, deterministic
across runs, 0.09 s per run.

## Run

```sh
python3 main.py   # reads /input/*.in (or $MAKEFUN_INPUT_DIR), writes /output/*.out
```

Standard library only. No network, no randomness, no model calls.
