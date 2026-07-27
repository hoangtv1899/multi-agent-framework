#!/usr/bin/env python3
"""
Which years can actually be simulated
src/core/forcing_availability.py

The forcing window is the ONE hard constraint on a request. Without forcing
there is no run at all; every other kind of data gap only costs you a
comparison afterwards. So this is the first thing Reception states, and it is
the thing a user steers by.

It is therefore READ FROM DISK, never asserted. The Reception prompt used to
carry the window as prose ("NLDAS-2 available 1980-2018") and that sentence had
drifted from the filesystem in both directions:

  * it stopped at 2018 while Compy holds complete years through 2023, so five
    perfectly runnable years were being refused;
  * it offered a Qian T62 fallback for non-CONUS domains, but elm_wrapper pins
    DATM_MODE=CLMMOSARTTEST, so no run can ever use Qian. Reception was
    promising forcing the code cannot produce.

A directory listing cannot drift. `scan_nldas()` counts the months that exist;
everything above it is arithmetic on that count.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

# DATM_MODE=CLMMOSARTTEST reads $DIN_LOC_ROOT/atm/datm7/NLDAS/clmforc.nldas.%ym.nc
# — a single stream, one file per month. This is the only forcing the framework
# can drive a run with; see elm_wrapper.FIXED_XML.
DEFAULT_DIN_LOC_ROOT = "/compyfs/inputdata"
NLDAS_SUBPATH        = "atm/datm7/NLDAS"
NLDAS_FILE           = re.compile(r"^clmforc\.nldas\.(\d{4})-(\d{2})\.nc$")
DATASET              = "NLDAS-2"
RESOLUTION           = "0.125 deg (~12 km)"
DATM_MODE            = "CLMMOSARTTEST"
MONTHS_PER_YEAR      = 12


def nldas_dir(root: Optional[str] = None) -> Path:
	"""The directory DATM actually reads. $DIN_LOC_ROOT wins if it is set."""
	root = root or os.environ.get("DIN_LOC_ROOT") or DEFAULT_DIN_LOC_ROOT
	return Path(root) / NLDAS_SUBPATH


def month_counts(directory) -> Dict[int, int]:
	"""{year: months present}. Filenames that do not match the pattern are
	ignored — the tree carries a couple of strays (clmforc.nldas.0016-06.nc)
	that would otherwise invent a year 16 AD."""
	counts: Dict[int, int] = {}
	directory = Path(directory)
	if not directory.is_dir():
		return counts
	for name in os.listdir(directory):
		m = NLDAS_FILE.match(name)
		if m:
			counts[int(m.group(1))] = counts.get(int(m.group(1)), 0) + 1
	return counts


def contiguous_span(years) -> List[int]:
	"""Longest run of consecutive years. Pure arithmetic, no filesystem.

	A run needs an UNBROKEN window: a gap year mid-record would abort the
	simulation partway rather than at submit time, so the longest gap-free run
	is what may be offered, not first..last.
	"""
	ys = sorted(set(int(y) for y in years))
	if not ys:
		return []
	best = run = [ys[0]]
	for y in ys[1:]:
		run = run + [y] if y == run[-1] + 1 else [y]
		if len(run) > len(best):
			best = run
	return best


def scan_nldas(root: Optional[str] = None) -> Dict:
	"""What the forcing tree holds. The only function here that touches disk."""
	d        = nldas_dir(root)
	counts   = month_counts(d)
	complete = [y for y, n in counts.items() if n >= MONTHS_PER_YEAR]
	span     = contiguous_span(complete)
	partial  = {y: n for y, n in counts.items() if n < MONTHS_PER_YEAR}
	return {
		"dataset":      DATASET,
		"resolution":   RESOLUTION,
		"datm_mode":    DATM_MODE,
		"path":         str(d),
		"exists":       d.is_dir(),
		"n_files":      sum(counts.values()),
		"yr_first":     span[0] if span else None,
		"yr_last":      span[-1] if span else None,
		"n_years":      len(span),
		"partial_years": dict(sorted(partial.items())),
		"excluded":     sorted(set(complete) - set(span)),
	}


def render_forcing_facts(root: Optional[str] = None) -> str:
	"""The FORCING block substituted into the Reception prompt.

	When the tree is missing this says so plainly instead of naming a window.
	Reception must not offer years it has not seen — an invented window is the
	one error here that wastes a whole queue slot.
	"""
	w = scan_nldas(root)
	if not w["exists"] or not w["yr_first"]:
		return (f"  FORCING WINDOW UNKNOWN — {w['path']} is not readable from\n"
		        f"  this machine. Do NOT state a year range. Ask the user which\n"
		        f"  period they want and record in run_settings.conflicts that\n"
		        f"  forcing availability could not be verified.")

	lines = [
		f"  • {w['dataset']}, {w['resolution']} — the ONLY forcing this",
		f"    framework can run (DATM_MODE={w['datm_mode']} is fixed in code).",
		f"    RUNNABLE YEARS: {w['yr_first']}-{w['yr_last']} "
		f"({w['n_years']} complete years, verified on disk just now).",
		f"    There is no other dataset and no fallback. A non-CONUS domain",
		f"    cannot be forced at all — say so rather than offering an",
		f"    alternative.",
	]
	if w["partial_years"]:
		ys = ", ".join(f"{y} ({n}/12 months)" for y, n in
		               list(w["partial_years"].items())[-3:])
		lines.append(f"    Incomplete, NOT runnable: {ys}.")
	if w["excluded"]:
		lines.append(f"    Complete but outside the unbroken span, so not "
		             f"offered: {w['excluded']}.")
	return "\n".join(lines)


def clamp(yr_start: int, yr_end: int, root: Optional[str] = None):
	"""Trim a requested period to the runnable window.

	Returns (start, end, note) — note is None when nothing was changed, else a
	sentence for run_settings.conflicts. Returns (None, None, note) when the
	request lies wholly outside the window, which is a real refusal rather than
	a silent shift to some other decade.
	"""
	w = scan_nldas(root)
	lo, hi = w["yr_first"], w["yr_last"]
	if lo is None:
		return yr_start, yr_end, f"forcing availability unverified ({w['path']} unreadable)"
	if yr_end < lo or yr_start > hi:
		return None, None, (f"requested {yr_start}-{yr_end} lies entirely outside "
		                    f"the {w['dataset']} window {lo}-{hi}; cannot run")
	s, e = max(yr_start, lo), min(yr_end, hi)
	if (s, e) == (yr_start, yr_end):
		return s, e, None
	return s, e, (f"requested {yr_start}-{yr_end} clamped to {s}-{e} — "
	              f"{w['dataset']} forcing covers {lo}-{hi}")


if __name__ == "__main__":
	import json
	import sys
	print(json.dumps(scan_nldas(), indent=2))
	print()
	print(render_forcing_facts())
	sys.exit(0)
