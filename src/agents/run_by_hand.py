#!/usr/bin/env python3
"""
How to make the new runs without the framework: RUN_BY_HAND.md
src/agents/run_by_hand.py

    render(preview, run_dir, strategy, brief) -> the whole page, or ""

For a whole-domain study the model server checks the design and builds every
run before the design page is drawn, and submits nothing
(01_inputs/design_preview.json and the files it names). This page tells a
modeller how to make the same runs by hand from those files. It replaces the
second part that design_review._appendix wrote, which stays for any other
kind of study.

WHAT CHANGED (the user's review of 2026-11-13). The old page had about 2,300
words for one run. Two scripts printed in full took about 800 of them, and a
closing section on the numbers tabulated from the runs belonged to the
analysis and said something false once the comparisons were approved before
the run. The page now follows the draft the user approved, in six steps:

    1. Get the code              the checkout, its commit, the local changes,
                                 each source file copied in, the fingerprint
    2. The control run's case    its case, its output and its recipe
    3. Make the changes          a row per difference from the control run's
                                 recipe, the files written for the run, and
                                 the script that creates the case
    4. Run the jobs              the stage script and a row per stage
    5. Check each job            a row per check, with its tolerance
    6. The input files           the inputs, their check, and the rest

Steps 3 to 5 are given for each run under its name; the others once. A study
of the control run alone makes no run: its page says how the control run was
made, in steps 1, 2 and the input files, and that nothing is run.

THE RULES. Every value comes from the server's records; no sentence carries a
number code did not fill in. A script is named, never printed. File hashes
other than the code's, the text of written files and the settings that only
grow by one entry are left out. A part whose record is missing is left out,
and the page is still written. No language model reads or writes any of it.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PREVIEW_FILE = "01_inputs/design_preview.json"

NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")

# A SETTING'S PLAIN NAME, where the page knows one. A row of the table of
# changes names the setting as the case knows it, after these words.
PLAIN_NAMES = {
    "RUN_STARTDATE": "Start date",
    "CCSM_CO2_PPMV": "CO2",
    "finidat": "Starting state",
    "fsurdat": "Surface data",
    "paramfile": "Parameter file",
    "use_nofire": "Fire",
    "suplnitro": "Nitrogen supplement",
    "suplphos": "Phosphorus supplement",
    "stream_year_first_ndep": "Nitrogen deposition, first year",
    "stream_year_last_ndep": "Nitrogen deposition, last year",
}
COMPONENT_NAMES = {"datm": "Weather"}
TOP_KEYS = (("compset", "Compset"), ("res", "Resolution"), ("mach", "Machine"),
            ("compiler", "Compiler"), ("project", "Project"), ("srcroot", "E3SM checkout"))

# THE OTHER FILES THE MODEL READS, by the name the land namelist gives them or
# the stream that carries them; one not here is named by its key.
FILE_NAMES = {
    "paramfile": "the parameter file",
    "fsnowoptics": "the snow optics file",
    "fsnowaging": "the snow aging file",
    "stream_fldfilename_popdens": "the population density file",
    "stream_fldfilename_ndep": "the nitrogen deposition file",
    "stream_fldfilename_lightng": "the lightning file",
    "fsoilordercon": "the soil order file",
}
STREAM_NAMES = {"presaero": "the aerosol deposition file", "topo": "the topography file"}
WEATHER_FIELDS = {"TBOT", "PRECTmms", "PRECT", "FSDS", "QBOT", "WIND", "PSRF", "FLDS"}

# WHAT EACH STAGE DOES, in the page's words; the stages are the server's
# (domain_case.plan_chain).
STAGE_WORDS = {"case": "creates and builds the case",
               "report": "writes each job's state and time"}

# THE CHECKS IN PLAIN WORDS. The model server describes what each job checks
# (run_gate.CHECKS_IN_WORDS); its words have colons and asides in brackets,
# and say "the control". The page's sentences below say the same in plain
# words, and each is used only while the server's words are the ones it was
# written from (SERVER_WORDS). Words the server has changed since are printed
# as the server gives them, so the page never says what a check does not do.
SERVER_WORDS = {
    "case": "before the model is built, the run's resolved namelists against the "
            "control's: exactly the keys its changes alter differ, with the values they "
            "set; and every file written for the run that the case must hold (a user "
            "stream file) is there unchanged",
    "smoke": "a few days from the starting state: this job's own logs (found by its job "
             "id) end cleanly, and the first day's means match the saved state it starts "
             "from, over vegetated cells (a cold start differs by about 100 %)",
    "year": "this job's own logs show the model dates it ran and a clean end; every "
            "history tape is complete with the fields its namelist asks for; a restart "
            "exists for the next year; no vegetated cell misses a value; GPP, peak LAI, "
            "water closure and the carbon stock lie within sanity bounds",
    "reference": "each year set against the control's same year (per-field bias and "
                 "daily correlation), reported and not held, since the run is meant to "
                 "differ",
}
PLAIN_WORDS = {
    "case": "Before the model is built, the run's settings differ from the control run's "
            "only in the settings its changes alter, with the values they set.",
    "smoke": "The job's own logs end cleanly. The first day's means match the saved state "
             "the run starts from, over the vegetated cells.",
    "year": "The job's own logs show the model dates it ran and a clean end. Every history "
            "file holds the fields its namelist asks for. A saved state exists for the "
            "next year. No vegetated cell misses a value. GPP, peak leaf area, the water "
            "balance and the carbon stock lie within their bounds.",
    "reference": "Each field's bias and daily correlation against the control run's same "
                 "year are reported.",
}


# ─────────────────────────────────────────────────────────────────────────────
# SMALL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _list(x: Any) -> List[Any]:
    return list(x) if isinstance(x, list) else []


def _load(run_dir, rel, as_json: bool = True):
    """A file the preview names, read; None when it cannot be."""
    if run_dir is None or not rel:
        return None
    p = Path(run_dir) / str(rel)
    try:
        if not p.is_file():
            return None
        text = p.read_text()
        return json.loads(text) if as_json else text
    except (OSError, ValueError):
        return None


def _exists(run_dir, rel) -> bool:
    try:
        return run_dir is not None and bool(rel) and (Path(run_dir) / str(rel)).is_file()
    except OSError:
        return False


def _code(x: Any) -> str:
    return f"`{x}`"


def _cell(text: Any) -> str:
    """A table cell: one line, its pipes escaped."""
    return " ".join(str(text if text is not None else "").split()).replace("|", "\\|")


def _table(head: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    if not rows:
        return []
    return (["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            + ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows] + [""])


def _join(xs: Sequence[str]) -> str:
    xs = [x for x in xs if x]
    return ("" if not xs else xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1])


def _number(n: int) -> str:
    """A small count in words, a larger one in digits."""
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else f"{n:,}"


def _count(n: int, one: str, many: str) -> str:
    """"three ways", "one way"."""
    return f"{_number(n)} {one if n == 1 else many}"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _sentence(text: str) -> str:
    t = _cap(text.strip())
    return t if t.endswith((".", "?", "!")) else t + "."


_BARE_CONTROL = re.compile(r"\bthe control\b(?! run)")


def _safe(text: Any) -> str:
    """The server's words as the page prints them: one line, "the control
    run" for "the control", and no dash as punctuation."""
    t = " ".join(str(text or "").split())
    t = t.replace(" \u2014 ", ", ").replace("\u2014", ", ").replace("\u2013", "-")
    return _BARE_CONTROL.sub("the control run", t)


def _rel(path: Any, run_dir) -> str:
    """A path as the page prints it: from the study folder where it lies
    inside it, else whole. A study folder that was moved still holds its own
    files, found by the folder's name in the path."""
    p = str(path)
    if run_dir is None or not p.startswith("/"):
        return p
    rd = Path(run_dir)
    for base in (rd, rd.resolve()):
        try:
            return str(Path(p).relative_to(base))
        except ValueError:
            continue
    mark = f"/{rd.name}/"
    if rd.name and mark in p:
        rest = p.split(mark, 1)[1]
        if (rd / rest).exists():
            return rest
    return p


def wall_words(wall: Any) -> str:
    """A wall time as a person says it: "01:55:00" is "1 h 55 min"."""
    s = str(wall or "").strip()
    m = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", s)
    if not m:
        return s
    d, h, mi, se = (int(x or 0) for x in m.groups())
    bits = [f"{d} d" if d else "", f"{h} h" if h else "", f"{mi} min" if mi else "",
            f"{se} s" if se else ""]
    return " ".join(b for b in bits if b) or "0 min"


def _decimal(x: float) -> str:
    """A tolerance as a plain decimal: 1e-05 is 0.00001."""
    t = f"{float(x):.12f}".rstrip("0").rstrip(".")
    return t if t not in ("", "-0") else "0"


def _month(label: str) -> str:
    m = re.fullmatch(r"(\d{4})-(\d{2})", str(label))
    return f"{MONTHS[int(m.group(2)) - 1]} {m.group(1)}" if m and 1 <= int(m.group(2)) <= 12 \
        else str(label)


def months_words(months: Sequence[str]) -> str:
    """"June, July and August of 2018 and 2019" for the same months of each
    year, else each year's own months."""
    by_year: Dict[str, List[int]] = {}
    other = []
    for x in months or []:
        m = re.fullmatch(r"(\d{4})-(\d{2})", str(x))
        if m and 1 <= int(m.group(2)) <= 12:
            by_year.setdefault(m.group(1), []).append(int(m.group(2)))
        else:
            other.append(str(x))
    if other or not by_year:
        return _join([str(x) for x in months or []])
    sets = {tuple(sorted(set(v))) for v in by_year.values()}
    if len(sets) == 1:
        ms = next(iter(sets))
        return f"{_join([MONTHS[k - 1] for k in ms])} of {_join(sorted(by_year))}"
    return _join([f"{_join([MONTHS[k - 1] for k in sorted(set(v))])} of {y}"
                  for y, v in sorted(by_year.items())])


def _uncomment(line: str) -> str:
    """A namelist line without its comment: from a "!" outside quotes."""
    q = None
    for i, ch in enumerate(line):
        if q:
            q = None if ch == q else q
        elif ch in "'\"":
            q = ch
        elif ch == "!":
            return line[:i]
    return line


_KEY = re.compile(r"^\s*([A-Za-z_]\w*(?:\(\d+\))?)\s*=\s*(.*)$")


def namelist(lines: Any) -> Dict[str, str]:
    """{key: value} of a user_nl file's lines, a continued value joined to
    its key's, comments left out."""
    out: Dict[str, str] = {}
    key = None
    for raw in _list(lines):
        s = " ".join(_uncomment(str(raw)).split())
        if not s:
            continue
        m = _KEY.match(s)
        if m:
            key = m.group(1)
            out[key] = m.group(2).strip()
        elif key:
            out[key] = (out[key] + " " + s).strip()
    return out


def _xml(recipe: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for e in _list(recipe.get("xml")):
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            out[str(e[0])] = str(e[1])
    return out


def _value(v: str) -> str:
    """A namelist value in a cell of its own: one quoted string without its
    quotes, anything else as it is written."""
    m = re.fullmatch(r"'([^']*)'|\"([^\"]*)\"", v.strip())
    return _code(m.group(1) if m.group(1) is not None else m.group(2)) if m else _code(v)


def _plain(value: str) -> str:
    """An XML setting's value: a path in code, a word or number as it is."""
    return _code(value) if "/" in value or " " in value else value


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: THE CODE
# ─────────────────────────────────────────────────────────────────────────────
def _sourcemods(recipe: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return [(str(k), _dict(v)) for k, v in _dict(recipe.get("sourcemods")).items()]


def _code_step(n: int, preview: Dict[str, Any], first: Dict[str, Any],
               runs: bool) -> List[str]:
    src = _dict(preview.get("source"))
    mods = _sourcemods(first)
    if not src and not mods:
        return []
    L = [f"## Step {n}. Get the code", ""]
    if src.get("error"):
        L += ["The model server could not read the E3SM checkout. "
              + _sentence(_safe(src["error"])), ""]
        src = {}
    rows = []
    if src.get("srcroot"):
        rows.append(("E3SM checkout", _code(src["srcroot"])))
    if src.get("commit"):
        rows.append(("Commit", _code(src["commit"])))
    if src:
        changed = [str(c).split(None, 1)[-1] for c in _list(src.get("changed"))]
        left = [str(u) for u in _list(src.get("patch_left_out"))]
        added = [u for u in _list(src.get("untracked")) if str(u) not in left]
        if src.get("patch"):
            what = _join(([f"changes {_count(len(changed), 'path', 'paths')}"] if changed else [])
                         + ([f"adds {_count(len(added), 'new file', 'new files')}"]
                            if added else []))
            rows.append(("Local changes",
                         f"{_code(src['patch'])}" + (f", which {what}" if what else "")
                         + ". Apply it with `git apply` at the checkout root, after the commit "
                           "and its submodules are checked out."))
        elif changed:
            rows.append(("Local changes", f"{_count(len(changed), 'path differs', 'paths differ')}"
                                          " from the commit, and no patch of them was saved."))
        else:
            rows.append(("Local changes", "none"))
        if left:
            rows.append(("Left out of the patch",
                         _join([_code(u) for u in left]) + ", too large to carry. Copy "
                         + ("it" if len(left) == 1 else "them") + " by hand."))
    for rel, m in mods:
        frm = str(m.get("from") or "")
        where = (str(Path(frm).parent) + "/") if frm else ""
        rows.append(("Source file copied into the case",
                     f"{_code(Path(rel).name)}" + (f", from {_code(where)}" if where else "")))
    if src.get("fingerprint"):
        rows.append(("Fingerprint of the code", _code(src["fingerprint"])))
    L += _table(["", ""], rows)
    if runs and src.get("fingerprint"):
        L += ["Each case job records the fingerprint of the code it built, so that the two "
              "can be compared.", ""]
    rep = _dict(_dict(preview.get("control")).get("reproduced"))
    if not runs and rep.get("fingerprint"):
        years = [str(y) for y in _list(rep.get("years"))]
        when = f" for {_join(years)}" if years else ""
        if rep["fingerprint"] == src.get("fingerprint"):
            L += [f"The control run was made again from this code{when}, and it came out the "
                  f"same bit for bit.", ""]
        else:
            L += [f"The control run was made again{when} from code with the fingerprint "
                  f"{_code(rep['fingerprint'])}, and it came out the same bit for bit. That "
                  f"is not the code of this checkout now.", ""]
    return L


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: THE CONTROL RUN'S CASE
# ─────────────────────────────────────────────────────────────────────────────
def _control_step(n: int, preview: Dict[str, Any], ctl_recipe: Dict[str, Any],
                  runs: List[Dict[str, Any]]) -> List[str]:
    ctl = _dict(preview.get("control"))
    rows = []
    if ctl.get("case_dir"):
        rows.append(("Case", _code(ctl["case_dir"])))
    if ctl.get("run_dir"):
        rows.append(("Output", _code(ctl["run_dir"])))
    if preview.get("control_recipe"):
        rows.append(("Recipe, as read from the case" if ctl.get("reuse") else "Recipe",
                     _code(preview["control_recipe"])))
    if not rows:
        return []
    head = "Start from the control run's case" if runs else "The control run's case"
    L = [f"## Step {n}. {head}", ""] + _table(["", ""], rows)
    names = [str(e.get("name")) for e in runs]
    if runs and ctl.get("reuse"):
        L += [("The new run is this recipe with the changes of step "
               f"{n + 1}." if len(runs) == 1 else
               f"Each new run is this recipe with its own changes of step {n + 1}.")
              + " Nothing is run for the control run.", ""]
    elif runs:
        made = "control" in names
        L += [("The control run is made too, as the run `control` below. " if made else "")
              + "Each new run is this recipe with its own changes.", ""]
    else:
        L += _recipe_lines(ctl_recipe, n)
    return L


def _recipe_lines(recipe: Dict[str, Any], n: int) -> List[str]:
    """What a recipe holds, for a study that makes no run: what its case was
    created with, then how many settings follow."""
    if not recipe:
        return []
    rows = [(label, _code(recipe[k])) for k, label in TOP_KEYS[:5] if recipe.get(k)]
    if recipe.get("start"):
        rows.append(("Start date", str(recipe["start"])))
    if not rows:
        return []
    L = ["The recipe holds what the case was created with.", ""] + \
        _table(["Setting", "Value"], rows)
    parts = []
    nx = len(_xml(recipe))
    if nx:
        parts.append(f"the {_count(nx, 'setting', 'settings')} made with `xmlchange` after that")
    for comp, body in _dict(recipe.get("user_nl")).items():
        if namelist(body):
            parts.append(f"the lines of `user_nl_{comp}`")
    uf = len(_dict(recipe.get("user_files")))
    if uf:
        parts.append(_count(uf, "file copied into the case", "files copied into the case"))
    sm = len(_dict(recipe.get("sourcemods")))
    if sm:
        parts.append(f"the source file{'s' if sm > 1 else ''} of step {n - 1}")
    if parts:
        L += [f"It also holds {_join(parts)}.", ""]
    return L


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: THE CHANGES
# ─────────────────────────────────────────────────────────────────────────────
def _label(key: str, where: str = "") -> str:
    name = PLAIN_NAMES.get(key)
    code = _code(key) + (f" in {_code(where)}" if where else "")
    return f"{name}, {code}" if name else code


def _streams_of(names: Sequence[str], comp: str) -> List[str]:
    return [n for n in names if str(n).startswith(f"user_{comp}.streams")]


def change_rows(run: Dict[str, Any], ctl: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """(the setting, its value in the control run, its value in the new run)
    for each way a run's recipe differs from the control run's: the case's
    creation, the XML settings, each user_nl key (a component the control run
    sets nothing in is one row), the source files, and a file the control
    run's case copies in that the run does not."""
    rows: List[Tuple[str, str, str]] = []
    for k, label in TOP_KEYS:
        if run.get(k) != ctl.get(k):
            rows.append((label, _code(ctl.get(k)) if ctl.get(k) else "not set",
                         _code(run.get(k)) if run.get(k) else "not set"))
    cx, rx = _xml(ctl), _xml(run)
    if run.get("start") != ctl.get("start") and cx.get("RUN_STARTDATE") == rx.get("RUN_STARTDATE"):
        rows.append(("Start date", str(ctl.get("start") or "not set"),
                     str(run.get("start") or "not set")))
    for k, v in rx.items():
        if cx.get(k) != v:
            rows.append((_label(k), _plain(cx[k]) if k in cx else "not set", _plain(v)))
    for k, v in cx.items():
        if k not in rx:
            rows.append((_label(k), _plain(v), "not set"))
    a_nl, b_nl = _dict(ctl.get("user_nl")), _dict(run.get("user_nl"))
    new_files = [n for n in _dict(run.get("user_files")) if n not in _dict(ctl.get("user_files"))]
    start_year = str(ctl.get("start") or "")[:4]
    for comp in dict.fromkeys(list(a_nl) + list(b_nl)):
        a, b = namelist(a_nl.get(comp)), namelist(b_nl.get(comp))
        if a == b:
            continue
        where = f"user_nl_{comp}"
        if not a or not b:
            # A COMPONENT ONE SIDE SETS NOTHING IN is one row: its lines together
            words = " and ".join(_code(f"{k} = {v}") for k, v in (b or a).items())
            more = len(_streams_of(new_files, comp)) if b else 0
            if more:
                words += (", read from one more stream file" if more == 1 else
                          f", read from {_number(more)} more stream files")
            label = (f"{COMPONENT_NAMES[comp]}, {_code(where)}" if comp in COMPONENT_NAMES
                     else _code(where))
            rows.append((label, "nothing set" if b else words, words if b else "nothing set"))
            continue
        for k, v in b.items():
            if a.get(k) == v:
                continue
            if k.lower() == "finidat" and a.get(k):
                was = (f"the state the control run started from in {start_year}" if start_year
                       else "the state the control run started from")
            else:
                was = _value(a[k]) if k in a else "not set"
            rows.append((_label(k, where), was, _value(v)))
        for k, v in a.items():
            if k not in b:
                rows.append((_label(k, where), _value(v), "not set"))
    a_sm, b_sm = _dict(ctl.get("sourcemods")), _dict(run.get("sourcemods"))
    for rel, m in b_sm.items():
        if _dict(a_sm.get(rel)).get("sha256") != _dict(m).get("sha256"):
            rows.append((f"Source file {_code(rel)}",
                         "another copy" if rel in a_sm else "none",
                         f"from {_code(_dict(m).get('from'))}"))
    for rel in a_sm:
        if rel not in b_sm:
            rows.append((f"Source file {_code(rel)}", "copied into the case", "not copied"))
    for name in _dict(ctl.get("user_files")):
        if name not in _dict(run.get("user_files")):
            rows.append((_code(name), "copied into the case", "not copied"))
    return rows


_HOW = {"factor": ("a factor on precipitation", "The factor"),
        "offset": ("an amount added to the air temperature", "The amount")}


def weather_words(w: Dict[str, Any]) -> str:
    """What a weather file the server wrote holds, from its own summary."""
    rec = _dict(w.get("records"))
    fields = [(str(k), _dict(f)) for k, f in _dict(w.get("fields")).items()]
    acting = [(k, f) for k, f in fields if _list(f.get("months"))]
    idle = [(k, f) for k, f in fields if not _list(f.get("months"))]
    span = ""
    if rec.get("first") and rec.get("last"):
        span = f"for each month from {_month(rec['first'])} to {_month(rec['last'])}"
    out = []
    names = [_HOW.get(str(f.get("how")), (f"`{k}`, {f.get('how')}",))[0] for k, f in acting]
    if names:
        out.append(_sentence(_join(names) + (f" {span}" if span else "")))
    elif span:
        out.append(_sentence(f"Values {span} that change nothing"))
    for k, f in acting:
        what = _HOW.get(str(f.get("how")), (None, f"The value of `{k}`"))[1]
        units = str(f.get("units") or "")
        unit = f" {units}" if units and units != "1" else ""
        vals = [v for v in _list(f.get("values")) if isinstance(v, (int, float))]
        val = ("is changed" if not vals else f"is {vals[0]:g}{unit}" if len(vals) == 1 else
               f"takes the values {_join([f'{v:g}{unit}' for v in vals])}")
        cells = f.get("cells")
        where = f" in {cells:,} cells" if isinstance(cells, int) else ""
        ident = f.get("identity")
        rest = f", and {ident:g}{unit} everywhere else" if isinstance(ident, (int, float)) else ""
        out.append(f"{what} {val}{where} in {months_words(f['months'])}{rest}.")
    for k, f in idle:
        name = _HOW.get(str(f.get("how")), (f"the value of `{k}`",))[0]
        name = name.replace("a factor", "the factor").replace("an amount", "the amount")
        ident = f.get("identity")
        if isinstance(ident, (int, float)):
            out.append(_sentence(f"{name} is {ident:g} everywhere"))
    return " ".join(out)


def _stream_reads(text: Optional[str]) -> List[str]:
    """The variables a data atmosphere stream file reads from its files."""
    if not text:
        return []
    m = re.search(r"<fieldInfo>.*?<variableNames>(.*?)</variableNames>", text, re.S)
    if not m:
        return []
    return [ln.split()[0] for ln in m.group(1).splitlines() if ln.split()]


def _find(path: Any, run_dir, *more: Any) -> Optional[str]:
    """A written file's text: at its own path, at its place in this study's
    folder, or at one of the other places given."""
    for p in (path, _rel(path, run_dir) if path else None, *more):
        if not p:
            continue
        q = Path(str(p)) if str(p).startswith("/") else (Path(run_dir) / str(p)
                                                        if run_dir is not None else None)
        try:
            if q is not None and q.is_file():
                return q.read_text()
        except OSError:
            continue
    return None


def _files_lines(e: Dict[str, Any], recipe: Dict[str, Any], ctl: Dict[str, Any],
                 run_dir) -> List[str]:
    """The table of the files written for a run, and what each holds."""
    weather = [w for w in _list(e.get("weather_files")) if isinstance(w, dict)]
    by_var = {str(k): _dict(f) for w in weather for k, f in _dict(w.get("fields")).items()}
    rows, dirs, all_weather = [], [], bool(weather)
    for w in weather:
        rows.append((w.get("file") or Path(str(w.get("path") or "")).name, weather_words(w)))
        dirs.append(str(Path(_rel(w["path"], run_dir)).parent) if w.get("path") else None)
    case_dir = Path(str(e.get("case_script"))).parent if e.get("case_script") else None
    for name, m in _dict(recipe.get("user_files")).items():
        if name in _dict(ctl.get("user_files")):
            continue
        m = _dict(m)
        stream = str(name).startswith("user_datm.streams")
        all_weather = all_weather and stream
        if stream:
            text = _find(m.get("from"), run_dir,
                         case_dir / "user_files" / name if case_dir else None)
            reads = [by_var[v] for v in _stream_reads(text) if v in by_var]
            what = _join(["the factor" if f.get("how") == "factor" else
                          "the amount" if f.get("how") == "offset" else "its values"
                          for f in reads])
            rows.append((name, "The stream file that tells the data atmosphere to read "
                               + (what if what else "the file of the change") + "."))
        else:
            rows.append((name, "A file the case copies in."))
        dirs.append(str(Path(_rel(m["from"], run_dir)).parent) if m.get("from") else None)
    if not rows:
        return []
    n = len(rows)
    lead = (f"{_cap(_count(n, 'file carries', 'files carry'))} the change of weather."
            if all_weather else f"{_cap(_count(n, 'file is', 'files are'))} written for the run.")
    if None not in dirs and len(set(dirs)) == 1:
        # one folder holds them all: said once, and each file by its name
        d = dirs[0].rstrip("/") + "/"
        lead += (f" It is in {_code(d)}." if n == 1 else f" Both are in {_code(d)}." if n == 2
                 else f" All {_number(n)} are in {_code(d)}.")
        table = _table(["File", "What it holds"], [(_code(f), w) for f, w in rows])
    else:
        # each file where it is, from the study folder where it lies inside it
        table = _table(["File", "What it holds"],
                       [(_code(f"{d}/{f}" if d else f), w) for (f, w), d in zip(rows, dirs)])
    return [lead, ""] + table


def _changes_step(n: int, e: Dict[str, Any], run_dir, ctl_recipe: Dict[str, Any],
                  head: str) -> List[str]:
    recipe = _load(run_dir, e.get("recipe"))
    recipe = recipe if isinstance(recipe, dict) else {}
    L = [f"{head} Step {n}. Make the changes", ""]
    if e.get("case_dir"):
        intro = f"The new case is {_code(e['case_dir'])}."
    elif e.get("case"):
        intro = f"The new case is named {_code(e['case'])}."
    else:
        intro = ""
    rows = change_rows(recipe, ctl_recipe) if recipe and ctl_recipe else None
    if rows is None:
        if e.get("recipe") and recipe:
            intro += f" Its recipe is {_code(e['recipe'])}."
        L += [intro.strip(), ""] if intro.strip() else []
    elif rows:
        L += [(intro + " " if intro else "") + "It differs from the control run's case in "
              f"{_count(len(rows), 'way', 'ways')}.", ""]
        L += _table(["Change", "The control run", "The new run"], rows)
    else:
        L += [(intro + " " if intro else "") + "Its recipe is the control run's, under a new "
                                               "case name.", ""]
    L += _files_lines(e, recipe, ctl_recipe, run_dir)
    script = _load(run_dir, e.get("case_script"), as_json=False)
    if e.get("case_script") and script is not None:
        L += [f"The script {_code(e['case_script'])} creates the case with these changes."
              + (" It refuses a case that already exists." if "case exists" in script else ""),
              ""]
    return L if len(L) > 2 else []


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: THE JOBS
# ─────────────────────────────────────────────────────────────────────────────
def _sbatch(script: Optional[str], key: str) -> Optional[str]:
    m = re.search(rf"^#SBATCH\s+--{key}[= ](\S+)", script or "", re.M)
    return m.group(1) if m else None


def _smoke_days(spec: Dict[str, Any], script: Optional[str]) -> Optional[int]:
    if isinstance(spec.get("smoke_days"), int):
        return spec["smoke_days"]
    m = re.search(r"gate smoke --days (\d+)", script or "")
    return int(m.group(1)) if m else None


def stage_words(stage: str, spec: Dict[str, Any], script: Optional[str],
                year_settings: Dict[str, Any]) -> str:
    if stage in STAGE_WORDS:
        return STAGE_WORDS[stage]
    if stage == "smoke":
        d = _smoke_days(spec, script)
        return (f"runs {d} day{'' if d == 1 else 's'} from the starting state" if d else
                "runs a few days from the starting state")
    m = re.fullmatch(r"y(\d{4})", stage)
    if m:
        sets = [s for s in _list(year_settings.get(m.group(1)))
                if isinstance(s, (list, tuple)) and len(s) > 1]
        return f"runs {m.group(1)}" + (", after setting " + _join([_code(f"{s[0]}={s[1]}")
                                                                    for s in sets])
                                       if sets else "")
    return ""


def _jobs_step(n: int, e: Dict[str, Any], run_dir, head: str) -> List[str]:
    jobs = [j for j in _list(e.get("jobs")) if isinstance(j, dict)]
    if not jobs:
        return []
    script = _load(run_dir, e.get("stage_script"), as_json=False)
    spec = _dict(_load(run_dir, e.get("chain_spec")))
    ex = _dict(_load(run_dir, e.get("expectations")))
    queues = list(dict.fromkeys(str(j.get("partition")) for j in jobs if j.get("partition")))
    L = [f"{head} Step {n}. Run the jobs", ""]
    if e.get("stage_script") and script is not None:
        nodes, account = _sbatch(script, "nodes"), _sbatch(script, "account")
        where = []
        if nodes and nodes.isdigit():
            where.append(f"on {_count(int(nodes), 'node', 'nodes')}")
        if len(queues) == 1:
            where.append(f"of the queue {_code(queues[0])}" if where else
                         f"in the queue {_code(queues[0])}")
        if account:
            where.append(f"under the account {_code(account)}")
        L += [f"The script {_code(e['stage_script'])} runs one stage as its own job"
              + (", " + " ".join(where) if where else "")
              + ". Submit it once for each stage, in this order.", ""]
    else:
        L += ["Each stage below is its own job"
              + (f", in the queue {_code(queues[0])}" if len(queues) == 1 else "")
              + ". They run in this order.", ""]
    several_queues = len(queues) > 1
    head_row = ["Stage", "What it does", "Wall time", "Waits for"] + (["Queue"] if several_queues
                                                                       else [])
    rows = []
    for j in jobs:
        stage = str(j.get("stage") or "")
        rows.append([stage, stage_words(stage, spec, script, _dict(ex.get("year_settings"))),
                     wall_words(j.get("walltime")),
                     j.get("after") or ("every job above" if stage == "report" else "nothing")]
                    + ([_code(j.get("partition"))] if several_queues else []))
    L += _table(head_row, rows)
    tail = ("Each job starts only when the one before it ended well, and a failure cancels "
            "the jobs after it.")
    if any(str(j.get("stage")) == "report" for j in jobs):
        tail += " The report job runs whatever happens."
    if e.get("after"):
        tail += f" The first job also waits for the case job of the run {_code(e['after'])}."
    L += [tail, ""]
    return L


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: THE CHECKS
# ─────────────────────────────────────────────────────────────────────────────
def check_words(kind: str, words: Dict[str, Any]) -> Optional[str]:
    """What one kind of job checks, in the page's plain words while the
    server's words are those they were written from; else the server's own."""
    said = words.get(kind)
    if not said:
        return None
    if " ".join(str(said).split()) == SERVER_WORDS.get(kind) and kind in PLAIN_WORDS:
        return PLAIN_WORDS[kind]
    return _sentence(_safe(said))


def _units_of(o: Dict[str, Any], weather: List[Dict[str, Any]]) -> str:
    name = Path(str(o.get("anomaly_file") or "")).name
    for w in weather:
        if w.get("file") == name or Path(str(w.get("path") or "")).name == name:
            f = _dict(_dict(w.get("fields")).get(str(o.get("anomaly_var"))))
            u = str(f.get("units") or "")
            return u if u and u != "1" else ""
    return ""


def rule_words(o: Dict[str, Any], weather: List[Dict[str, Any]]) -> str:
    """One output rule of a run's expectations, in a sentence, with its
    tolerance."""
    why = _safe(o.get("why"))
    rule = str(o.get("rule") or "")
    tol = o.get("tolerance")
    if rule == "all_zero":
        field = o.get("field") or _join([str(x) for x in _list(o.get("fields"))])
        return _sentence(f"{_code(field)} is 0 in every cell and every record of the year"
                         + (f", because {why}" if why else ""))
    if rule == "matches_anomaly" and isinstance(tol, (int, float)) and tol > 0:
        if o.get("how") == "factor":
            return _sentence(f"{why}, within a relative difference of {_decimal(tol)}")
        units = _units_of(o, weather)
        return _sentence(f"{why}, within {_decimal(tol)}" + (f" {units}" if units else ""))
    s = _sentence(why) if why else _sentence(f"The rule {_code(rule)} holds")
    if isinstance(tol, (int, float)):
        s += " The tolerance is zero." if tol == 0 else f" The tolerance is {_decimal(tol)}."
    return s


def _checks_step(n: int, e: Dict[str, Any], run_dir, words: Dict[str, Any],
                 head: str) -> List[str]:
    ex = _dict(_load(run_dir, e.get("expectations")))
    spec = _dict(_load(run_dir, e.get("chain_spec")))
    jobs = [j for j in _list(e.get("jobs")) if isinstance(j, dict)]
    stages = {str(j.get("stage")) for j in jobs}
    rows = []
    case = check_words("case", words)
    if case and ("case" in stages or not jobs):
        files = [str(k) for k in _dict(ex.get("case_files"))]
        if files and all(".streams." in f for f in files):
            case += (" The stream file is in the case, unchanged." if len(files) == 1 else
                     f" The {_number(len(files))} stream files are in the case, "
                     f"unchanged.")
        elif files:
            case += (f" The {_count(len(files), 'file', 'files')} written for the run "
                     f"{'is' if len(files) == 1 else 'are'} in the case, unchanged.")
        rows.append(("case", case))
    smoke = check_words("smoke", words)
    if smoke and ("smoke" in stages or (not jobs and spec.get("smoke", True))):
        rows.append(("smoke", smoke))
    year = check_words("year", words)
    if year:
        rows.append(("each year", year))
    weather = [w for w in _list(e.get("weather_files")) if isinstance(w, dict)]
    rules = [o for o in _list(ex.get("output")) if isinstance(o, dict)]
    if rules:
        changes = {str(o.get("change")) for o in rules if o.get("change")}
        rows.append(("each year, the changes" if len(changes) > 1 else "each year, the change",
                     " ".join(rule_words(o, weather) for o in rules)))
    ref = _dict(spec.get("reference"))
    against = check_words("reference", words)
    if ref and against:
        gated = [int(y) for y in _list(ref.get("gate_years")) if str(y).isdigit()]
        if against == PLAIN_WORDS["reference"]:
            against += (" They are not held to a limit, because the run is meant to differ."
                        if not gated else
                        f" In {_join([str(y) for y in gated])} they are also held to the model "
                        f"server's limits.")
        rows.append(("each year, against the control run", against))
    if not rows:
        return []
    L = [f"{head} Step {n}. Check each job", ""]
    stage = Path(str(e.get("stage_script"))).name if e.get("stage_script") else None
    kept = None
    if e.get("stage_script"):
        mine = str(Path(str(e["stage_script"])).parent / "expectations.json")
        kept = mine if _exists(run_dir, mine) else None
    if kept is None and _exists(run_dir, e.get("expectations")):
        kept = str(e["expectations"])
    lead = (f"{_code(stage)} makes these checks itself." if stage else "")
    if kept:
        lead += f" The values that each check must meet are in {_code(kept)}."
    if lead.strip():
        L += [lead.strip(), ""]
    L += _table(["Job", "What must hold"], rows)
    return L


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: THE INPUT FILES
# ─────────────────────────────────────────────────────────────────────────────
def _nl_files(recipe: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(key, path) for every file the land namelist names."""
    out = []
    for k, v in namelist(_dict(recipe.get("user_nl")).get("elm")).items():
        m = re.fullmatch(r"'([^']*/[^']*)'", v.strip()) or re.fullmatch(r'"([^"]*/[^"]*)"',
                                                                         v.strip())
        if m:
            out.append((k, m.group(1)))
    return out


def _weather(s: Dict[str, Any]) -> bool:
    return bool(set(map(str, _list(s.get("fields")))) & WEATHER_FIELDS)


def _stream_cell(s: Dict[str, Any]) -> str:
    if s.get("error"):
        return f"not readable. {_sentence(_safe(s['error']))}"
    n, first, last = s.get("n_files"), s.get("first"), s.get("last")
    files = (f"the file {_code(first)}" if not last or last == first or n == 1 else
             f"the {n:,} files {_code(first)} to {_code(last)}" if isinstance(n, int) else
             f"the files {_code(first)} to {_code(last)}")
    return files + (f" in {_code(s['data_dir'])}" if s.get("data_dir") else "")


def _stream_name(s: Dict[str, Any]) -> str:
    name = str(s.get("stream") or "")
    tail = name.split("datm.streams.txt.", 1)[-1]
    key = tail.split(".", 1)[0]
    return STREAM_NAMES.get(key, f"the files of the stream {_code(name)}")


def _inputs_step(n: int, preview: Dict[str, Any], first: Dict[str, Any], runs: bool,
                 start_step: Optional[int], start_row: Optional[str],
                 ctl_recipe: Optional[Dict[str, Any]] = None) -> List[str]:
    ic = _dict(preview.get("inputs_check"))
    rows = []
    nl = dict(_nl_files(first))
    xml = _xml(first)
    if ic:
        surface, domain, start = ic.get("surface"), ic.get("domain"), ic.get("restart")
    else:
        surface, start = nl.get("fsurdat"), nl.get("finidat")
        domain = (f"{xml['LND_DOMAIN_PATH'].rstrip('/')}/{xml['LND_DOMAIN_FILE']}"
                  if xml.get("LND_DOMAIN_FILE") and xml.get("LND_DOMAIN_PATH") else
                  xml.get("LND_DOMAIN_FILE"))
    if surface:
        rows.append(("Surface data", _code(surface)))
    if domain:
        rows.append(("Grid file", _code(domain)))
    if start:
        rows.append(("Starting state", f"the saved state of step {start_step}"
                     if start_step and start_row == start else _code(start)))
    streams = [s for s in _list(preview.get("forcing")) if isinstance(s, dict)]
    weather = [s for s in streams if _weather(s)]
    for s in weather:
        rows.append(("Weather" if len(weather) == 1 else f"Weather, {_code(s.get('stream'))}",
                     _stream_cell(s)))
    # a stream the server could not read is named as such, never called the weather
    for s in streams:
        if s.get("error") and s not in weather:
            rows.append((f"The stream {_code(s.get('stream'))}", _stream_cell(s)))
    if not rows:
        return []
    L = [f"## Step {n}. The input files", ""] + _table(["Input", "File"], rows)
    if ic:
        c = _dict(ic.get("checks"))
        three = bool(surface and domain and start)
        cells = c.get("active_cells")
        L += [f"The model server checked "
              + ("the first three" if three else "the surface data, the grid file and the "
                                                  "starting state")
              + " as one set"
              + (f" of {cells:,} cells" if isinstance(cells, int) else "")
              + (", and the set passed." if ic.get("pass") else
                 ", and the set failed. " + " ".join(_sentence(_safe(r))
                                                     for r in _list(ic.get("reasons"))))
              + (f" Its record is {_code(ic['path'])}." if ic.get("path") else ""), ""]
    # the rest, where they are the control run's: a file a run's change sets is
    # a row of the table of changes, and not said here to be the control run's
    theirs = dict(_nl_files(ctl_recipe)) if runs and ctl_recipe else None
    others = [FILE_NAMES.get(k, _code(k)) for k, v in _nl_files(first)
              if k not in ("fsurdat", "finidat") and (theirs is None or theirs.get(k) == v)]
    others += [_stream_name(s) for s in streams if s not in weather and not s.get("error")]
    if others:
        L += [("The other files that the model reads are those of the control run's case. "
               if runs else
               "The other files that the model reads are named in the control run's recipe. ")
              + f"They include {_join(list(dict.fromkeys(others)))}.", ""]
    return L


# ─────────────────────────────────────────────────────────────────────────────
# THE PAGE
# ─────────────────────────────────────────────────────────────────────────────
def _changed_start(run: Dict[str, Any], ctl: Dict[str, Any]) -> Optional[str]:
    """The run's starting state, where step 3 lists it as a change."""
    a = namelist(_dict(ctl.get("user_nl")).get("elm"))
    b = namelist(_dict(run.get("user_nl")).get("elm"))
    v = b.get("finidat")
    if not v or a.get("finidat") == v:
        return None
    m = re.fullmatch(r"['\"](.*)['\"]", v.strip())
    return m.group(1) if m else v


def render(preview, run_dir, strategy=None, brief=None) -> str:
    """The whole text of RUN_BY_HAND.md, or "" when the server built nothing:
    no record, a refused design or one that would not build."""
    preview = _dict(preview)
    if not preview or preview.get("refused") or preview.get("error"):
        return ""
    runs = [e for e in _list(preview.get("runs")) if isinstance(e, dict)]
    ctl_recipe = _dict(_load(run_dir, preview.get("control_recipe")))
    first = (_dict(_load(run_dir, runs[0].get("recipe"))) if runs else {}) or ctl_recipe
    names = [str(e.get("name")) for e in runs]
    if len(runs) == 1:
        title = f"# How to make this run by hand: {names[0]}"
        who = "the new run"
    elif runs:
        title = "# How to make these runs by hand"
        who = f"the new runs {_join([_code(x) for x in names])}"
    else:
        title = "# How to make the control run by hand"
        who = None
    if runs:
        opening = (f"This page is for a modeller who wants to make {who} without the framework. "
                   f"The model server wrote every file of this study's folder that the page "
                   f"names when it checked and built the run{'s' if len(runs) > 1 else ''} "
                   f"for the design page. Nothing has been submitted. A path that does not "
                   f"start with a slash is inside this study's folder.")
    else:
        opening = ("This page is for a modeller who wants to make the control run again "
                   "without the framework. This study makes no new run. It reads the control "
                   "run's output as it stands, so nothing is run and nothing has been "
                   "submitted. The model server wrote every file of this study's folder that "
                   "the page names when it checked the design. A path that does not start "
                   "with a slash is inside this study's folder.")
    body: List[str] = []
    n = 1
    code = _code_step(n, preview, first, bool(runs))
    if code:
        body += code
        n += 1
    control = _control_step(n, preview, ctl_recipe, runs)
    if control:
        body += control
        n += 1
    starts = set()
    if runs:
        changes_n = n
        words = _dict(preview.get("checks_in_words"))
        for e in runs:
            head = "###" if len(runs) > 1 else "##"
            if len(runs) > 1:
                body += [f"## Run {_code(e.get('name'))}", ""]
            recipe = _dict(_load(run_dir, e.get("recipe")))
            starts.add(_changed_start(recipe, ctl_recipe) if recipe and ctl_recipe else None)
            body += _changes_step(changes_n, e, run_dir, ctl_recipe, head)
            body += _jobs_step(changes_n + 1, e, run_dir, head)
            body += _checks_step(changes_n + 2, e, run_dir, words, head)
        n = changes_n + 3
    one_start = next(iter(starts)) if len(starts) == 1 else None
    body += _inputs_step(n, preview, first, bool(runs), changes_n if runs else None, one_start,
                         ctl_recipe)
    if not body:
        return ""
    return "\n".join([title, "", opening, ""] + body).rstrip("\n") + "\n"
