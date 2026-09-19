#!/usr/bin/env python3
"""A declared key list that fails when a producer writes a key it does not name.

    src/core/keyset.py

THE BUG THIS EXISTS FOR. Nine times between 2026-08-07 and 2026-08-13 a value
was computed, written, and then not carried — because somewhere downstream a
hand-maintained list of key names decided what to copy, and the new key was not
on it. None of them raised. `spinup_dropped` was measured by the extractor,
folded into experiment.json, read by step 4 — and arrived as None, because
step 0 rebuilt its `data` dict from a list of eleven names and that was not one
of them. The value existed at every layer and vanished at one.

A whitelist that silently drops is indistinguishable from a whitelist that is
right. That is the whole problem: nothing about the run says a key went
missing, so the only way to find it is to already suspect it.

WHY THE LISTS ARE NOT SIMPLY DELETED. Copying everything would fix the drops
and cause worse: an object added upstream becomes a repr in a JSON artifact, a
station's name leaks into a row that is supposed to carry measurements, and the
package grows fields nobody declared. The list is doing real work. It just has
to be *complete* about what it decided.

SO A KEY MUST BE CLASSIFIED, NOT MERELY OMITTED. A KeySet holds two lists:

    keep   the keys this copy takes
    drop   the keys it deliberately leaves, each because someone read it,
           routed it elsewhere, or judged it not worth carrying

A key in neither is not a decision — it is a key that appeared after the list
was written. That raises, names itself, and says where the list lives. Fixing
it is one line, and the line is a decision someone made on purpose.

THE OTHER HALF: A LIST MAY NAME A KEY NOBODY WRITES. `COLUMN_METADATA` has
asked for `wtd_prior_m`, `wtd_prior_source` and `wtd_prior_uncertainty_m` since
the Fan prior left the sampler on 2026-08-07 — three names describing a
capability the pipeline no longer has. That cannot raise: a key can be legally
absent on one run and present on the next (`station_id` is null on every column
that was not pinned). So it is reported instead. Every `take()` records which
declared keys actually appeared, and `report()` at the end of a run names the
ones that never did.

Two symmetric failures, two different responses:

    producer wrote a key the list does not name   →  raises, immediately
    list names a key no producer ever wrote       →  reported, after the run

USAGE

    COLUMNS = KeySet(
        "COLUMN_METADATA",
        keep = ("lat", "lon", ...),
        drop = {"id": "the join key, read by _merge_column_metadata itself"},
        source = "columns.json → columns[*]",
    )

    picked = COLUMNS.take(src)          # -> {kept key: value}, or raises

STRICTNESS. `take()` raises by default, because a startup error is the point.
`IDEAS_KEYSET_STRICT=0` downgrades it to a printed warning — for getting a run
out the door when a key appears at an awkward hour. It is not a fix, and the
warning says so.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Tuple


class UnclassifiedKey(RuntimeError):
    """A producer wrote a key the receiving list neither keeps nor drops."""


# Every KeySet built, in construction order, so one call can report on all of
# them. A module-level registry rather than a caller-held list: the sites are
# in four files across two packages, and asking each to remember to register
# is the same class of omission this module exists to catch.
_REGISTRY: List["KeySet"] = []


def _strict() -> bool:
    return os.environ.get("IDEAS_KEYSET_STRICT", "1") not in ("0", "no", "off")


class KeySet:
    """The keys a copy takes, the keys it leaves, and why it leaves them.

    `drop` may be a mapping of key -> reason or a bare iterable of keys. The
    reason is not decoration: it is what tells the next reader whether a
    dropped key is consumed somewhere else or genuinely discarded, which is
    exactly what took an afternoon to reconstruct for `record_start`.
    """

    def __init__(self,
                 name: str,
                 keep: Iterable[str],
                 drop: Any = (),
                 optional: Iterable[str] = (),
                 source: str = "",
                 where: str = ""):
        self.name = name
        self.keep: Tuple[str, ...] = tuple(keep)
        self.drop: Dict[str, str] = (dict(drop) if isinstance(drop, Mapping)
                                     else {k: "" for k in drop})
        # KEPT, BUT ALLOWED TO BE MISSING. `station_id` is absent from every
        # column of a run that pinned nothing, and reporting it there would
        # teach a reader to skip the report — which is the only way this
        # mechanism can fail. A key is optional because its ABSENCE is a
        # legitimate state of the run, not because nobody got round to it.
        self.optional: frozenset = frozenset(optional)
        self.source = source          # the artifact these keys arrive in
        self.where = where            # the file:symbol holding this list
        self._seen: set = set()       # declared keys that actually appeared
        self._calls = 0

        both = set(self.keep) & set(self.drop)
        if both:
            raise ValueError(f"{name}: {sorted(both)} listed as both kept "
                             f"and dropped — say which")
        stray = self.optional - set(self.keep)
        if stray:
            raise ValueError(f"{name}: {sorted(stray)} marked optional but "
                             f"not kept — optional qualifies a kept key")
        _REGISTRY.append(self)

    # ── the assertion ────────────────────────────────────────────────────
    def take(self, src: Mapping[str, Any]) -> Dict[str, Any]:
        """The kept keys of `src`. Raises if `src` holds a key nobody named.

        Absent kept keys come back as None, which is what every caller here
        did before and what the packaged row means by "not measured". The
        error is only ever about a key that is PRESENT and unclassified.
        """
        self.check(src)
        return {k: src.get(k) for k in self.keep}

    def check(self, src: Mapping[str, Any]) -> List[str]:
        """Assert on `src`, and record which declared keys it carried.

        Split from take() so a caller that builds its dict by hand — step 0's
        context, which routes experiment.json to three different places, and
        the metadata merge, which copies conditionally — can assert without
        handing over the copy. Both halves of the audit run here, so a caller
        that only checks still feeds report().
        """
        self._calls += 1
        self._seen |= (set(src) & set(self.keep))
        unknown = sorted(set(src) - set(self.keep) - set(self.drop))
        if unknown:
            self._fail(unknown)
        return unknown

    def _fail(self, unknown: List[str]) -> None:
        at = f"\n  the list is in  {self.where}" if self.where else ""
        src = f"\n  the key came from  {self.source}" if self.source else ""
        msg = (f"{self.name}: {len(unknown)} key(s) written by the producer "
               f"that this list neither keeps nor drops: {unknown}"
               f"{src}{at}\n"
               f"  Add each to `keep` to carry it, or to `drop` with the "
               f"reason it is not carried.\n"
               f"  An unlisted key is not a decision — it is a value that "
               f"would have gone missing without saying so.")
        if _strict():
            raise UnclassifiedKey(msg)
        print(f"   ⚠️  {msg}\n   ⚠️  IDEAS_KEYSET_STRICT=0 — the key(s) above "
              f"ARE being dropped. This is not a fix.")

    # ── the report ───────────────────────────────────────────────────────
    @property
    def never_seen(self) -> List[str]:
        """Declared keys no producer wrote, across every take() this run.

        Empty before anything was taken — a list that was never exercised
        says nothing about its keys, and reporting all of them would be
        noise on any run that did not touch this stage.
        """
        if not self._calls:
            return []
        return sorted(set(self.keep) - self._seen - self.optional)

    def __repr__(self) -> str:                                  # pragma: no cover
        return (f"KeySet({self.name!r}, keep={len(self.keep)}, "
                f"drop={len(self.drop)}, calls={self._calls})")


def report(verbose: bool = False) -> List[str]:
    """One line per list that asked for a key nothing produced.

    Called at the end of packaging. Returns the lines as well as printing
    them, so a test can assert on them without capturing stdout.
    """
    out: List[str] = []
    for ks in _REGISTRY:
        missing = ks.never_seen
        if missing:
            out.append(f"{ks.name}: declared but never produced in "
                       f"{ks._calls} record(s) — {missing}"
                       + (f"  [{ks.where}]" if ks.where else ""))
        elif verbose and ks._calls:
            out.append(f"{ks.name}: {len(ks.keep)} kept, "
                       f"{len(ks.drop)} dropped, {ks._calls} record(s) — ok")
    for line in out:
        print(f"   · {line}")
    return out


def registry() -> List["KeySet"]:
    """Every KeySet built in this process. For tests and the audit."""
    return list(_REGISTRY)
