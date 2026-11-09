#!/usr/bin/env python3
"""
No dashes as punctuation in the model's words kept in a record
src/agents/analysis/dashes.py

    strip_dashes(text)      one string: an em-dash or an en-dash used as
                            punctuation becomes a comma; a numeric range keeps
                            its en-dash ("0.78–0.84")
    strip_dashes_in(obj)    every string inside a dict or a list, keys and
                            non-strings untouched

ONE HELPER FOR EVERY RECORD (the user's decision on G part 2, 2026-11-08).
The rule "no em-dash" held for the page and the study record, because step 3
stripped its own claims and step 4 stripped the page's copy, but the model's
other words reached their records as written: question 1's plan review kept
two em-dashes in analysis_plan_review.json. Every model-written string the
framework copies into a JSON or Markdown record now goes through here first:
the plan's text fields (before the plan file is written and hashed), the plan
review's why and problems, an amendment's why, step 2b's notes, reasoning,
figure texts and skipped reasons, step 3's feedback and open causes, and the
planned phase's reasons. The verbatim exchanges (*_prompt.txt, *_reply.txt)
stay verbatim: they are the record of what was said.

This was step3_interpret._strip_dashes; that name stays as an alias.
"""
import re
from typing import Any

EM_DASH = "\u2014"
EN_DASH = "\u2013"


def strip_dashes(text) -> str:
    """Em- and en-dashes used as punctuation become a comma; ranges keep theirs."""
    t = str(text or "")
    t = t.replace(f" {EM_DASH} ", ", ").replace(EM_DASH, ", ")
    t = re.sub(rf"(?<!\d)\s*{EN_DASH}\s*(?!\d)", ", ", t)
    return t


def strip_dashes_in(obj: Any) -> Any:
    """`obj` with every string inside it through strip_dashes: the values of a
    dict (never its keys, which name things) and the items of a list. None,
    numbers and booleans are returned as they are."""
    if isinstance(obj, str):
        return strip_dashes(obj)
    if isinstance(obj, dict):
        return {k: strip_dashes_in(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strip_dashes_in(v) for v in obj]
    return obj
