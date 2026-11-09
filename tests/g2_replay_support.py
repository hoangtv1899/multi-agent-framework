"""What the replay tests of G part 2's fixes share (2026-11-08): a scripted
model keyed by client label, and the guard that no real model client is ever
constructed. Not a test module; the test files import it.

A REPLAY, NOT A CALL. Each client label (step2a_plan, step3_plan_review,
step2a_plan_rewrite, step3_plan_review_rewrite, step2_investigate,
step3_interpret) gets its own list of replies, given back in order; every
prompt is kept for the test to read. A label asked more often than it has
replies fails the test, unless the replay is told what to answer then.
"""
import json
from typing import Any, Dict, List, Optional


class Replies:
    """The scripted model of one replay: {label: [reply, ...]}."""

    def __init__(self, by_label: Dict[str, List[Any]], extra: Optional[Dict[str, Any]] = None):
        self.by_label = {k: list(v) for k, v in by_label.items()}
        self.extra = dict(extra or {})          # a label's answer once its list is used up
        self.asked: List[tuple] = []            # (label, prompt), in the order asked

    def clients(self) -> Dict[str, "Client"]:
        labels = set(self.by_label) | set(self.extra)
        return {lab: Client(self, lab) for lab in labels}

    def prompts(self, label: str) -> List[str]:
        return [p for lab, p in self.asked if lab == label]

    def labels_asked(self) -> List[str]:
        return [lab for lab, _ in self.asked]


class Client:
    """One label's scripted client: ask() gives back that label's next reply."""

    def __init__(self, owner: Replies, label: str):
        self.owner, self.label = owner, label

    def ask(self, messages, **kw):
        content = messages[0]["content"]
        if isinstance(content, list):                 # text and images: keep the text
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        self.owner.asked.append((self.label, content))
        queue = self.owner.by_label.get(self.label) or []
        if queue:
            r = queue.pop(0)
        elif self.label in self.owner.extra:
            r = self.owner.extra[self.label]
        else:
            raise AssertionError(f"{self.label} was asked more often than the replay holds")
        if isinstance(r, BaseException):
            raise r
        # as a model writes it: its characters as they are, never escaped
        return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)


def record_strings(path) -> List[str]:
    """Every string a JSON, JSON-lines or Markdown record holds, decoded (a
    JSON file keeps a dash as the escape \\u2014, so its text alone would
    hide one)."""
    from pathlib import Path
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".md":
        return [text]
    docs = [json.loads(x) for x in text.splitlines() if x.strip()] if p.suffix == ".jsonl" \
        else [json.loads(text)]
    out: List[str] = []

    def walk(x):
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for k, v in x.items():
                out.append(str(k))
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for d in docs:
        walk(d)
    return out


def forbid_model_clients(monkeypatch) -> None:
    """Constructing the real SimpleLLMClient raises, wherever a step imports
    it: no model call of any kind in these tests."""
    import agents.llm_agent as LA

    class NoModel:
        def __init__(self, *a, **k):
            raise AssertionError("a real model client was constructed in a test")
    monkeypatch.setattr(LA, "SimpleLLMClient", NoModel)
    for mod in ("agents.planner", "agents.tool_loop", "agents.analyzer_agent"):
        try:
            monkeypatch.setattr(f"{mod}.SimpleLLMClient", NoModel)
        except (ImportError, AttributeError):
            pass
