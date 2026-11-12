#!/usr/bin/env python3
"""docs/ARCHITECTURE.md, generated from the source tree.

    python tools/make_architecture_doc.py            # rewrite the document
    python tools/make_architecture_doc.py --check    # exit 1 when it is stale

GENERATED, NOT WRITTEN (2026-11-03). tools/make_arch_map.py draws the pipeline
as a page; this script answers the questions a reader asks of the code itself:
which parts call a language model and with which model id, which MCP tools
exist and who calls them, where framework code reaches a server without MCP,
which third-party libraries are imported, and the order of a whole-domain
study. Every one of those answers is read from the source by parsing it (the
`ast` module), never from a list kept by hand, so the document cannot claim
something the code no longer does. tests/test_architecture_doc.py regenerates
it and fails when the committed copy differs.

WHAT IS STILL WRITTEN BY HAND. Two things the code cannot say: the title of
each whole-domain stage with the functions that carry it out (FLOW below,
checked on every run against the source), and a few sentences of orientation.
Both are marked "Hand-written" in the document.

HOW EACH ANSWER IS READ.
  * A function CALLS A LANGUAGE MODEL when it calls `.ask(...)` or
    `...completions.create(...)`; it HOLDS one when it constructs
    SimpleLLMClient, LLMAgent, ToolLoopAgent or an openai/anthropic client.
    Whether a stage reaches a model is a walk of the call graph from its
    functions, resolving calls through imports, `self` and attributes whose
    class the code assigns (self.planner = Planner(...)).
  * A framework function CALLS AN MCP TOOL when the tool's name appears as a
    whole string literal, bare or as server__tool, in its body or in a module
    or class constant it uses: that is the form every call site takes (a call,
    a tool constant, the Reception allowlist). A constant no function uses is
    credited to its module. A prompt names a tool as a word or as server__tool.
  * A RECORD is a string literal (or a constant bound to one, or a logical
    name in core/run_layout.LAYOUT) naming a .json, .md, .csv, .log, .patch or
    .txt file. The constant may be the module's, an imported one (also under
    an alias), or one of a module loaded by path into a local
    (DC = self._sibling_module("domain_case"); DC.CHAIN_FILE). It is WRITTEN
    or READ where it is named inside the file argument of a writing call
    (p.write_text, json.dump, open(p, "w")) or a reading one (p.read_text,
    p.exists(), open(p), _read(p)), directly or through a local name bound to
    it earlier (p = run_dir / "x.json", or a loop over record names). Named
    anywhere else (a value in a dict, a message) it is only named.
  * The tools a server SERVES are its @mcp.tool functions, or the types.Tool
    entries its @server.list_tools() function returns.
  * QUOTED TEXT (docstrings, comments) has each dash used as a pause replaced
    by a semicolon, so the document uses none.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("docs") / "ARCHITECTURE.md"
REGISTRY = "mcp_config.template.json"
PROMPTS = Path("src") / "agents" / "prompts"

# THE WHOLE-DOMAIN FLOW, hand-written (2026-11-03). The order of the stages and
# their titles are a reading of workflow.py and execute_plan that the call graph
# alone does not give (the base class's stages are reached through overrides).
# Everything else in a stage's entry is read from these functions: whether a
# language model is reached, the MCP tools named, the records read and written.
# A function named here that no longer exists stops the generator.
FLOW = [
    ("Reception reads the request, asks the person, gathers the place",
     ["src/agents/reception_llm.py::LLMReceptionAgent.__init__",
      "src/agents/reception_llm.py::LLMReceptionAgent.process",
      "workflow.py::_save_reception"]),
    ("The Planner designs the experiment against the server's rulebook",
     ["workflow.py::WorkflowCoordinator._plan",
      "workflow.py::WorkflowCoordinator._domain_rulebook",
      "src/agents/planner.py::Planner.plan"]),
    ("The model server checks and builds the design before anyone reads it; "
     "a refusal goes back to the Planner",
     ["workflow.py::WorkflowCoordinator._review_ready",
      "mcp/elm-mcp/src/elm_exp_manager.py::ELMExpManager.preview_design"]),
    ("The design review page, and the person's decision",
     ["src/agents/design_review.py::write_review",
      "src/agents/design_review.py::record_decision"]),
    ("The experiment record: control, runs and their checked changes",
     ["mcp/elm-mcp/src/elm_exp_manager.py::ELMExpManager._build_domain_experiment"]),
    ("Submit each run as a chain of scheduler jobs, with job B behind them",
     ["mcp/elm-mcp/src/elm_exp_manager.py::ELMExpManager._submit_experiment",
      "mcp/elm-mcp/src/elm_exp_manager.py::ELMExpManager._submit_job_b"]),
    ("Follow the chains and collect what each run produced",
     ["mcp/elm-mcp/src/elm_exp_manager.py::ELMExpManager._poll_experiment",
      "mcp/elm-mcp/src/elm_exp_manager.py::ELMExpManager._collect_experiment"]),
    ("Extract each run's year numbers, and the regional numbers computed by "
     "the server's code (job B)",
     ["mcp/elm-mcp/src/elm_exp_manager.py::ELMExpManager._extract_experiment",
      "mcp/elm-mcp/src/elm_exp_manager.py::ELMExpManager._regional",
      "mcp/elm-mcp/src/domain_analysis.py::experiment"]),
    ("Package the experiment record (job B)",
     ["mcp/elm-mcp/src/elm_exp_manager.py::ELMExpManager._package_experiment"]),
    ("The numbers report, rendered from the records (job B)",
     ["src/agents/analysis/domain_report.py::build"]),
    ("Analyzer step 0: read the records into one context (job B)",
     ["src/agents/analysis/step0_context.py::load"]),
    ("Analyzer steps 2 and 3: investigate, interpret, audit the claims (job B)",
     ["src/agents/analysis/step3_interpret.py::investigate_and_interpret",
      "src/agents/analysis/step3_interpret.py::audit"]),
    ("Analyzer step 4: the report (job B)",
     ["src/agents/analysis/step4_report.py::build",
      "src/agents/analysis/step4_report.py::write"]),
]

LLM_CLASSES = {"SimpleLLMClient", "LLMAgent", "ToolLoopAgent"}
SDK_CLIENTS = {("openai", "OpenAI"), ("openai", "AsyncOpenAI"),
               ("anthropic", "Anthropic"), ("anthropic", "AsyncAnthropic")}
RECORD_RE = re.compile(r"^[\w.*/{}<>-]*[\w*}>]\.(json|md|csv|log|patch|txt)$")
DASH_RE = re.compile(r"\s*[\u2014\u2013]\s*")


# ─────────────────────────────────────────────────────────────────────────────
# Reading the tree
# ─────────────────────────────────────────────────────────────────────────────
def _ignored(rels):
    """The subset of `rels` git ignores; empty when git is not available."""
    if not rels:
        return set()
    try:
        r = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT,
                           input="\n".join(rels), capture_output=True, text=True)
        return set(r.stdout.split())
    except OSError:
        return set()


def _walk(top, pattern="*.py"):
    base = ROOT / top
    if base.is_file():
        return [top]
    out = []
    for p in sorted(base.rglob(pattern)):
        rel = p.relative_to(ROOT).as_posix()
        if "__pycache__" in rel or "/_vendor/" in rel or "/." in rel:
            continue
        out.append(rel)
    return out


def _dash(text):
    """A quoted line without the dash used as a pause (see the module notes)."""
    text = re.sub(r"(?<=\d)\u2013(?=\d)", "-", text or "")
    return DASH_RE.sub("; ", text).strip()


def _first_line(doc):
    """A docstring's first line, carried across its wrap: lines are joined
    until one ends a sentence, the paragraph ends, or the next is a file path
    (the repo's module docstrings put their own path under the title)."""
    if not doc:
        return ""
    out = []
    for line in doc.strip().split("\n\n")[0].splitlines():
        line = line.strip()
        if out and re.match(r"^[\w./-]+\.py$", line):
            break
        out.append(line)
        if re.search(r"[.!?:]$", line):
            break
    return _dash(" ".join(out))


class Func:
    def __init__(self, mod, node, cls=None):
        self.mod, self.node, self.cls = mod, node, cls
        self.qual = f"{cls.name}.{node.name}" if cls else node.name
        self.key = f"{mod.rel}::{self.qual}"


class Cls:
    def __init__(self, mod, node):
        self.mod, self.node, self.name = mod, node, node.name
        self.key = f"{mod.rel}::{node.name}"
        self.methods = {}
        self.consts = {}
        self.const_strings = {}         # NAME -> every string in its value
        self.attr_types = {}            # self.<attr> = SomeClass(...)
        self.base_names = [b.id if isinstance(b, ast.Name) else
                           b.attr if isinstance(b, ast.Attribute) else None
                           for b in node.bases]


class Mod:
    def __init__(self, rel, kind):
        self.rel, self.kind = rel, kind
        self.path = ROOT / rel
        self.text = self.path.read_text(errors="ignore")
        self.tree = ast.parse(self.text, filename=rel)
        self.lines = self.text.splitlines()
        self.doc = ast.get_docstring(self.tree) or ""
        self.imports = {}               # alias -> ("mod", dotted) | ("sym", dotted, name)
        self.consts = {}                # NAME -> str
        self.const_strings = {}         # NAME -> every string in its value
        self.layout = {}                # logical name -> record file (run_layout)
        self.funcs, self.classes = {}, {}
        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs[node.name] = Func(self, node)
            elif isinstance(node, ast.ClassDef):
                c = Cls(self, node)
                self.classes[node.name] = c
                for b in node.body:
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        c.methods[b.name] = Func(self, b, c)
                    elif isinstance(b, (ast.Assign, ast.AnnAssign)):
                        if isinstance(b, ast.Assign):
                            c.consts.update(_str_bindings(b))
                        c.const_strings.update(_strings_bound(b))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                self.consts.update(_str_bindings(node))
                self.const_strings.update(_strings_bound(node))
                self.layout.update(_layout_bindings(node))
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.imports[a.asname or a.name.split(".")[0]] = (
                        ("mod", a.name) if a.asname else ("mod", a.name.split(".")[0]))
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    pkg = self.dotted.rsplit(".", node.level)[0] if self.dotted else ""
                    base = f"{pkg}.{base}".strip(".") if base else pkg
                for a in node.names:
                    self.imports[a.asname or a.name] = ("sym", base, a.name)
        for c in self.classes.values():
            for f in c.methods.values():
                for n in ast.walk(f.node):
                    if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                            and _callee_name(n.value.func)):
                        for t in n.targets:
                            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                                    and t.value.id == "self"):
                                c.attr_types[t.attr] = n.value.func

    @property
    def dotted(self):
        p = Path(self.rel)
        if p.parts[0] == "src":
            parts = list(p.with_suffix("").parts[1:])
            if parts[-1] == "__init__":
                parts = parts[:-1]
            return ".".join(parts)
        return p.stem if p.name != "__init__.py" else p.parent.name

    def segment(self, node):
        try:
            return ast.get_source_segment(self.text, node) or ""
        except Exception:                                       # noqa: BLE001
            return ""

    def all_funcs(self):
        yield from self.funcs.values()
        for c in self.classes.values():
            yield from c.methods.values()


def _str_value(v):
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
        return v.value
    if isinstance(v, ast.JoinedStr):
        return "".join(p.value if isinstance(p, ast.Constant) else "*" for p in v.values)
    return None


def _str_bindings(node):
    """NAME -> string for `NAME = "..."` and `A, B = "...", "..."`."""
    out = {}
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for t in targets:
        if isinstance(t, ast.Name) and node.value is not None:
            s = _str_value(node.value)
            if s is not None:
                out[t.id] = s
        elif isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
            for a, b in zip(t.elts, node.value.elts):
                s = _str_value(b)
                if isinstance(a, ast.Name) and s is not None:
                    out[a.id] = s
    return out


def _strings_bound(node):
    """NAME -> the strings a constant holds at its top level: a string
    (TOOL = "..."), the members of a set, list or tuple (RECEPTION_TOOLS =
    ("list_servers", ...)), or the values of a dict (DOMAIN_MENUS).

    ONE LEVEL ONLY (2026-11-03). A deeper string is a description, not a name
    the code acts on: a manager's FIELD_SEMANTICS says a field comes `from`
    build_conceptual_columns, which is no call to that tool."""
    v = node.value
    if v is None:
        return {}
    items = ([v] if isinstance(v, ast.Constant) else
             v.elts if isinstance(v, (ast.Set, ast.List, ast.Tuple)) else
             v.values if isinstance(v, ast.Dict) else [])
    strs = [n.value for n in items
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {t.id: strs for t in targets if isinstance(t, ast.Name) and strs}


def _layout_bindings(node):
    """{"reception": "reception.json", ...} from a module-level dict whose keys
    are names and whose values are a record path or a tuple of them, the first
    being where a run writes it (core/run_layout.LAYOUT)."""
    v = node.value
    if not isinstance(v, ast.Dict):
        return {}
    out = {}
    for k, val in zip(v.keys, v.values):
        first = val.elts[0] if isinstance(val, ast.Tuple) and val.elts else val
        s = _str_value(first)
        if (isinstance(k, ast.Constant) and isinstance(k.value, str) and s
                and RECORD_RE.match(s)):
            out[k.value] = s
    return out if len(out) >= 3 else {}


def _callee_name(f):
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


class Tree:
    """Every framework and server module, parsed once."""

    def __init__(self):
        fw = _walk("src") + _walk("workflow.py") + [
            r for r in _walk("mcp") if r.endswith("_exp_manager.py")]
        srv = [r for r in _walk("mcp") if not r.endswith("_exp_manager.py")]
        ign = _ignored(fw + srv)
        self.mods = {}
        for rel in fw:
            if rel not in ign:
                self.mods[rel] = Mod(rel, "framework")
        for rel in srv:
            if rel not in ign:
                self.mods[rel] = Mod(rel, "server")
        self.by_dotted = defaultdict(list)
        for m in self.mods.values():
            self.by_dotted[m.dotted].append(m)
        self.funcs = {f.key: f for m in self.mods.values() for f in m.all_funcs()}
        self.classes = {c.key: c for m in self.mods.values() for c in m.classes.values()}
        self._subs = defaultdict(set)
        for c in self.classes.values():
            for b in self.bases(c):
                self._subs[b.key].add(c.key)
        self.edges = {k: self._callees(f) for k, f in self.funcs.items()}
        # LOADERS BY PATH (2026-11-03). The names of the functions that load a
        # module from a file (spec_from_file_location), so a local bound to
        # `self._sibling_module("domain_case")` is known to be that module and
        # its constants (DC.CHAIN_FILE) are read like an imported module's.
        self.loaders = {f.node.name for f in self.funcs.values()
                        if any(isinstance(n, ast.Call)
                               and _callee_name(n.func) == "spec_from_file_location"
                               for n in ast.walk(f.node))}

    def loaded_module(self, mod, call):
        """The module a loader call returns, when it names one beside `mod`."""
        if not (isinstance(call, ast.Call) and _callee_name(call.func) in self.loaders
                and call.args and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)):
            return None
        return self.mods.get((Path(mod.rel).parent / f"{call.args[0].value}.py").as_posix())

    # ── name resolution ─────────────────────────────────────────────────────
    def module(self, dotted, near=None):
        cands = self.by_dotted.get(dotted) or []
        if len(cands) > 1 and near is not None:
            same = [m for m in cands if m.path.parent == near.path.parent]
            cands = same or cands
        return cands[0] if cands else None

    def resolve(self, mod, name):
        """What a bare name means in `mod`: a Func, Cls or Mod, or None."""
        if name in mod.funcs:
            return mod.funcs[name]
        if name in mod.classes:
            return mod.classes[name]
        imp = mod.imports.get(name)
        if not imp:
            return None
        if imp[0] == "mod":
            return self.module(imp[1], mod)
        sub = self.module(f"{imp[1]}.{imp[2]}" if imp[1] else imp[2], mod)
        if sub is not None:
            return sub
        target = self.module(imp[1], mod)
        if target is None:
            return None
        return target.funcs.get(imp[2]) or target.classes.get(imp[2])

    def bases(self, c):
        out = []
        for n in c.base_names:
            b = self.resolve(c.mod, n) if n else None
            if isinstance(b, Cls):
                out.append(b)
                out.extend(self.bases(b))
        return out

    def subclasses(self, c):
        out, todo = [], list(self._subs.get(c.key, ()))
        while todo:
            k = todo.pop()
            out.append(self.classes[k])
            todo.extend(self._subs.get(k, ()))
        return out

    def method(self, c, name, overrides=True):
        """`name` on class c: its own, its bases', and its subclasses' overrides."""
        found = []
        for k in [c] + self.bases(c):
            if name in k.methods:
                found.append(k.methods[name])
                break
        if overrides:
            found += [s.methods[name] for s in self.subclasses(c) if name in s.methods]
        return found

    def as_class(self, mod, expr):
        if isinstance(expr, ast.Name):
            r = self.resolve(mod, expr.id)
        elif isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
            m = self.resolve(mod, expr.value.id)
            r = (m.classes.get(expr.attr) if isinstance(m, Mod) else None)
        else:
            r = None
        return r if isinstance(r, Cls) else None

    def call_targets(self, f, call, local_types):
        """The Funcs a call may reach (constructors resolve to __init__)."""
        mod, fn = f.mod, call.func
        out = []

        def of(obj, attr=None):
            if isinstance(obj, Func) and attr is None:
                out.append(obj)
            elif isinstance(obj, Cls):
                out.extend(self.method(obj, attr or "__init__", overrides=attr is not None))
            elif isinstance(obj, Mod) and attr is not None:
                t = obj.funcs.get(attr) or obj.classes.get(attr)
                of(t)

        if isinstance(fn, ast.Name):
            of(self.resolve(mod, fn.id))
        elif isinstance(fn, ast.Attribute):
            v, attr = fn.value, fn.attr
            if isinstance(v, ast.Name) and v.id in ("self", "cls") and f.cls:
                out.extend(self.method(f.cls, attr))
            elif isinstance(v, ast.Name) and v.id in local_types:
                of(local_types[v.id], attr)
            elif isinstance(v, ast.Name):
                of(self.resolve(mod, v.id), attr)
            elif (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
                  and v.value.id == "self" and f.cls):
                for k in [f.cls] + self.bases(f.cls):
                    if v.attr in k.attr_types:
                        c = self.as_class(k.mod, k.attr_types[v.attr])
                        if c:
                            of(c, attr)
                        break
            elif isinstance(v, ast.Call):
                c = self.as_class(mod, v.func)
                if c:
                    of(c, attr)
        return [t for t in out if t is not None]

    def _callees(self, f):
        local_types = {}
        for n in ast.walk(f.node):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                c = self.as_class(f.mod, n.value.func)
                if c:
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            local_types[t.id] = c
        out = []
        for n in ast.walk(f.node):
            if isinstance(n, ast.Call):
                for t in self.call_targets(f, n, local_types):
                    if t.key != f.key and t.key not in out:
                        out.append(t.key)
        return out

    # ── language models ─────────────────────────────────────────────────────
    def llm_use(self, f):
        """("calls", ...) / ("holds", ...) facts about one function body."""
        calls, holds = [], []
        for n in ast.walk(f.node):
            if not isinstance(n, ast.Call):
                continue
            fn = n.func
            if isinstance(fn, ast.Attribute) and fn.attr == "ask":
                calls.append(f.mod.segment(fn))
            elif (isinstance(fn, ast.Attribute) and fn.attr == "create"
                  and isinstance(fn.value, ast.Attribute) and fn.value.attr == "completions"):
                calls.append(f.mod.segment(fn))
            name = _callee_name(fn)
            if name in LLM_CLASSES:
                tgt = self.resolve(f.mod, name) if isinstance(fn, ast.Name) else None
                if tgt is None or isinstance(tgt, Cls):
                    holds.append(name)
            if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                    and (fn.value.id, fn.attr) in SDK_CLIENTS):
                holds.append(f"{fn.value.id}.{fn.attr}")
        return calls, holds

    def llm_reach(self):
        """Every function from which a language-model call is reachable, with
        the next step on one shortest path to it."""
        roots = {k for k, f in self.funcs.items() if self.llm_use(f)[0]}
        rev = defaultdict(set)
        for k, cs in self.edges.items():
            for c in cs:
                rev[c].add(k)
        nxt = {k: None for k in roots}
        frontier = sorted(roots)
        while frontier:
            new = []
            for k in frontier:
                for caller in sorted(rev.get(k, ())):
                    if caller not in nxt:
                        nxt[caller] = k
                        new.append(caller)
            frontier = new
        return nxt


# ─────────────────────────────────────────────────────────────────────────────
# Records a function names, and which way
# ─────────────────────────────────────────────────────────────────────────────
def _units(node):
    """Leaf statements, and the header expressions of compound ones."""
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(n, (ast.If, ast.While)):
            yield n.test
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            yield n.iter
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for it in n.items:
                yield it.context_expr
        elif isinstance(n, ast.stmt) and not isinstance(
                n, (ast.Try, ast.If, ast.While, ast.For, ast.AsyncFor, ast.With,
                    ast.AsyncWith, getattr(ast, "TryStar", ast.Try), ast.Match)):
            yield n


def _record_names(tree, mod, node, aliases=None):
    """Record file names one AST node mentions, literally or by a constant.

    A constant is found in this module or its classes, in a module it imports
    (`X.CONST`, or `from x import CONST as ALIAS`), or in a module a loader
    loaded by path into a local (`aliases`: DC = self._sibling_module(...))."""
    out = set()
    aliases = aliases or {}
    consts = dict(mod.consts)
    for c in mod.classes.values():
        consts.update(c.consts)
    for n in ast.walk(node):
        s = _str_value(n) if isinstance(n, (ast.Constant, ast.JoinedStr)) else None
        if s is None and isinstance(n, ast.Name) and n.id in consts:
            s = consts[n.id]
        if s is None and isinstance(n, ast.Name) and n.id in mod.imports:
            imp = mod.imports[n.id]
            src = tree.module(imp[1], mod) if imp[0] == "sym" else None
            s = src.consts.get(imp[2]) if src is not None else None
        if s is None and isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Name) and n.value.id in aliases:
                s = aliases[n.value.id].consts.get(n.attr)
            elif isinstance(n.value, ast.Name) and n.value.id in ("self", "cls"):
                s = consts.get(n.attr)
            elif isinstance(n.value, ast.Name):
                m = tree.resolve(mod, n.value.id)
                if isinstance(m, Mod):
                    s = m.consts.get(n.attr)
            elif isinstance(n.value, ast.Call):
                m = tree.loaded_module(mod, n.value)
                s = m.consts.get(n.attr) if m is not None else None
        if s and RECORD_RE.match(s.strip()):
            out.add(s.strip().rsplit("/", 1)[-1])
        if isinstance(n, ast.Call):
            # a record asked for by its logical name: run_layout.resolve(rd, "strategy")
            fn, target = n.func, None
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                target = tree.resolve(mod, fn.value.id)
            elif isinstance(fn, ast.Name):
                r = tree.resolve(mod, fn.id)
                target = r.mod if isinstance(r, Func) else None
            if isinstance(target, Mod) and target.layout:
                for a in n.args:
                    if isinstance(a, ast.Constant) and a.value in target.layout:
                        out.add(target.layout[a.value].rsplit("/", 1)[-1])
    return out


WRITE_ATTRS = {"write_text", "write_bytes", "savefig", "to_csv", "to_netcdf", "write"}
READ_ATTRS = {"read_text", "read_bytes", "exists", "is_file"}
READ_NAME = re.compile(r"^_?(read|load)\w*$")


def _io_targets(call):
    """("write"|"read", [the nodes naming the file]) for one call, or None.

    The file is the call's receiver (p.write_text, p.exists), the first argument
    of open() or of a reader (_read(p), load_json(p)), or json.dump's second."""
    fn = call.func
    name = _callee_name(fn)
    if isinstance(fn, ast.Attribute) and name in WRITE_ATTRS:
        return "write", [fn.value] + (call.args[:1] if name == "savefig" else [])
    if isinstance(fn, ast.Attribute) and name in READ_ATTRS:
        return "read", [fn.value]
    if name == "open" and call.args:
        mode = call.args[1] if len(call.args) > 1 else next(
            (k.value for k in call.keywords if k.arg == "mode"), None)
        m = mode.value if isinstance(mode, ast.Constant) else ""
        return ("write" if any(c in str(m) for c in "wax") else "read"), call.args[:1]
    if name == "dump" and len(call.args) > 1:
        return "write", call.args[1:2]
    if name in ("copy", "copy2", "copyfile") and len(call.args) > 1:
        return "write", call.args[1:2]
    if name and READ_NAME.match(name) and name not in ("loads",) and call.args:
        return "read", call.args[:2]
    return None


def records_of(tree, f):
    """{"reads": set, "writes": set, "names": set} for one function body.

    A record is read or written when it is named inside the file argument of a
    reading or writing call (see _io_targets), directly or through a local
    name bound to it earlier (p = run_dir / "x.json"). A record named anywhere
    else (a value inside a dict, a message) is only named."""
    mod = f.mod
    got = {"reads": set(), "writes": set(), "names": set()}
    bound = {}                          # local name -> records it was built from
    aliases = {}                        # local name -> module loaded by path
    loop_target = {}                    # id(a for loop's iter) -> its target
    for n in ast.walk(f.node):
        if isinstance(n, ast.Assign):
            m = tree.loaded_module(mod, n.value)
            if m is not None:
                aliases.update({t.id: m for t in n.targets if isinstance(t, ast.Name)})
        elif isinstance(n, (ast.For, ast.AsyncFor)) and isinstance(n.target, ast.Name):
            loop_target[id(n.iter)] = n.target.id

    def recs_in(node):
        r = _record_names(tree, mod, node, aliases)
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in bound:
                r |= bound[n.id]
        return r

    units = sorted(_units(f.node), key=lambda u: (u.lineno, u.col_offset))
    for u in units:
        allrecs = recs_in(u)
        if not allrecs:
            continue
        hit = set()
        for c in ast.walk(u):
            if not isinstance(c, ast.Call):
                continue
            io = _io_targets(c)
            if not io:
                continue
            for t in io[1]:
                r = recs_in(t)
                got["writes" if io[0] == "write" else "reads"] |= r
                hit |= r
        rest = allrecs - hit
        if rest and isinstance(u, ast.Assign) and not hit:
            for t in u.targets:
                if isinstance(t, ast.Name):
                    bound[t.id] = bound.get(t.id, set()) | rest
        # A LOOP OVER RECORD NAMES binds its variable the way `p = ...` does
        # (2026-11-03): `for name in ("RUN_SUMMARY.json", ...)` then
        # `p = run_dir / name; p.exists()` reads each of them.
        if rest and not hit and id(u) in loop_target:
            v = loop_target[id(u)]
            bound[v] = bound.get(v, set()) | rest
        got["names"] |= rest
    got["names"] -= got["reads"] | got["writes"]
    return got


def _tool_of(v, tools):
    """The tool a string names, bare or as server__tool, or None."""
    if v in tools:
        return v
    if "__" in v and v.split("__", 1)[1] in tools:
        return v.split("__", 1)[1]
    return None


def prompt_names_tool(txt, tool):
    """Does a prompt's text name this tool? As a word, or as server__tool:
    `_` is a word character, so there is no word boundary between `__` and
    the name."""
    return bool(re.search(rf"(?:\b|__){re.escape(tool)}\b", txt))


def tools_named(f, tools):
    """MCP tool names this function names as a whole string literal, in its
    body or in a module or class constant it uses.

    A CONSTANT COUNTS FOR THE FUNCTION THAT USES IT (2026-11-03): sweep_menu
    holds its tool in TOOL and fetch() calls call_tool_json(TOOL, {});
    Reception's tools are RECEPTION_TOOLS, used by reception_tools()."""
    out = set()
    mod = f.mod
    for n in ast.walk(f.node):
        strs = []
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            strs = [n.value]
        elif isinstance(n, ast.Name) and n.id in mod.const_strings:
            strs = mod.const_strings[n.id]
        elif (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and f.cls
              and n.value.id in ("self", "cls", f.cls.name)
              and n.attr in f.cls.const_strings):
            strs = f.cls.const_strings[n.attr]
        for v in strs:
            t = _tool_of(v, tools)
            if t:
                out.add(t)
    return out


def unused_tool_constants(mod, tools):
    """{tool: [NAME, ...]} for module or class constants naming a tool that no
    function of the module uses; such a constant is credited to the module."""
    used = set()
    for f in mod.all_funcs():
        for n in ast.walk(f.node):
            if isinstance(n, ast.Name):
                used.add(n.id)
            elif isinstance(n, ast.Attribute):
                used.add(n.attr)
    out = defaultdict(list)
    tables = [mod.const_strings] + [c.const_strings for c in mod.classes.values()]
    for table in tables:
        for name, strs in table.items():
            if name in used:
                continue
            for v in strs:
                t = _tool_of(v, tools)
                if t and name not in out[t]:
                    out[t].append(name)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MCP servers
# ─────────────────────────────────────────────────────────────────────────────
def registry():
    d = json.loads((ROOT / REGISTRY).read_text())
    return d.get("mcp_servers") or {}


def _entry_rel(spec):
    for a in spec.get("args") or []:
        if a.endswith(".py"):
            tail = a.split("multi-agent-framework/", 1)[-1]
            if (ROOT / tail).is_file():
                return tail
    return None


def served_tools(tree, rel):
    """[(name, what it does)] for each tool a server file serves.

    Two forms are read. FastMCP's `@mcp.tool()` on a function: its name and the
    first line of its docstring. The low-level `Server` API (daymet, geology,
    weather): each `types.Tool(name=..., description=...)` returned by the
    function decorated `@server.list_tools()`, with the description's first
    sentence."""
    m = tree.mods.get(rel)
    if m is None:
        return []
    out = []
    for node in m.tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            fn = d.func if isinstance(d, ast.Call) else d
            if isinstance(fn, ast.Attribute) and fn.attr == "tool":
                name = node.name
                if isinstance(d, ast.Call):
                    for kw in d.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            name = kw.value.value
                out.append((name, _first_line(ast.get_docstring(node))))
                break
            if isinstance(fn, ast.Attribute) and fn.attr == "list_tools":
                for c in ast.walk(node):
                    if not (isinstance(c, ast.Call) and _callee_name(c.func) == "Tool"):
                        continue
                    kws = {k.arg: k.value for k in c.keywords}
                    name = kws.get("name")
                    if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
                        continue
                    desc = _str_value(kws.get("description")) or ""
                    first = re.split(r"(?<=[.!?])\s(?=[A-Z(])", " ".join(desc.split()),
                                     maxsplit=1)[0]
                    out.append((name.value, _dash(first)))
                break
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────
def _cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def _code_list(items, empty="none"):
    items = sorted(set(items))
    return ", ".join(f"`{i}`" for i in items) if items else empty


def _comment_blocks(mod, start, end):
    """Comment paragraphs between two line numbers (1-based, inclusive)."""
    blocks, cur = [], []
    for i in range(max(start, 1), min(end, len(mod.lines)) + 1):
        s = mod.lines[i - 1].strip()
        body = s.lstrip("#").strip()
        if s.startswith("#") and body:
            cur.append(body)
        else:
            if cur:
                blocks.append(" ".join(cur))
            cur = []
    if cur:
        blocks.append(" ".join(x for x in cur if x))
    return [b for b in blocks if b]


WHY_RE = re.compile(r"(?i:by path|in-process|in process|tool call)|\bMCP\b")


def _why_sentences(text):
    """The sentences of `text` that say why something is loaded in-process,
    narrowed to the parenthesis that holds the phrase when a sentence is long."""
    out = []
    flat = " ".join((text or "").split())
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", flat):
        m = WHY_RE.search(sent)
        if not m:
            continue
        if len(sent) > 300:
            a = sent.rfind("(", 0, m.start())
            b = sent.find(")", m.end())
            if a >= 0 and b > a:
                sent = sent[a + 1:b]
        out.append(_dash(sent))
    return out


def _comment_above(mod, line, lo, n=2):
    """Up to `n` comment blocks above `line`, nearest first, not above `lo`."""
    found, i = [], line - 1
    while i > lo and len(found) < n and line - i <= 15:
        if mod.lines[i - 1].strip().startswith("#"):
            j = i
            while j - 1 > lo and mod.lines[j - 2].strip().startswith("#"):
                j -= 1
            found += _comment_blocks(mod, j, i)[::-1]
            i = j - 1
        else:
            i -= 1
    return [_dash(b) for b in found[:n]][::-1]


def _reason(mod, node, prev_end, at_line=None):
    """Why code loads a server's module in-process, in the code's own words:
    what the function is (its docstring's first line), the comment paragraphs
    above it and the docstring sentences that speak of loading by path,
    in-process or MCP, and the comments just above the loading statement."""
    parts = []
    doc = ast.get_docstring(node) or ""
    if doc:
        parts.append(_first_line(doc))
    for b in _comment_blocks(mod, prev_end + 1, node.lineno - 1):
        if WHY_RE.search(b):
            parts.append(_dash(b))
    parts += _why_sentences(doc.split("\n\n", 1)[1] if "\n\n" in doc else "")
    if at_line:
        parts += _comment_above(mod, at_line, node.lineno)
    seen, out = set(), []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p if re.search(r"[.!?:)]$", p) else p + ".")
    return " ".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# The sections
# ─────────────────────────────────────────────────────────────────────────────
def _model_default(tree, f):
    """What a function's `model` parameter defaults to, resolved."""
    a = f.node.args
    params = a.args + a.kwonlyargs
    defaults = ([None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
                + list(a.kw_defaults))
    for p, d in zip(params, defaults):
        if p.arg == "model":
            return NO_DEFAULT if d is None else _render_expr(tree, f, d)
    return None


NO_DEFAULT = "no default"


def _render_expr(tree, f, e, depth=0):
    """An expression as written, with what it comes to when the code says."""
    text = " ".join((f.mod.segment(e) or ast.unparse(e)).split())
    val = _const_value(tree, f, e, depth)
    if val is None or val == text.strip("'\""):
        return f"`{text}`" if val is None else f'`"{val}"`'
    return f'`{text}` = `"{val}"`'


def _const_value(tree, f, e, depth=0):
    if depth > 4:
        return None
    if isinstance(e, ast.Constant) and isinstance(e.value, str):
        return e.value
    if isinstance(e, ast.Name):
        if e.id in f.mod.consts:
            return f.mod.consts[e.id]
        a = f.node.args
        params = a.args + a.kwonlyargs
        defaults = ([None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
                    + list(a.kw_defaults))
        for p, d in zip(params, defaults):
            if p.arg == e.id and d is not None:
                return _const_value(tree, f, d, depth + 1)
        return None
    if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name):
        m = tree.resolve(f.mod, e.value.id)
        if isinstance(m, Mod):
            return m.consts.get(e.attr)
        # a command-line option: add_argument("--model", default="...")
        for n in ast.walk(f.mod.tree):
            if (isinstance(n, ast.Call) and _callee_name(n.func) == "add_argument"
                    and any(isinstance(a, ast.Constant) and a.value == f"--{e.attr}"
                            for a in n.args)):
                d = next((k.value for k in n.keywords if k.arg == "default"), None)
                return _str_value(d) if d is not None else None
    if (isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute)
            and e.func.attr == "get" and len(e.args) == 2):
        return _const_value(tree, f, e.args[1], depth + 1)
    return None


def _model_passes(tree, comp_mod, reach, kind="framework"):
    """Every call into this module from `kind` code that passes model=..."""
    out = []
    for k, f in sorted(tree.funcs.items()):
        if f.mod.kind != kind:
            continue
        local_types = {}
        for n in ast.walk(f.node):
            if not isinstance(n, ast.Call):
                continue
            kw = [k_ for k_ in n.keywords if k_.arg == "model"]
            if not kw:
                continue
            tg = [t for t in tree.call_targets(f, n, local_types) if t.mod is comp_mod]
            if not tg:
                continue
            out.append(f"`{f.mod.rel}::{f.qual}` calls `{tg[0].qual}` with "
                       f"model={_render_expr(tree, f, kw[0].value)}")
    return sorted(set(out))


def section_llm(tree, reach, lines):
    lines += ["## 1. Components that call a language model", "",
              "A component is listed when one of its functions calls a model "
              "(`.ask(...)` or `...completions.create(...)`) or constructs a model "
              "client. The client itself is in section 2. \"Called from\" lists the "
              "framework functions outside the module whose calls reach a model "
              "through it, and then any code under `mcp/` (a server's own script) "
              "that does.", ""]
    client_mod = "src/agents/llm_agent.py"
    for rel, m in sorted(tree.mods.items()):
        if m.kind != "framework" or rel == client_mod:
            continue
        uses = {f.qual: tree.llm_use(f) for f in m.all_funcs()}
        uses = {q: u for q, u in uses.items() if u[0] or u[1]}
        if not uses:
            continue
        lines += [f"### `{rel}`", ""]
        if m.doc:
            lines += [f"{_first_line(m.doc)}", ""]
        for q in sorted(uses):
            calls, holds = uses[q]
            what = []
            if calls:
                what.append("calls the model (" + ", ".join(
                    f"`{c}`" for c in sorted(set(calls))) + ")")
            if holds:
                what.append("constructs " + _code_list(holds))
            lines.append(f"- `{q}` {' and '.join(what)}")
        defaults = []
        if "DEFAULT_MODEL" in m.consts:
            defaults.append(f'`DEFAULT_MODEL = "{m.consts["DEFAULT_MODEL"]}"`')
        seen = set()
        for f in m.all_funcs():
            d = _model_default(tree, f)
            if d and (f.qual in uses or (f.cls and f.node.name == "__init__"
                                         and any(q.startswith(f.cls.name + ".") for q in uses))):
                if f.qual not in seen:
                    seen.add(f.qual)
                    defaults.append(f"`{f.qual}(model)` has no default; its caller sets it"
                                    if d == NO_DEFAULT else
                                    f"`{f.qual}(model)` defaults to {d}")
        lines.append("- Default model id: " + ("; ".join(defaults) if defaults else
                                               "set nowhere in this module"))
        passes = _model_passes(tree, m, reach)
        if passes:
            lines.append("- Model id as the framework passes it:")
            lines += [f"  - {p}" for p in passes]
        prompts = set()
        for n in ast.walk(m.tree):
            if (isinstance(n, ast.Call) and _callee_name(n.func) == "load_prompt"
                    and n.args and isinstance(n.args[0], ast.Constant)):
                prompts.add((PROMPTS / f"{n.args[0].value}.txt").as_posix())
        inline = set()
        # the functions that reach the model, and the helpers they call in this
        # module (step 2's task_for, which picks TASK or DOMAIN_TASK)
        reaching = {f.key for f in m.all_funcs() if f.key in reach}
        todo = list(reaching)
        while todo:
            for c in tree.edges.get(todo.pop(), ()):
                if c not in reaching and tree.funcs[c].mod is m:
                    reaching.add(c)
                    todo.append(c)
        for f in m.all_funcs():
            if f.key not in reaching:
                continue
            for n in ast.walk(f.node):
                if (isinstance(n, ast.Name) and n.id.isupper() and n.id in m.consts
                        and len(m.consts[n.id]) >= 200):
                    inline.add(n.id)
                if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                        and n.value.id == "self" and f.cls and n.attr.isupper()
                        and len(f.cls.consts.get(n.attr, "")) >= 200):
                    inline.add(n.attr)
        lines.append("- Prompt files: " + _code_list(prompts))
        lines.append("- Prompt text written in the module: " + _code_list(inline))
        recs = {"reads": set(), "writes": set(), "names": set()}
        for f in m.all_funcs():
            for k_, v in records_of(tree, f).items():
                recs[k_] |= v
        recs["names"] -= recs["reads"] | recs["writes"]
        lines.append(f"- Records it reads: {_code_list(recs['reads'])}; writes: "
                     f"{_code_list(recs['writes'])}; names without opening: "
                     f"{_code_list(recs['names'])}")
        callers = {"framework": set(), "server": set()}
        for k, cs in tree.edges.items():
            f = tree.funcs[k]
            if f.mod is m:
                continue
            if any(c in reach and tree.funcs[c].mod is m for c in cs):
                callers[f.mod.kind].add(f"{f.mod.rel}::{f.qual}")
        lines.append("- Called from: " + _code_list(callers["framework"],
                                                    "no framework function"))
        # SCRIPTS UNDER mcp/ COUNT TOO (2026-11-03). A server's own script
        # (mcp/elm-mcp/scripts/analyze_agentic.py) may drive an agent defined
        # here; "no framework function" alone would read as "unused".
        if callers["server"]:
            lines.append("- Called from (code under `mcp/`): "
                         + _code_list(callers["server"]))
            for p in _model_passes(tree, m, reach, kind="server"):
                lines.append(f"  - {p}")
        lines.append("")


def section_client(tree, lines):
    lines += ["## 2. The language-model endpoint and client", "",
              "Every construction of a model SDK client in framework code, with the "
              "endpoint it is given and the environment variable its key is read from "
              "(the variable's name only).", ""]
    rows = []
    for rel, m in sorted(tree.mods.items()):
        if m.kind != "framework":
            continue
        for f in m.all_funcs():
            for n in ast.walk(f.node):
                if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and isinstance(n.func.value, ast.Name)
                        and (n.func.value.id, n.func.attr) in SDK_CLIENTS):
                    continue
                url = next((kw.value.value for kw in n.keywords if kw.arg == "base_url"
                            and isinstance(kw.value, ast.Constant)), "the SDK's default")
                env = sorted({c.args[0].value for c in ast.walk(f.node)
                              if isinstance(c, ast.Call) and _callee_name(c.func) in
                              ("getenv", "get") and c.args
                              and isinstance(c.args[0], ast.Constant)
                              and isinstance(c.args[0].value, str)
                              and c.args[0].value.isupper()})
                d = _model_default(tree, f)
                rows.append((rel, f.qual, f"{n.func.value.id}.{n.func.attr}", url,
                             _code_list(env), d if d and d != NO_DEFAULT else "none"))
    lines += ["| where | function | SDK client | endpoint | key read from | default model id |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r[0]}` | `{r[1]}` | `{r[2]}` | `{r[3]}` | {r[4]} | {r[5]} |")
    m = tree.mods.get("src/agents/llm_agent.py")
    if m:
        lines += ["", "The classes in `src/agents/llm_agent.py`:", ""]
        for c in m.classes.values():
            lines.append(f"- `{c.name}`: {_first_line(ast.get_docstring(c.node))}")
    sites = _reply_limits(tree)
    if sites:
        lines += ["", "The reply limit (`max_tokens`) at each model call in framework "
                  "code, as the code sets it:", ""]
        lines += [f"- {s}" for s in sites]
    lines.append("")


def _param_default(f, name):
    """The default of parameter `name` of f, as a node, or None."""
    a = f.node.args
    params = a.args + a.kwonlyargs
    defaults = ([None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
                + list(a.kw_defaults))
    return next((d for p, d in zip(params, defaults) if p.arg == name), None)


def _limit_value(tree, f, e):
    """(value, where it is set) for a max_tokens expression, or (None, None).

    A number is itself; a parameter is its default; `self.max_tokens` is what
    the class (or a base) assigns to it, a number or a parameter's default."""
    if isinstance(e, ast.Constant) and isinstance(e.value, int):
        return e.value, None
    if isinstance(e, ast.Name):
        d = _param_default(f, e.id)
        if isinstance(d, ast.Constant):
            return d.value, f"the default of `{f.qual}({e.id})`"
        return None, None
    if (isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name)
            and e.value.id == "self" and f.cls):
        for k in [f.cls] + tree.bases(f.cls):
            for g in k.methods.values():
                for n in ast.walk(g.node):
                    if (isinstance(n, ast.Assign) and len(n.targets) == 1
                            and isinstance(n.targets[0], ast.Attribute)
                            and n.targets[0].attr == e.attr
                            and isinstance(n.targets[0].value, ast.Name)
                            and n.targets[0].value.id == "self"):
                        if isinstance(n.value, ast.Constant):
                            return n.value.value, f"set in `{g.qual}`"
                        if isinstance(n.value, ast.Name):
                            d = _param_default(g, n.value.id)
                            if isinstance(d, ast.Constant):
                                return d.value, f"the default of `{g.qual}({n.value.id})`"
    return None, None


def _reply_limits(tree):
    """One line per `...completions.create(...)` call in framework code: the
    max_tokens it sends (its keyword, or a `kwargs["max_tokens"] = ...` the
    function sets before `**kwargs`), resolved, with the framework calls that
    pass a different value into the parameter or constructor it comes from.

    PER CALL, NOT PER CLIENT (2026-11-03). SimpleLLMClient.ask sends its own
    limit, but Reception's tool loop and the agentic Analyzer call the SDK
    directly with theirs, so one number for every client would be wrong."""
    out = []
    for f in sorted(tree.funcs.values(), key=lambda x: x.key):
        if f.mod.kind != "framework":
            continue
        for n in ast.walk(f.node):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "create" and isinstance(n.func.value, ast.Attribute)
                    and n.func.value.attr == "completions"):
                continue
            e = next((k.value for k in n.keywords if k.arg == "max_tokens"), None)
            if e is None:
                spread = {k.value.id for k in n.keywords
                          if k.arg is None and isinstance(k.value, ast.Name)}
                e = next((a.value for a in ast.walk(f.node)
                          if isinstance(a, ast.Assign) and len(a.targets) == 1
                          and isinstance(a.targets[0], ast.Subscript)
                          and isinstance(a.targets[0].value, ast.Name)
                          and a.targets[0].value.id in spread
                          and isinstance(a.targets[0].slice, ast.Constant)
                          and a.targets[0].slice.value == "max_tokens"), None)
            if e is None:
                out.append(f"`{f.key}`: none sent (the endpoint's own limit)")
                continue
            val, where = _limit_value(tree, f, e)
            text = " ".join(f.mod.segment(e).split())
            line = f"`{f.key}`: `{text}`" if val is None else (
                f"`{f.key}`: {val}" + (f" (`{text}`, {where})" if where else ""))
            line += _limit_overrides(tree, f, e)
            out.append(line)
    return sorted(set(out))


def _limit_overrides(tree, f, e):
    """", callers pass ..." for framework calls that set the parameter the
    limit defaults from (`_chat(..., max_tokens=1600)`, `ToolLoopAgent(...,
    max_tokens=...)`)."""
    owner, param = None, None
    if isinstance(e, ast.Name) and _param_default(f, e.id) is not None:
        owner, param = f, e.id
    elif isinstance(e, ast.Attribute) and f.cls:
        for g in f.cls.methods.values():
            for n in ast.walk(g.node):
                if (isinstance(n, ast.Assign) and len(n.targets) == 1
                        and isinstance(n.targets[0], ast.Attribute)
                        and n.targets[0].attr == e.attr and isinstance(n.value, ast.Name)):
                    owner, param = g, n.value.id
    if owner is None:
        return ""
    passes = set()
    for g in tree.funcs.values():
        if g.mod.kind != "framework":
            continue
        for n in ast.walk(g.node):
            if not isinstance(n, ast.Call):
                continue
            kw = next((k.value for k in n.keywords if k.arg == param), None)
            if kw is None or g.key == owner.key:
                continue
            if any(t.key == owner.key for t in tree.call_targets(g, n, {})):
                passes.add(f"`{g.qual}` passes `{' '.join(g.mod.segment(kw).split())}`")
    return ("; " + ", ".join(sorted(passes))) if passes else ""


def section_mcp(tree, lines):
    reg = registry()
    srv_dirs = sorted({Path(r).parts[1] for r in tree.mods if r.startswith("mcp/")})
    fw = [f for f in tree.funcs.values() if f.mod.kind == "framework"]
    fw_mods = [m for m in tree.mods.values() if m.kind == "framework"]
    prompts = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / PROMPTS).glob("*.txt"))
    prompt_text = {p: (ROOT / p).read_text(errors="ignore") for p in prompts}

    # A tool found by a name pattern (model_servers' describe_*_capabilities):
    # a module-level re.compile in a module that calls tools, matched in full.
    patterns = []
    for m in tree.mods.values():
        if m.kind != "framework" or "call_tool_json" not in m.text:
            continue
        for nd in m.tree.body:
            if (isinstance(nd, ast.Assign) and isinstance(nd.value, ast.Call)
                    and _callee_name(nd.value.func) == "compile" and nd.value.args
                    and isinstance(nd.value.args[0], ast.Constant)
                    and isinstance(nd.targets[0], ast.Name)):
                name, pat = nd.targets[0].id, nd.value.args[0].value
                users = [f for f in m.all_funcs() if any(
                    isinstance(x, ast.Name) and x.id == name for x in ast.walk(f.node))]
                patterns.append((re.compile(pat), name, users))

    def callers(tools):
        out = defaultdict(set)
        for f in fw:
            for t in tools_named(f, tools):
                out[t].add(f"{f.mod.rel}::{f.qual}")
        for m in fw_mods:
            for t, names in unused_tool_constants(m, tools).items():
                out[t].add(f"{m.rel} (the constant {', '.join(sorted(names))})")
        for rx, name, users in patterns:
            for t in tools:
                if rx.fullmatch(t):
                    for f in users:
                        out[t].add(f"{f.mod.rel}::{f.qual} (by the pattern {name})")
        for p, txt in prompt_text.items():
            for t in tools:
                if prompt_names_tool(txt, t):
                    out[t].add(f"{p} (prompt)")
        return out

    lines += ["## 3. MCP servers and the tools they serve", "",
              f"The registry the framework reads is `{REGISTRY}` (copied to the "
              "git-ignored `mcp_config.json` on each machine). A tool's callers are "
              "the framework functions that name it as a whole string literal, bare "
              "or as `server__tool`, in their body or in a module or class constant "
              "they use (a constant no function uses is credited to its module), and "
              "the prompts that name it. A tool with no caller may still be reached "
              "by a language model choosing it from a server's tool list.", ""]
    in_registry = set()
    for name in sorted(reg):
        spec = reg[name]
        rel = _entry_rel(spec)
        lines += [f"### `{name}`", ""]
        desc = _dash(spec.get("description") or "")
        if desc:
            lines += [f"Registry description: {desc}", ""]
        if rel is None:
            cmd = Path(spec.get("command") or "").name
            lines += [f"Entry: the command `{cmd}`. Its arguments name no server file "
                      "in this repository, so its tools are not read here.", ""]
            continue
        in_registry.add(Path(rel).parts[1])
        tools = served_tools(tree, rel)
        by = callers({t for t, _ in tools})
        lines += [f"Entry: `{rel}`, {len(tools)} tool{'' if len(tools) == 1 else 's'}.", "",
                  "| tool | what it does (its docstring's first line, or its description's "
                  "first sentence) | called from |",
                  "|---|---|---|"]
        for t, doc in tools:
            lines.append(f"| `{t}` | {_cell(doc)} | {_cell(_code_list(by.get(t, ()), 'no framework caller'))} |")
        lines.append("")
    # Tools the framework names that no server in this repository serves.
    known_tools = set()
    for d in srv_dirs:
        for t, _ in served_tools(tree, f"mcp/{d}/main.py"):
            known_tools.add(t)
    called = defaultdict(set)
    for f in fw:
        if not f.mod.rel.startswith("mcp/"):
            continue
        for n in ast.walk(f.node):
            if (isinstance(n, ast.Call) and _callee_name(n.func) in ("_mcp_call", "call_tool_json")
                    and len(n.args) >= 1):
                a = n.args[1] if _callee_name(n.func) == "_mcp_call" and len(n.args) > 1 else n.args[0]
                if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                        and a.value not in known_tools:
                    called[a.value].add(f"{f.mod.rel}::{f.qual}")
    if called:
        lines += ["### Tools called on a server whose source is not in this repository", "",
                  "| tool | called from |", "|---|---|"]
        for t in sorted(called):
            lines.append(f"| `{t}` | {_cell(_code_list(called[t]))} |")
        lines.append("")
    unreg = [d for d in srv_dirs if d not in in_registry and (ROOT / "mcp" / d / "main.py").is_file()]
    if unreg:
        lines += ["### Server directories the registry does not register", "",
                  "| directory | tools | framework callers |", "|---|---|---|"]
        for d in unreg:
            tools = served_tools(tree, f"mcp/{d}/main.py")
            by = callers({t for t, _ in tools})
            n_called = sorted({c for t in by for c in by[t]})
            lines.append(f"| `mcp/{d}` | {_code_list(t for t, _ in tools)} | "
                         f"{_cell(_code_list(n_called, 'none'))} |")
        lines.append("")


def _server_dir(rel):
    p = Path(rel)
    return p.parts[1] if p.parts[0] == "mcp" else None


def _prev_end(f):
    """The last line of whatever precedes this function in its file or class."""
    body = f.cls.node.body if f.cls else f.mod.tree.body
    end = f.cls.node.lineno if f.cls else 0
    for nd in body:
        if nd is f.node:
            break
        if isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(nd, "end_lineno", nd.lineno)
    return end


def section_direct(tree, lines):
    lines += ["## 4. Where framework code reaches a server without MCP", "",
              "Framework code that loads a module from a server's directory in its own "
              "process instead of calling one of the server's tools: each function that "
              "loads a module from a file path (`spec_from_file_location`) where the path "
              "or the caller is under `mcp/`, and each import of a module that lives "
              "only under `mcp/` from outside that module's directory. The reason is "
              "quoted from the code: the function's docstring, the comments that speak "
              "of loading by path, in-process or MCP, and the comments just above the "
              "loading statement.", ""]
    rows = []
    for L in sorted(tree.funcs.values(), key=lambda x: x.key):
        if L.mod.kind != "framework":
            continue
        specs = [n for n in ast.walk(L.node) if isinstance(n, ast.Call)
                 and _callee_name(n.func) == "spec_from_file_location"]
        if not specs:
            continue
        used = {n.id for n in ast.walk(L.node) if isinstance(n, ast.Name)}
        consts = [nd for nd in L.mod.tree.body if isinstance(nd, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in used for t in nd.targets)]
        body = L.mod.segment(L.node) + " ".join(L.mod.segment(c) for c in consts)
        sites = []                      # callers that name the module they load
        for f in tree.funcs.values():
            if f.mod.kind != "framework":
                continue
            bound = {}
            for n in ast.walk(f.node):
                if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                        and _callee_name(n.value.func) == L.node.name and n.value.args
                        and isinstance(n.value.args[0], ast.Constant)):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            bound[t.id] = n.value.args[0].value
            for n in ast.walk(f.node):
                if (isinstance(n, ast.Call) and _callee_name(n.func) == L.node.name
                        and n.args and isinstance(n.args[0], ast.Constant)):
                    arg = n.args[0].value
                    attrs = sorted({a.attr for a in ast.walk(f.node)
                                    if isinstance(a, ast.Attribute)
                                    and isinstance(a.value, ast.Name)
                                    and bound.get(a.value.id) == arg})
                    sites.append((f, arg, attrs))
        # A loader crosses into a server when its own path names mcp/, or when
        # every call site sits in a server directory (a manager loading what
        # lies beside it). tools/ loaders and the like are not crossings.
        crosses = ('"mcp"' in body or "mcp/" in body
                   or (sites and all(_server_dir(s[0].mod.rel) for s in sites)))
        if not crosses:
            continue
        loads, seen = [], set()
        for f, arg, attrs in sorted(sites, key=lambda s: (s[0].key, s[1])):
            if (f.key, arg) in seen:
                continue
            seen.add((f.key, arg))
            target = (f"`{Path(f.mod.rel).parent.as_posix()}/{arg}.py`"
                      if _server_dir(f.mod.rel) else f"`{arg}`")
            use = f" (uses {_code_list(attrs)})" if attrs else ""
            loads.append(f"`{f.mod.rel}::{f.qual}` loads {target}{use}")
        if not loads:
            paths = [" ".join(L.mod.segment(c).split()) for c in consts]
            paths = [p for p in paths if "mcp" in p and len(p) < 240]
            what = " ".join(L.mod.segment(specs[0]).split())
            loads = [f"`{what}`" + (f", with `{'`; `'.join(paths)}`" if paths else "")]
            for n in ast.walk(L.node):
                if isinstance(n, ast.Assign) and "mcp" in L.mod.segment(n) and \
                        any(isinstance(t, ast.Name) for t in n.targets):
                    loads.append(f"where `{' '.join(L.mod.segment(n).split())}`")
        rows.append((f"`{L.key}`", "<br>".join(loads),
                     _reason(L.mod, L.node, _prev_end(L), at_line=specs[0].lineno)))

    # Imports of a module that lives only under mcp/, from outside its directory.
    mcp_mods = defaultdict(set)
    for rel in tree.mods:
        if rel.startswith("mcp/"):
            p = Path(rel)
            mcp_mods[p.stem if p.name != "__init__.py" else p.parent.name].add(rel)
    src_tops = {m.dotted.split(".")[0] for m in tree.mods.values()
                if m.rel.startswith("src/") or m.rel == "workflow.py"}
    for f in sorted(tree.funcs.values(), key=lambda x: x.key):
        if f.mod.kind != "framework":
            continue
        for n in ast.walk(f.node):
            if isinstance(n, ast.ImportFrom) and not n.level and n.module:
                names = [(n.module.split(".")[0], ", ".join(a.name for a in n.names))]
            elif isinstance(n, ast.Import):
                names = [(a.name.split(".")[0], a.name) for a in n.names]
            else:
                continue
            for top, what in names:
                if top not in mcp_mods or top in src_tops:
                    continue
                where = sorted(mcp_mods[top])
                here = f.mod.path.parent.relative_to(ROOT).as_posix()
                if all(Path(w).parent.as_posix() == here for w in where):
                    continue
                rows.append((f"`{f.key}`",
                             f"imports `{what}` from `{top}` ({_code_list(where)})",
                             _reason(f.mod, f.node, _prev_end(f), at_line=n.lineno)))
    lines += ["| loader or importer | what it loads | the reason the code gives |",
              "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {_cell(r[1])} | {_cell(r[2])} |")
    lines.append("")


OPTIONAL_RE = re.compile(r"^#\s+([A-Za-z][\w.\-]*)\s+#(.*)$")


def _requirements():
    """Import name -> how requirements.txt lists it: `name`, or `optional:
    name` for a commented line of the optional block, reading `imported as X`.

    THE OPTIONAL BLOCK IS A LIST TOO (2026-11-03). requirements.txt names a few
    packages as commented lines (`# rasterio   # what it is for`) because one
    feature each needs them; they are listed on purpose, not missing."""
    out = {}
    p = ROOT / "requirements.txt"
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        opt = OPTIONAL_RE.match(line.strip())
        if opt:
            name, note, label = opt.group(1), opt.group(2), f"optional: `{opt.group(1)}`"
        else:
            s = line.split("#", 1)
            pkg = s[0].strip()
            if not pkg:
                continue
            name = re.split(r"[<>=!~\[ ]", pkg)[0]
            note = s[1] if len(s) > 1 else ""
            label = f"`{name}`"
        out.setdefault(name.lower(), label)
        m = re.search(r"imported as (\w+)", note)
        if m:
            out.setdefault(m.group(1).lower(), label)
    return out


def section_libs(tree, lines):
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    # A NAME IS LOCAL ONLY WHEN IT CAN BE IMPORTED FROM HERE (2026-11-03): a
    # module file, a package (a directory with __init__.py), or a directory
    # directly under a `src/` the code puts on sys.path (agents, core). Any
    # other directory name is not: the top-level `mcp/` holds the servers, and
    # `import mcp` is the mcp SDK every server and the client import.
    local = set()
    for rel in list(tree.mods) + _walk("tools") + _walk("tests") + _walk("eval"):
        p = Path(rel)
        local.add(p.stem)
        for i in range(1, len(p.parts) - 1):
            d = Path(*p.parts[:i + 1])
            if p.parts[i - 1] == "src" or (ROOT / d / "__init__.py").is_file():
                local.add(p.parts[i])
    local |= {m.dotted.split(".")[0] for m in tree.mods.values()}
    # a name a loader registers by path (compare_pflotran) is this repository's
    local |= {n.args[0].value for m in tree.mods.values() for n in ast.walk(m.tree)
              if isinstance(n, ast.Call) and _callee_name(n.func) == "spec_from_file_location"
              and n.args and isinstance(n.args[0], ast.Constant)
              and isinstance(n.args[0].value, str)}
    req = _requirements()
    use = defaultdict(lambda: {"framework": set(), "server": set()})
    for rel, m in tree.mods.items():
        for n in ast.walk(m.tree):
            tops = []
            if isinstance(n, ast.Import):
                tops = [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and not n.level and n.module:
                tops = [n.module.split(".")[0]]
            for t in tops:
                if t in stdlib or t in local:
                    continue
                label = rel if m.kind == "framework" else "mcp/" + _server_dir(rel)
                use[t][m.kind].add(label)
    lines += ["## 5. Third-party libraries", "",
              "Every top-level package imported by framework code (`src/`, `workflow.py`, "
              "the experiment managers) or by the servers under `mcp/`, that is neither "
              "the standard library nor a module of this repository. Imports inside a "
              "function count: many are deferred so a missing package fails only the "
              "code that needs it. `tools/`, `tests/` and `eval/` are not scanned.", "",
              "| package (import name) | in requirements.txt | framework modules | servers |",
              "|---|---|---|---|"]
    for t in sorted(use, key=str.lower):
        r = req.get(t.lower())
        lines.append(f"| `{t}` | {r or 'no'} | "
                     f"{_cell(_code_list(use[t]['framework']))} | "
                     f"{_cell(_code_list(use[t]['server']))} |")
    lines.append("")


def _path_to_llm(tree, reach, key):
    chain, k = [], key
    while k is not None and len(chain) < 12:
        chain.append(tree.funcs[k].qual)
        k = reach.get(k)
    return chain


def section_flow(tree, reach, lines):
    all_tools = set()
    for d in sorted({_server_dir(r) for r in tree.mods if r.startswith("mcp/")}):
        all_tools |= {t for t, _ in served_tools(tree, f"mcp/{d}/main.py")}
    lines += ["## 6. The whole-domain flow", "",
              "> Hand-written: the order of the stages, their titles and the functions "
              "named for each (`FLOW` in the generator; every function is checked to "
              "exist). Read from those functions: whether a language model is reached "
              "(with one call path to it), the MCP tools they name, and the records "
              "they read and write, counting the functions they call (two calls deep) "
              "in the same directory. Records are found by the rules in the "
              "generator's notes (a file name written in the code, or a constant "
              "bound to one, inside a reading or writing call), so `none` means none "
              "found by those rules; a name built at run time is not seen.",
              ""]
    for i, (title, anchors) in enumerate(FLOW, 1):
        funcs = [tree.funcs[a] for a in anchors]
        llm = [a for a in anchors if a in reach]
        scope = []
        for f in funcs:
            todo, seen = [(f.key, 0)], set()
            while todo:
                k, d = todo.pop()
                if k in seen:
                    continue
                seen.add(k)
                scope.append(tree.funcs[k])
                if d < 2:
                    todo += [(c, d + 1) for c in tree.edges[k]
                             if tree.funcs[c].mod.path.parent == f.mod.path.parent]
        recs = {"reads": set(), "writes": set()}
        tools = set()
        for g in scope:
            r = records_of(tree, g)
            recs["reads"] |= r["reads"]
            recs["writes"] |= r["writes"]
            tools |= tools_named(g, all_tools)
        lines.append(f"{i}. **{title}**")
        lines.append("   - code: " + ", ".join(f"`{a}`" for a in anchors))
        if llm:
            path = " -> ".join(f"`{q}`" for q in _path_to_llm(tree, reach, llm[0]))
            lines.append(f"   - language model: yes, {path}")
        else:
            lines.append("   - language model: no; code only")
        if tools:
            lines.append(f"   - MCP tools named: {_code_list(tools)}")
        lines.append(f"   - reads: {_code_list(recs['reads'])}")
        lines.append(f"   - writes: {_code_list(recs['writes'])}")
    lines.append("")


def build():
    tree = Tree()
    missing = [a for _, anchors in FLOW for a in anchors if a not in tree.funcs]
    if missing:
        raise SystemExit("FLOW names functions that do not exist: " + ", ".join(missing))
    reach = tree.llm_reach()
    lines = ["# Architecture, generated from the source", "",
             "<!-- Generated by tools/make_architecture_doc.py. Do not edit by hand: "
             "change the code or the generator and run `python tools/make_architecture_doc.py`. "
             "tests/test_architecture_doc.py fails when this file is out of date. -->", "",
             "> Hand-written: the framework designs and runs ELM experiments. Reception "
             "and the Planner are language-model agents that design a study; code checks "
             "the design; the model server (`mcp/elm-mcp`) owns what is known about ELM, "
             "builds the runs and computes their numbers; the numbers report and the "
             "Analyzer read those records. Everything below this note is read from "
             "the code except the `FLOW` list in section 6 (the order of the stages, "
             "their titles and the functions named for each), which is hand-written "
             "and marked so there. The page drawn by "
             "`tools/make_arch_map.py` is the same picture with hand-written detail, and "
             "`ARCHITECTURE.md` at the top of the repository is the hand-written overview.", ""]
    section_llm(tree, reach, lines)
    section_client(tree, lines)
    section_mcp(tree, lines)
    section_direct(tree, lines)
    section_libs(tree, lines)
    section_flow(tree, reach, lines)
    text = "\n".join(lines).rstrip() + "\n"
    if "\u2014" in text:
        raise SystemExit("the document would contain an em-dash; see _dash()")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when the committed document differs; write nothing")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    text = build()
    out = ROOT / a.out
    if a.check:
        old = out.read_text() if out.is_file() else ""
        if old != text:
            print(f"{a.out} is out of date; run: python tools/make_architecture_doc.py")
            return 1
        print(f"{a.out} is current")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {a.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
