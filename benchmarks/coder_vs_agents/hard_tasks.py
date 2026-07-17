#!/usr/bin/env python
"""Hard / complex / long-horizon benchmark tasks.

Each task is multi-file, stateful, or algorithmically deep, and is graded
by a HIDDEN test suite the agent never sees (written into the workspace
only at verify time, after the agent has finished). Same schema as
tasks.py so bench.py can run it via BENCH_TASKS=hard_tasks.
"""

from __future__ import annotations

import re as _re
import subprocess
from pathlib import Path


def _run(py: str, ws: Path, code: str, timeout: int = 90):
    p = subprocess.run([py, "-c", code], cwd=ws, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or ""), (p.stderr or "")


_EVAL_CALL = _re.compile(r"(?<![\w.])(eval|exec)\s*\(")


def _uses_builtin_eval(txt: str) -> bool:
    """True only for the *builtin* eval/exec/ast - not a user's `.eval()` /
    `_eval()` / `def eval(...)` (a recursive-descent evaluator legitimately
    names a method `eval`)."""
    if "literal_eval" in txt:
        return True
    if _re.search(r"(?m)^\s*(?:import\s+ast\b|from\s+ast\b)", txt):
        return True
    defined = set(_re.findall(r"\bdef\s+(eval|exec)\b", txt))
    for m in _EVAL_CALL.finditer(txt):
        name = m.group(1)
        prefix = txt[: m.start()].rsplit("\n", 1)[-1]
        if prefix.lstrip().startswith("def "):
            continue  # a definition, not a call
        if name in defined:
            continue  # calling their own shadowing function
        return True
    return False


def _forbidden_calc(ws: Path) -> str | None:
    for f in [*sorted(ws.glob("calc/*.py")), *sorted(ws.glob("*.py"))]:
        if not f.is_file():
            continue
        if _uses_builtin_eval(f.read_text(errors="ignore")):
            return f"{f.relative_to(ws)} uses builtin eval/exec/ast"
    return None


def _forbidden_regex(ws: Path) -> str | None:
    for f in sorted(ws.glob("*.py")):
        if not f.is_file():
            continue
        txt = f.read_text(errors="ignore")
        if _re.search(r"(?m)^\s*(?:import\s+re\b|from\s+re\b)", txt):
            return f"{f.relative_to(ws)} imports the re module"
        if _uses_builtin_eval(txt):
            return f"{f.relative_to(ws)} uses builtin eval/exec"
    return None


# ── H1: recursive-descent arithmetic evaluator (multi-module) ──────
def v_h1(ws: Path, py: str):
    bad = _forbidden_calc(ws)
    if bad:
        return False, bad
    code = r"""
import calc
exprs = ["2+3*4","(2+3)*4","2**3**2","-2**2","10/4","1+2-3+4",
         "((1+2)*(3+4))","-(3+4)*2","2*-3","3.5*2+1","100/8/2",
         "2**-1","(1-2)**3","2+2*2-2/2"]
for e in exprs:
    got = calc.evaluate(e); exp = eval(e)
    assert abs(float(got)-float(exp)) < 1e-9, (e, got, exp)
try:
    calc.evaluate("1/0"); raise SystemExit("no ZeroDivisionError")
except ZeroDivisionError:
    pass
print("OK")
"""
    rc, out, err = _run(py, ws, code)
    return rc == 0 and "OK" in out, f"rc={rc} {(err or out).strip()[-220:]}"


# ── H2: KV store with TTL + LRU + persistence + CLI ────────────────
def v_h2(ws: Path, py: str):
    code = r"""
import kvstore
t = [1000.0]
clk = lambda: t[0]
s = kvstore.KVStore(2, now=clk)
s.set('a', 1); s.set('b', 2)
assert s.get('a') == 1            # 'a' now most-recently-used
s.set('c', 3)                     # over capacity -> evict LRU ('b')
assert len(s) == 2, len(s)
try:
    s.get('b'); raise SystemExit("LRU: 'b' should have been evicted")
except KeyError:
    pass
assert s.get('c') == 3 and s.get('a') == 1
# TTL expiry against injected clock
s.set('d', 4, ttl=10)
t[0] = 1005.0
assert s.get('d') == 4
t[0] = 1011.0
try:
    s.get('d'); raise SystemExit("TTL: 'd' should have expired")
except KeyError:
    pass
# persistence round-trip
p = kvstore.KVStore(5, now=clk)
p.set('x', 'hello'); p.set('y', [1, 2, 3])
p.save('dump.json')
q = kvstore.KVStore.load('dump.json')
assert q.get('x') == 'hello' and q.get('y') == [1, 2, 3]
print("OK")
"""
    rc, out, err = _run(py, ws, code)
    class_ok = rc == 0 and "OK" in out
    # CLI persistence (bonus, must not crash the gate)
    cli_ok = False
    try:
        subprocess.run(
            [py, "kvstore.py", "set", "foo", "bar"],
            cwd=ws,
            capture_output=True,
            text=True,
            timeout=30,
        )
        g = subprocess.run(
            [py, "kvstore.py", "get", "foo"], cwd=ws, capture_output=True, text=True, timeout=30
        )
        cli_ok = "bar" in (g.stdout or "")
    except Exception:  # noqa: BLE001
        cli_ok = False
    detail = f"class={'ok' if class_ok else 'FAIL'} cli={'ok' if cli_ok else 'no'}"
    if not class_ok:
        detail += f" :: rc={rc} {(err or out).strip()[-180:]}"
    return class_ok, detail  # class behaviour is the gate; CLI is reported only


# ── H3: refactor a monolith, keep tests green, add a feature ───────
_APP = """# messy monolithic account ledger
accounts = {}

def create_account(name):
    if name in accounts:
        raise ValueError("exists")
    accounts[name] = 0
    return name

def deposit(name, amount):
    if amount <= 0:
        raise ValueError("amount must be positive")
    accounts[name] = accounts[name] + amount
    return accounts[name]

def withdraw(name, amount):
    if amount <= 0:
        raise ValueError("amount must be positive")
    if accounts[name] < amount:
        raise ValueError("insufficient funds")
    accounts[name] = accounts[name] - amount
    return accounts[name]

def balance(name):
    return accounts[name]

def transfer(src, dst, amount):
    withdraw(src, amount)
    deposit(dst, amount)
    return (accounts[src], accounts[dst])
"""

_TEST_APP = """import pytest
import app

def setup_function(fn):
    app.accounts.clear()

def test_create_and_deposit():
    app.create_account("alice")
    assert app.deposit("alice", 100) == 100

def test_withdraw():
    app.create_account("bob")
    app.deposit("bob", 50)
    assert app.withdraw("bob", 20) == 30

def test_transfer():
    app.create_account("a")
    app.create_account("b")
    app.deposit("a", 100)
    assert app.transfer("a", "b", 40) == (60, 40)

def test_insufficient():
    app.create_account("c")
    with pytest.raises(ValueError):
        app.withdraw("c", 10)
"""


def v_h3(ws: Path, py: str):
    # 1) restore the pristine provided suite (agent must not weaken it) and run it
    (ws / "test_app.py").write_text(_TEST_APP)
    r1 = subprocess.run(
        [py, "-m", "pytest", "-q", "test_app.py"],
        cwd=ws,
        capture_output=True,
        text=True,
        timeout=120,
    )
    provided_ok = r1.returncode == 0
    # 2) hidden feature test: history + undo_last
    code = r"""
import app
app.accounts.clear()
app.create_account("x")
app.deposit("x", 100)
app.withdraw("x", 30)
assert app.balance("x") == 70
h = app.history("x")
assert isinstance(h, list) and len(h) >= 2, h
app.undo_last("x")            # reverse the withdraw(30)
assert app.balance("x") == 100, app.balance("x")
print("OK")
"""
    rc, out, err = _run(py, ws, code)
    feature_ok = rc == 0 and "OK" in out
    # 3) structural: did they actually split into modules?
    modules = [
        p.name
        for p in ws.rglob("*.py")
        if p.name not in ("test_app.py",)
        and not p.name.startswith("_")
        and "__pycache__" not in str(p)
    ]
    refactored = len([m for m in modules if m != "app.py"]) >= 1
    ok = provided_ok and feature_ok
    return ok, (
        f"provided_tests={'pass' if provided_ok else 'FAIL'} "
        f"feature={'pass' if feature_ok else 'FAIL'} "
        f"refactored={'yes' if refactored else 'no'}"
        + ("" if feature_ok else f" :: {(err or out).strip()[-140:]}")
    )


# ── H4: mini regex engine (hard algorithm, no `re`) ────────────────
def v_h4(ws: Path, py: str):
    bad = _forbidden_regex(ws)
    if bad:
        return False, bad
    code = r"""
import reengine
cases = [
    ("abc","xabcy",True),("abc","abx",False),
    ("a.c","axc",True),("a.c","ac",False),
    ("ab*c","ac",True),("ab*c","abbbc",True),
    ("ab+c","ac",False),("ab+c","abc",True),
    ("colou?r","color",True),("colou?r","colour",True),("colou?r","colouur",False),
    ("^abc","abcd",True),("^abc","xabc",False),
    ("abc$","xabc",True),("abc$","abcx",False),
    ("^a.c$","abc",True),("^a.c$","abcd",False),
    ("[abc]+","xbcay",True),
    ("[a-z]+","ABC",False),("[a-z]+","aBc",True),
    ("[^0-9]+","abc",True),
    ("[0-9]$","x9",True),("[0-9]$","x9a",False),
    ("a[bc]*d","ad",True),("a[bc]*d","abcbcd",True),("a[bc]*d","abxd",False),
]
bad = []
for pat, txt, exp in cases:
    got = reengine.search(pat, txt)
    if bool(got) != exp:
        bad.append((pat, txt, exp, got))
assert not bad, f"{len(bad)} wrong: {bad[:4]}"
print("OK")
"""
    rc, out, err = _run(py, ws, code)
    return rc == 0 and "OK" in out, f"rc={rc} {(err or out).strip()[-220:]}"


TASKS = [
    {
        "id": "h1_expr_eval",
        "kind": "coding",
        "title": "Recursive-descent expression evaluator (multi-module)",
        "prompt": (
            "Create a Python package in a directory named calc/ that evaluates "
            "arithmetic expressions. Requirements:\n"
            "- calc/__init__.py must expose a function evaluate(expr: str) -> float.\n"
            "- Support binary operators + - * / and ** (exponentiation), "
            "parentheses, unary minus, and integer and decimal number literals. "
            "Whitespace is insignificant.\n"
            "- Follow standard Python operator precedence and associativity "
            "(** is right-associative; e.g. 2**3**2 == 512 and -2**2 == -4 and "
            "2**-1 == 0.5).\n"
            "- Division by zero must raise ZeroDivisionError.\n"
            "- Implement it as a real tokenizer + recursive-descent parser + "
            "evaluator split across separate modules (e.g. calc/tokenizer.py, "
            "calc/parser.py, calc/evaluator.py). Do NOT use eval(), exec(), "
            "ast.literal_eval, or the ast module.\n"
            "Then stop."
        ),
        "seed": {},
        "verify": v_h1,
    },
    {
        "id": "h2_kvstore",
        "kind": "coding",
        "title": "KV store: TTL + LRU + persistence + CLI",
        "prompt": (
            "Create a file kvstore.py implementing an in-memory key-value store.\n"
            "- class KVStore(capacity: int, *, now=time.monotonic): capacity is the "
            "max number of live entries; now is a zero-argument callable returning a "
            "float clock (injectable for testing).\n"
            "- set(key, value, ttl=None): store value; if ttl (seconds) is given, the "
            "entry expires ttl seconds after now().\n"
            "- get(key): return the value, or raise KeyError if missing or expired. "
            "A successful get counts as recent use for LRU.\n"
            "- delete(key): remove a key; raise KeyError if absent.\n"
            "- __len__: number of live (non-expired) entries.\n"
            "- When inserting a NEW key would exceed capacity, evict the "
            "least-recently-used key first. Treat expired entries as absent.\n"
            "- save(path): write the store to a JSON file. load(path): a classmethod "
            "reconstructing a store from that file.\n"
            "Also provide a CLI so that `python kvstore.py set KEY VALUE`, "
            "`python kvstore.py get KEY`, and `python kvstore.py delete KEY` operate "
            "on a JSON file store.json in the current directory, persisting between "
            "invocations (get prints the value).\n"
            "Then stop."
        ),
        "seed": {},
        "verify": v_h2,
    },
    {
        "id": "h3_refactor",
        "kind": "coding",
        "title": "Refactor monolith, keep tests green, add feature",
        "prompt": (
            "This project has a messy monolithic app.py with a passing test suite "
            "test_app.py. Do ALL of the following:\n"
            "1. Refactor the code into a cleaner structure with at least two separate "
            "modules (for example a data/model module and a service/operations "
            "module), replacing the single global accounts dict with a proper "
            "structure.\n"
            "2. IMPORTANT: keep the public functions in app.py working with the SAME "
            "names, signatures, and behavior so that the existing test_app.py keeps "
            "passing unchanged (app.py may delegate to the new modules; it must still "
            "expose an `accounts` mapping supporting .clear() and membership).\n"
            "3. Add a transaction-history feature: record every deposit, withdrawal, "
            "and transfer. Add history(name) in app.py returning a list of records "
            "(each including at least the operation type and amount) in chronological "
            "order, and undo_last(name) that reverses the most recent operation "
            "affecting that account.\n"
            "Then stop."
        ),
        "seed": {"app.py": _APP, "test_app.py": _TEST_APP},
        "verify": v_h3,
    },
    {
        "id": "h4_regex_engine",
        "kind": "coding",
        "title": "Mini regex engine (backtracking, no `re`)",
        "prompt": (
            "Create a file reengine.py exposing search(pattern: str, text: str) -> "
            "bool: return True if pattern matches anywhere in text (substring "
            "search), else False. Support, WITHOUT importing Python's re module and "
            "without eval/exec:\n"
            "- literal characters\n"
            "- . matches any single character\n"
            "- * (zero or more), + (one or more), ? (zero or one), each applying to "
            "the preceding element\n"
            "- ^ anchors to the start of the text, $ anchors to the end\n"
            "- character classes: [abc], ranges like [a-z], and negation like [^0-9]\n"
            "Implement real matching (backtracking or an NFA). Then stop."
        ),
        "seed": {},
        "verify": v_h4,
    },
]
