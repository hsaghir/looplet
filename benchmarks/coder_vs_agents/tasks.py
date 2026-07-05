#!/usr/bin/env python
"""Benchmark task set: 5 coding + 4 non-coding tasks, each with a
deterministic verifier.

A task is a dict:
    id       - short identifier
    kind     - "coding" | "other"
    title    - human label
    prompt   - the EXACT instruction handed to both agents
    seed     - {relative_path: content} written into the workspace before the run
    verify   - callable(ws: Path, py: str) -> (passed: bool, detail: str)
               `py` is the path to a python interpreter that has pytest.

Verifiers must be self-contained and deterministic: no network, no
randomness, stable expected outputs.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


def _run_py(py: str, ws: Path, code: str, args: list[str] | None = None, timeout: int = 40):
    cmd = [py, "-c", code] if args is None else [py, *args]
    p = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def _read(ws: Path, name: str = "answer.txt") -> str | None:
    f = ws / name
    return f.read_text().strip() if f.exists() else None


# ── C1: fizzbuzz (easy, from scratch) ──────────────────────────────
def v_fizzbuzz(ws: Path, py: str):
    code = (
        "import fizzbuzz\n"
        "r=list(fizzbuzz.fizzbuzz(15))\n"
        "exp=['1','2','Fizz','4','Buzz','Fizz','7','8','Fizz','Buzz','11','Fizz','13','14','FizzBuzz']\n"
        "assert r==exp, r\n"
        "print('OK')\n"
    )
    rc, out, err = _run_py(py, ws, code)
    return rc == 0 and "OK" in out, f"rc={rc} {(err or out).strip()[-160:]}"


# ── C2: roman numerals (medium algorithm) ──────────────────────────
def v_roman(ws: Path, py: str):
    code = (
        "import roman\n"
        "cases={1:'I',4:'IV',9:'IX',40:'XL',58:'LVIII',90:'XC',400:'CD',"
        "944:'CMXLIV',1994:'MCMXCIV',3888:'MMMDCCCLXXXVIII',2023:'MMXXIII'}\n"
        "for k,v in cases.items():\n"
        "    g=roman.int_to_roman(k)\n"
        "    assert g==v,(k,g,v)\n"
        "print('OK')\n"
    )
    rc, out, err = _run_py(py, ws, code)
    return rc == 0 and "OK" in out, f"rc={rc} {(err or out).strip()[-160:]}"


# ── C3: bug fix in existing code ───────────────────────────────────
def v_stats(ws: Path, py: str):
    code = (
        "import stats\n"
        "assert stats.mean([1,2,3,4])==2.5\n"
        "assert stats.mean([2,2,2])==2.0\n"
        "assert stats.median([3,1,2])==2\n"
        "assert stats.median([1,2,3,4])==2.5\n"
        "assert stats.median([5,3,1,4,2])==3\n"
        "assert stats.mean([1,2])==1.5\n"
        "print('OK')\n"
    )
    rc, out, err = _run_py(py, ws, code)
    return rc == 0 and "OK" in out, f"rc={rc} {(err or out).strip()[-160:]}"


# ── C4: write a passing pytest suite ───────────────────────────────
def v_tests(ws: Path, py: str):
    tf = ws / "test_calc.py"
    if not tf.exists():
        return False, "no test_calc.py"
    txt = tf.read_text()
    for fn in ("add", "sub", "mul", "divide"):
        if fn not in txt:
            return False, f"test file never references {fn}()"
    if "raises" not in txt:
        return False, "no pytest.raises for divide-by-zero"
    p = subprocess.run(
        [py, "-m", "pytest", "-q", "test_calc.py"],
        cwd=ws, capture_output=True, text=True, timeout=90,
    )
    out = p.stdout + p.stderr
    m = re.search(r"(\d+) passed", out)
    npass = int(m.group(1)) if m else 0
    ok = p.returncode == 0 and npass >= 4
    return ok, f"pytest rc={p.returncode} passed={npass}"


# ── C5: build a CLI tool (generalization tested at verify time) ────
def v_wordfreq(ws: Path, py: str):
    (ws / "input.txt").write_text(
        "apple apple apple banana banana cherry cherry cherry cherry date\n"
    )
    rc, out, err = _run_py(py, ws, "", args=["wordfreq.py", "input.txt"])
    got = out.strip()
    exp = "cherry: 4\napple: 3\nbanana: 2"
    return got == exp, f"rc={rc} out={got!r} err={err.strip()[-120:]}"


# ── N1: extract + compute from JSON ────────────────────────────────
def v_json_revenue(ws: Path, py: str):
    a = _read(ws)
    if a is None:
        return False, "no answer.txt"
    m = re.search(r"-?\d+(?:\.\d+)?", a.replace(",", ""))
    if not m:
        return False, f"no number in {a!r}"
    return abs(float(m.group()) - 132.46) < 0.01, f"answer={a!r} (want 132.46)"


# ── N2: aggregate a CSV ────────────────────────────────────────────
def v_csv_region(ws: Path, py: str):
    a = _read(ws)
    if a is None:
        return False, "no answer.txt"
    lines = [ln.strip() for ln in a.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False, f"need 2 lines, got {a!r}"
    region_ok = lines[0].lower() == "west"
    m = re.search(r"\d+", lines[1])
    total_ok = bool(m) and int(m.group()) == 420
    return region_ok and total_ok, f"answer={a!r} (want west / 420)"


# ── N3: reasoning / word problem ───────────────────────────────────
def v_word_problem(ws: Path, py: str):
    a = _read(ws)
    if a is None:
        return False, "no answer.txt"
    m = re.search(r"\b(\d{1,2}:\d{2})\b", a)
    return bool(m) and m.group(1) == "10:30", f"answer={a!r} (want 10:30)"


# ── N4: dedupe + sort extraction ───────────────────────────────────
def v_emails(ws: Path, py: str):
    a = _read(ws)
    if a is None:
        return False, "no answer.txt"
    got = [ln.strip().lower() for ln in a.splitlines() if ln.strip()]
    exp = [
        "alice@example.com",
        "bob@work.org",
        "carol.smith@sub.domain.io",
        "dave@test.co",
    ]
    return got == exp, f"answer_lines={got} (want {exp})"


TASKS = [
    {
        "id": "c1_fizzbuzz",
        "kind": "coding",
        "title": "FizzBuzz from scratch",
        "prompt": (
            "Create a file named fizzbuzz.py that defines a function "
            "fizzbuzz(n) returning a list of length n. For each i from 1 "
            "to n inclusive, the element is 'Fizz' if i is divisible by 3, "
            "'Buzz' if divisible by 5, 'FizzBuzz' if divisible by both 3 "
            "and 5, otherwise the string form of i. Then stop."
        ),
        "seed": {},
        "verify": v_fizzbuzz,
    },
    {
        "id": "c2_roman",
        "kind": "coding",
        "title": "Integer to Roman numerals",
        "prompt": (
            "Create a file named roman.py that defines int_to_roman(n), "
            "converting an integer between 1 and 3999 to its uppercase "
            "Roman numeral string (e.g. 1994 -> 'MCMXCIV'). Then stop."
        ),
        "seed": {},
        "verify": v_roman,
    },
    {
        "id": "c3_bugfix",
        "kind": "coding",
        "title": "Fix bugs in stats.py",
        "prompt": (
            "The file stats.py contains two buggy functions. Fix them so "
            "that: mean([1,2,3,4]) == 2.5, mean([2,2,2]) == 2.0, "
            "median([3,1,2]) == 2, and median([1,2,3,4]) == 2.5 (the "
            "average of the two middle values after sorting). Keep the same "
            "function names. Then stop."
        ),
        "seed": {
            "stats.py": (
                "def mean(xs):\n"
                "    return sum(xs) // len(xs)\n"
                "\n"
                "def median(xs):\n"
                "    n = len(xs)\n"
                "    return xs[n // 2]\n"
            )
        },
        "verify": v_stats,
    },
    {
        "id": "c4_writetests",
        "kind": "coding",
        "title": "Write a passing pytest suite",
        "prompt": (
            "Write a pytest test file named test_calc.py that tests all "
            "four functions in calc.py (add, sub, mul, divide), including "
            "asserting that divide(x, 0) raises ZeroDivisionError. Make "
            "sure running pytest passes. Then stop."
        ),
        "seed": {
            "calc.py": (
                "def add(a, b):\n    return a + b\n\n"
                "def sub(a, b):\n    return a - b\n\n"
                "def mul(a, b):\n    return a * b\n\n"
                "def divide(a, b):\n"
                "    if b == 0:\n"
                "        raise ZeroDivisionError('division by zero')\n"
                "    return a / b\n"
            )
        },
        "verify": v_tests,
    },
    {
        "id": "c5_wordfreq_cli",
        "kind": "coding",
        "title": "Word-frequency CLI",
        "prompt": (
            "Create a file named wordfreq.py: a command-line program that "
            "takes one argument, the path to a text file. It reads the "
            "file, splits it into words (a word is a maximal run of ASCII "
            "letters; treat text case-insensitively by lowercasing), counts "
            "word frequencies, and prints the top 3 most frequent words, "
            "one per line, formatted exactly as 'word: count'. Break ties "
            "alphabetically (a before b). Then stop."
        ),
        "seed": {},
        "verify": v_wordfreq,
    },
    {
        "id": "n1_json_revenue",
        "kind": "other",
        "title": "Compute revenue from JSON",
        "prompt": (
            "Read the file data.json. It contains a list of orders under "
            "the key 'orders'; each order has an 'items' list where every "
            "item has a numeric 'qty' and 'price'. Compute the total "
            "revenue = sum of qty*price over every item in every order. "
            "Write ONLY that total, rounded to 2 decimal places, to a file "
            "named answer.txt (no other text). Then stop."
        ),
        "seed": {
            "data.json": json.dumps(
                {
                    "orders": [
                        {"id": 1, "items": [{"qty": 2, "price": 9.99}, {"qty": 1, "price": 4.50}]},
                        {"id": 2, "items": [{"qty": 3, "price": 2.00}]},
                        {"id": 3, "items": [{"qty": 1, "price": 100.00}, {"qty": 2, "price": 0.99}]},
                    ]
                },
                indent=2,
            )
        },
        "verify": v_json_revenue,
    },
    {
        "id": "n2_csv_region",
        "kind": "other",
        "title": "Aggregate a CSV",
        "prompt": (
            "Read the file sales.csv (columns: region, amount). Sum the "
            "amount per region. Write to a file named answer.txt the name "
            "of the region with the highest total on the first line, and "
            "that total as an integer on the second line. Then stop."
        ),
        "seed": {
            "sales.csv": (
                "region,amount\n"
                "north,100\n"
                "south,250\n"
                "north,300\n"
                "east,50\n"
                "south,100\n"
                "west,420\n"
            )
        },
        "verify": v_csv_region,
    },
    {
        "id": "n3_word_problem",
        "kind": "other",
        "title": "Reasoning word problem",
        "prompt": (
            "Solve this problem and write ONLY the answer to a file named "
            "answer.txt, formatted as HH:MM in 24-hour time. A train leaves "
            "a station at 09:00 traveling at 60 km/h. A second train leaves "
            "the same station at 09:30 traveling at 90 km/h in the same "
            "direction on a parallel track. At what clock time does the "
            "second train catch up to the first? Then stop."
        ),
        "seed": {},
        "verify": v_word_problem,
    },
    {
        "id": "n4_emails",
        "kind": "other",
        "title": "Extract, dedupe, sort emails",
        "prompt": (
            "Extract all unique email addresses from the file contacts.txt. "
            "Sort them alphabetically in ascending order and write them to "
            "a file named answer.txt, one per line, with no duplicates. "
            "Then stop."
        ),
        "seed": {
            "contacts.txt": (
                "From: alice@example.com\n"
                "CC: bob@work.org, carol.smith@sub.domain.io\n"
                "Please reply to alice@example.com or dave@test.co for details.\n"
            )
        },
        "verify": v_emails,
    },
]
