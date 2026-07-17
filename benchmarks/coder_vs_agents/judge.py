#!/usr/bin/env python
"""Blind LLM-as-judge for the non-coding suite.

For each task it reads both agents' answer.md, then asks two neutral
judge models (different family from the Claude contestants) to score
them on a rubric. To control position bias, every (task, judge) pair is
judged in BOTH orders (looplet-as-A and copilot-as-A). Scores are mapped
back to the real tool afterwards. The judge never learns which system
wrote which answer.

Usage: python judge.py            # judges runs_soft/, writes results_judge.json
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import soft_tasks

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_soft"
PROXY = "http://127.0.0.1:19823/v1/chat/completions"
JUDGES = ["gemini-3.1-pro-preview", "gpt-5.5"]
CRITERIA = ["accuracy", "completeness", "clarity", "practicality"]
SUFFIX = soft_tasks._SUFFIX


def _read_answer(task_id: str, tool: str) -> str:
    d = RUNS / task_id / tool
    for name in ("answer.md", "ANSWER.md", "answer.txt"):
        p = d / name
        if p.exists():
            return p.read_text(errors="ignore").strip()[:12000]
    return ""


def _call(model: str, prompt: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        PROXY,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def _parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


_RUBRIC = """You are an impartial senior engineer evaluating two AI assistant \
responses to the same task. Judge only on merit; do not reward length for its \
own sake. Be critical and discriminating - use the full 1-5 range.

TASK:
{task}

RESPONSE A:
{a}

RESPONSE B:
{b}

Score each response with integers 1-5 on:
- accuracy: technical correctness; no errors, myths, or hand-waving
- completeness: covers the important aspects with appropriate depth
- clarity: well-organized, clear, easy to follow
- practicality: actionable, realistic, sound tradeoffs and judgment

Then choose the overall winner. Reply with ONLY compact JSON:
{{"A":{{"accuracy":n,"completeness":n,"clarity":n,"practicality":n}},\
"B":{{"accuracy":n,"completeness":n,"clarity":n,"practicality":n}},\
"winner":"A|B|tie","reason":"one sentence"}}"""


def _total(scores: dict) -> int:
    return sum(int(scores.get(c, 0)) for c in CRITERIA)


def main() -> int:
    judgments = []
    for task in soft_tasks.TASKS:
        tid = task["id"]
        clean_prompt = task["prompt"].replace(SUFFIX, "").strip()
        ans = {t: _read_answer(tid, t) for t in ("looplet", "copilot")}
        for judge in JUDGES:
            # counterbalanced: order 0 -> looplet is A; order 1 -> copilot is A
            for order, (a_tool, b_tool) in enumerate(
                [("looplet", "copilot"), ("copilot", "looplet")]
            ):
                a_txt = ans[a_tool] or "(no answer produced)"
                b_txt = ans[b_tool] or "(no answer produced)"
                prompt = _RUBRIC.format(task=clean_prompt, a=a_txt, b=b_txt)
                try:
                    raw = _call(judge, prompt)
                except Exception as exc:  # noqa: BLE001
                    judgments.append(
                        {
                            "task": tid,
                            "judge": judge,
                            "order": order,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                parsed = _parse(raw)
                if not parsed:
                    judgments.append(
                        {
                            "task": tid,
                            "judge": judge,
                            "order": order,
                            "error": "unparseable",
                            "raw": raw[:200],
                        }
                    )
                    continue
                a_total = _total(parsed.get("A", {}))
                b_total = _total(parsed.get("B", {}))
                w = parsed.get("winner", "tie")
                win_tool = a_tool if w == "A" else b_tool if w == "B" else "tie"
                judgments.append(
                    {
                        "task": tid,
                        "judge": judge,
                        "order": order,
                        "scores": {a_tool: a_total, b_tool: b_total},
                        "criteria": {a_tool: parsed.get("A", {}), b_tool: parsed.get("B", {})},
                        "winner": win_tool,
                        "reason": parsed.get("reason", "")[:200],
                    }
                )
                print(
                    f"  {tid:18s} {judge:22s} order{order}  "
                    f"looplet={parsed.get('A' if a_tool == 'looplet' else 'B', {})}"
                    f" -> win={win_tool}"
                )

    (HERE / "results_judge.json").write_text(json.dumps({"judgments": judgments}, indent=2))
    _summary(judgments)
    return 0


def _summary(judgments: list[dict]) -> None:
    valid = [j for j in judgments if "scores" in j]
    tasks = []
    for j in valid:
        if j["task"] not in tasks:
            tasks.append(j["task"])

    print(
        f"\n{'=' * 74}\nBLIND JUDGE SUMMARY  ({len(valid)} valid judgments, "
        f"{len(valid) // max(len(tasks), 1)}/task)\n"
    )
    print(f"{'task':20s} {'looplet':>9s} {'copilot':>9s}  winner")
    tot = {"looplet": [], "copilot": []}
    wins = {"looplet": 0, "copilot": 0, "tie": 0}
    for t in tasks:
        js = [j for j in valid if j["task"] == t]
        lp = sum(j["scores"]["looplet"] for j in js) / len(js)
        cp = sum(j["scores"]["copilot"] for j in js) / len(js)
        tot["looplet"].append(lp)
        tot["copilot"].append(cp)
        w = "looplet" if lp - cp > 0.5 else "copilot" if cp - lp > 0.5 else "tie"
        wins[w] += 1
        print(f"{t:20s} {lp:9.1f} {cp:9.1f}  {w}")
    n = max(len(tasks), 1)
    print(f"\n{'AVG /20':20s} {sum(tot['looplet']) / n:9.1f} {sum(tot['copilot']) / n:9.1f}")
    # judgment-level win tally (each judgment = one vote)
    votes = {"looplet": 0, "copilot": 0, "tie": 0}
    for j in valid:
        votes[j["winner"]] += 1
    print(f"per-task wins   looplet={wins['looplet']} copilot={wins['copilot']} tie={wins['tie']}")
    print(
        f"judgment votes  looplet={votes['looplet']} copilot={votes['copilot']} tie={votes['tie']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
