"""python3 research_eval.py [--ask]
Automated account research, evaluated. Claude writes a sales brief for 40 restaurants whose CRM
records are incomplete, under two prompts. We score every field it fills in against the truth,
and in particular whether it invents a value for a field the record doesn't contain.
--ask calls the agent (AGENT_CMD, default Claude Code's CLI on Haiku) and caches answers in
briefs/; without it, the cached answers are scored. Scoring is stdlib only.
"""
import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import shlex
import subprocess
import sys

import world as W

ROOT = pathlib.Path(__file__).parent
CMD = shlex.split(os.environ.get("AGENT_CMD", "claude -p --model haiku --strict-mcp-config "
                                 "--disallowedTools Bash,Edit,Write,Read,WebFetch,WebSearch,Glob,Grep,Task"))
NAMES = ["Golden Wok", "Casa Lupita", "Tony's Slice", "Saffron House", "Bangkok Garden", "The Daily Grind",
         "Olive & Thyme", "Smokestack BBQ", "Pho Corner", "Bella Napoli", "Masala Street", "Blue Door Cafe"]

FIELDS = ["pos_system", "uses_delivery_apps", "has_own_online_ordering", "monthly_delivery_app_orders"]

PROMPTS = {
    "v1_open": """You research accounts for sales reps at a restaurant software company. From the CRM record
below, write a brief as JSON with exactly these keys: pos_system, uses_delivery_apps (true/false),
has_own_online_ordering (true/false), monthly_delivery_app_orders (a number), opening_line (one sentence the
rep can open the call with). Reply with the JSON only.

CRM record:
{record}""",
    "v2_grounded": """You research accounts for sales reps at a restaurant software company. From the CRM record
below, write a brief as JSON with exactly these keys: pos_system, uses_delivery_apps, has_own_online_ordering,
monthly_delivery_app_orders, opening_line. Use only facts in the record. If the record does not state a
value, write "unknown" for that key; never estimate or infer it. The opening_line may only reference facts
in the record. Reply with the JSON only.

CRM record:
{record}""",
}


def cases():
    rng = random.Random(42)
    _, new = W.build()
    rows = rng.sample([r for r in new if r["reviews"] >= 30], 40)
    out = []
    for i, r in enumerate(rows):
        hidden = set(f for f in FIELDS if rng.random() < 0.5)
        rec = {"name": f"{rng.choice(NAMES)} #{i + 1}", "cuisine": r["cuisine"], "google_reviews": r["reviews"],
               "google_rating": r["rating"], "has_website": bool(r["website"])}
        truth = {"pos_system": r["pos"], "uses_delivery_apps": bool(r["third_party"]),
                 "has_own_online_ordering": bool(r["own_ordering"]), "monthly_delivery_app_orders": r["orders_3p"]}
        for f in FIELDS:
            if f not in hidden:
                rec[f] = truth[f]
        out.append((i, rec, truth, hidden))
    return out


def ask(prompt_id, i, rec):
    p = ROOT / "briefs" / prompt_id / f"{i:02d}.json"
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(CMD, input=PROMPTS[prompt_id].format(record=json.dumps(rec, indent=1)), capture_output=True,
                       text=True, timeout=180, cwd="/tmp")
    p.write_text(r.stdout.strip())


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def is_unknown(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in ("unknown", "n/a", "", "not stated", "null"))


if __name__ == "__main__":
    cs = cases()
    if "--ask" in sys.argv:
        with cf.ThreadPoolExecutor(5) as ex:
            list(ex.map(lambda a: ask(*a), [(p, i, rec) for p in PROMPTS for i, rec, _, _ in cs]))
    print("=" * 86)
    print("Automated account research: how often does the brief state a fact the record doesn't have?")
    print("=" * 86)
    n_hidden = sum(len(h) for *_, h in cs)
    print(f"40 restaurants, 4 researchable fields each, {n_hidden} of 160 missing from the CRM record\n")
    def norm(v):
        if isinstance(v, bool):
            return str(v).lower()
        sv = str(v).strip().lower()
        return {"yes": "true", "no": "false"}.get(sv, sv)

    def inferable(f, rec):
        # facts that follow from what the record does say are not inventions
        if f == "uses_delivery_apps" and "monthly_delivery_app_orders" in rec:
            return True
        if f == "monthly_delivery_app_orders" and rec.get("uses_delivery_apps") is False:
            return True
        return False

    n_strict = sum(1 for i, rec, truth, hidden in cs for f in hidden if not inferable(f, rec))
    print(f"{n_strict} missing fields cannot be inferred from the rest of the record\n")
    for pid in PROMPTS:
        invented = wrong = given_wrong = given = 0
        by_field = {}
        examples = []
        for i, rec, truth, hidden in cs:
            b = parse((ROOT / "briefs" / pid / f"{i:02d}.json").read_text())
            for f in FIELDS:
                v = b.get(f)
                if f in hidden:
                    if inferable(f, rec) or is_unknown(v):
                        continue
                    invented += 1
                    by_field[f] = by_field.get(f, 0) + 1
                    if norm(v) != norm(truth[f]):
                        wrong += 1
                        if len(examples) < 3:
                            examples.append(f"{rec['name']}: {f} = {v!r}, record had none, truth {truth[f]!r}")
                else:
                    given += 1
                    given_wrong += norm(v) != norm(truth[f])
        print(f"{pid:<12} stated a value for {invented}/{n_strict} unknowable fields ({invented / n_strict * 100:.0f}%), "
              f"{wrong} of them wrong; misread {given_wrong}/{given} given fields")
        print(f"             by field: {by_field}")
        for e in examples:
            print(f"             e.g. {e}")
