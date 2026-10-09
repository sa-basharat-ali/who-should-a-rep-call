"""python3 run.py  -- stdlib only, a few seconds. Trains a propensity score and an uplift model on
last quarter, then lets each pick this quarter's 5,000 leads and prices what each pick is worth."""
import math
import random
import time

import world as W

T0 = time.time()
hist, new = W.build()
CAP = 5000


def feats(r):
    return [1.0, r["intent"], math.log1p(r["reviews"]) / 6, r["third_party"], r["own_ordering"],
            math.log1p(r["orders_3p"]) / 7, (1 - r["own_ordering"]) * math.log1p(r["orders_3p"]) / 7,
            r["website"], (r["rating"] - 4) , float(r["reviews"] < 40), r["intent"] * math.log1p(r["orders_3p"]) / 7]


def fit(rows, epochs=12, lr=0.08, l2=1e-4, seed=1):
    rng = random.Random(seed)
    X = [feats(r) for r in rows]
    y = [r["signed"] for r in rows]
    w = [0.0] * len(X[0])
    idx = list(range(len(X)))
    for ep in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            z = sum(a * b for a, b in zip(w, X[i]))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            g = p - y[i]
            w = [wj - lr * (g * xj + l2 * wj) for wj, xj in zip(w, X[i])]
    return lambda r: 1 / (1 + math.exp(-max(-30, min(30, sum(a * b for a, b in zip(w, feats(r)))))))


def money(x):
    return f"${x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"${x / 1e3:,.0f}K"


ltv = lambda r: W.PRICE * r["_months"]
cost = CAP * W.REP_COST

# models ------------------------------------------------------------------------------------------
worked = [r for r in hist if r["worked"]]
prop = fit(worked)                                            # P(sign | worked), observational
rt = [r for r in hist if r["in_holdout_test"]]
m1 = fit([r for r in rt if r["worked"]], epochs=40, seed=2)    # randomized: worked
m0 = fit([r for r in rt if not r["worked"]], epochs=40, seed=3)  # randomized: not worked
uplift = lambda r: m1(r) - m0(r)

policies = {
    "old rule: intent first, then most reviews": sorted(new, key=lambda r: (-r["intent"], -r["reviews"]))[:CAP],
    "propensity score: most likely to sign if worked": sorted(new, key=lambda r: -prop(r))[:CAP],
    "uplift model: most changed by a rep (from the holdout)": sorted(new, key=lambda r: -uplift(r))[:CAP],
    "oracle: true lift (unknowable)": sorted(new, key=lambda r: -r["_tau"])[:CAP],
}
random.seed(5)
policies["random 5,000"] = random.sample(new, CAP)

print("=" * 92)
print("Who should a rep call? Propensity vs uplift on a synthetic restaurant outbound book")
print("=" * 92)
print(f"last quarter: {len(hist):,} leads, {sum(r['worked'] for r in hist):,} worked "
      f"({len(rt):,}-lead randomized holdout inside it); this quarter: {len(new):,} leads, capacity {CAP:,} worked")
print(f"economics: ${W.PRICE}/month, rep cost ${W.REP_COST} per worked lead, LTV = price x expected months retained\n")
print(f"{'policy':<56}{'signed':>8}{'of which':>10}{'measured':>11}{'incremental':>13}")
print(f"{'':<56}{'(worked)':>8}{'rep-caused':>10}{'LTV:CAC':>11}{'LTV:CAC':>13}")
res = {}
for name, chosen in policies.items():
    signed = sum(r["_p0"] + r["_tau"] for r in chosen)
    inc = sum(r["_tau"] for r in chosen)
    meas = sum((r["_p0"] + r["_tau"]) * ltv(r) for r in chosen) / cost
    incr = sum(r["_tau"] * ltv(r) for r in chosen) / cost
    res[name] = (signed, inc, meas, incr)
    print(f"{name:<56}{signed:>8.0f}{inc / signed * 100:>9.0f}%{meas:>10.2f}x{incr:>12.2f}x")

p, u, o = res["propensity score: most likely to sign if worked"], res["uplift model: most changed by a rep (from the holdout)"], res["old rule: intent first, then most reviews"]
print(f"\n  the propensity score reports {p[2] / o[2]:.1f}x the old rule's LTV:CAC; its incremental LTV:CAC is {p[3] / o[3]:.1f}x")
print(f"  uplift vs propensity: {u[0] - p[0]:+.0f} signed deals, {u[1] - p[1]:+.0f} rep-caused deals, "
      f"incremental LTV {money((u[3] - p[3]) * cost)} for the same {CAP:,} calls")
overlap = len({r['rid'] for r in policies['propensity score: most likely to sign if worked']} &
              {r['rid'] for r in policies['uplift model: most changed by a rep (from the holdout)']})
print(f"  the two lists share {overlap:,} of {CAP:,} restaurants")
intent_p = sum(r["intent"] for r in policies["propensity score: most likely to sign if worked"])
intent_u = sum(r["intent"] for r in policies["uplift model: most changed by a rep (from the holdout)"])
print(f"  restaurants already on the pricing page: {intent_p:,} of the propensity list, {intent_u:,} of the uplift list")
print(f"\nran in {time.time() - T0:.1f}s")
