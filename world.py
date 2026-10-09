"""A synthetic outbound book for a restaurant software company with a flat monthly subscription.

Every independent restaurant has a chance of signing this quarter with no rep at all (it found you,
a friend runs it, it was already shopping) and a separate lift if a rep works it. The two are not
the same restaurants. A restaurant already on your pricing page is likely to sign either way; one
paying heavy delivery-app commissions with no ordering of its own is the one a rep actually moves.

Last quarter's history is what a real team has: reps worked the leads an old score ranked highest,
plus a small randomized holdout where a random half of a random sample was worked and half wasn't.
Fixed seed. No real company, restaurant or rep.
"""
import math
import random

SEED = 499  # the monthly price, in dollars
CUISINES = ["pizza", "mexican", "chinese", "american", "indian", "thai", "mediterranean", "cafe"]
PRICE, REP_COST = 499, 600  # $/month; fully loaded rep cost per worked lead


def sig(x):
    return 1 / (1 + math.exp(-x))


def restaurant(rng, rid):
    reviews = int(rng.lognormvariate(5.0, 1.0))
    third_party = rng.random() < 0.72
    own_ordering = rng.random() < (0.25 if third_party else 0.45)
    orders_3p = int(rng.lognormvariate(6.2, 0.8)) if third_party else 0  # monthly app orders
    intent = rng.random() < 0.08          # visited pricing page / booked a demo
    website = rng.random() < 0.7
    pos = rng.choice(["Toast", "Square", "Clover", "Other"])
    r = dict(rid=rid, cuisine=rng.choice(CUISINES), reviews=reviews, rating=round(rng.uniform(3.4, 4.9), 1),
             third_party=int(third_party), own_ordering=int(own_ordering), orders_3p=orders_3p,
             intent=int(intent), website=int(website), pos=pos, metro_tier=rng.choice([1, 1, 2, 2, 3]))
    # ground truth, never observed: baseline signing and the rep's lift
    p0 = sig(-4.3 + 2.9 * intent + 0.35 * math.log1p(reviews) / 2 + 0.3 * website)
    commission_pain = math.log1p(orders_3p) / 7 * (1 - own_ordering)
    tau = 0.16 * commission_pain * (1 - 0.8 * intent) * (0.4 if reviews < 40 else 1.0) + 0.012
    r["_p0"], r["_tau"] = p0, min(tau, 0.95 - p0)
    # retention: busier restaurants stay longer (months, expected)
    r["_months"] = 10 + 14 * min(1.0, math.log1p(orders_3p + reviews) / 8)
    return r


def build():
    rng = random.Random(SEED)
    hist = [restaurant(rng, i) for i in range(20000)]
    new = [restaurant(rng, 100000 + i) for i in range(20000)]
    # last quarter: the old score was a hand-built rule (intent first, then reviews)
    # a randomized holdout drawn from the WHOLE book, including leads the rule would have worked
    rand = rng.sample(hist, 3000)
    rand_ids0 = {r["rid"] for r in rand}
    rand_treated = {r["rid"] for r in rand[:1500]}
    order = sorted([r for r in hist if r["rid"] not in rand_ids0], key=lambda r: (-r["intent"], -r["reviews"]))
    worked = {r["rid"] for r in order[:5000]}
    for r in hist:
        r["worked"] = int(r["rid"] in worked or r["rid"] in rand_treated)
    rand_ids = {r["rid"] for r in rand}
    for r in hist:
        r["in_holdout_test"] = int(r["rid"] in rand_ids)
        p = r["_p0"] + (r["_tau"] if r["worked"] else 0)
        r["signed"] = int(rng.random() < p)
    return hist, new
