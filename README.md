# Who should a rep call?

```
python3 run.py              # stdlib only, under a second: propensity vs uplift
python3 research_eval.py    # stdlib only: scores 80 cached Claude account briefs
python3 research_eval.py --ask   # optional: asks Claude again (needs the Claude Code CLI)
```

Two questions every data-driven outbound team runs into, on a synthetic book for a restaurant
software company with a flat $499/month subscription. 20,000 independent restaurants last quarter,
20,000 this quarter, capacity to work 5,000. Fixed seed; no real company, restaurant or rep.

Each restaurant has two hidden numbers: the chance it signs this quarter with no rep at all (it's
already on your pricing page, a friend runs it), and the lift a rep adds. They are different
restaurants. One already shopping signs either way; one paying heavy delivery-app commissions with
no ordering of its own is the one a rep actually moves.

## 1. The dashboard picks the wrong model

Last quarter reps worked the leads an old rule ranked highest, plus a randomized holdout: 3,000
leads from the whole book, half worked and half not. Two models learn from that history and each
picks this quarter's 5,000 leads.

| policy | signed | rep-caused | LTV:CAC as measured | LTV:CAC on rep-caused deals |
|---|---|---|---|---|
| old rule: intent first, then most reviews | 1,174 | 30% | 4.19x | 1.29x |
| propensity score: most likely to sign if worked | 1,367 | 49% | **4.98x** | 2.49x |
| uplift model: most changed by a rep (holdout-trained) | 1,020 | 79% | 3.74x | **2.94x** |
| oracle: true lift (unknowable) | 1,036 | 80% | 3.85x | 3.10x |

Measured LTV:CAC credits every signed deal to sales, so it ranks the propensity score first and the
uplift model below the old rule. Counting only deals a rep caused reverses that: the uplift model
lands within 0.16 of the oracle, and adds 133 rep-caused deals ($1.34M of lifetime value) over the
propensity score for the same 5,000 calls. It sends reps to zero restaurants already on the pricing
page; the propensity list spends 1,257 calls on them.

## 2. Automated account research invents the fact the pitch depends on

Claude (Haiku, through the Claude Code CLI) wrote a JSON brief for 40 restaurants whose CRM records
were missing some fields: POS system, delivery-app use, own online ordering, monthly app orders.
68 missing fields can't be inferred from anything else in the record.

| prompt | stated a value for an unknowable field | wrong | misread a given field |
|---|---|---|---|
| open: "write a brief" | 12 of 68 (18%) | 4 | 0 of 75 |
| grounded: "use only the record, else unknown" | 0 of 68 | 0 | 0 of 75 |

10 of the 12 were "has own online ordering", the fact a pitch for a restaurant's own ordering is
built on. Three of those restaurants already had it. A brief that says otherwise sends the rep in
with the wrong opening line.

## The parts I got wrong

1. I first drew the randomized holdout only from leads the old rule skipped. The uplift model had
   never seen a pricing-page restaurant go unworked, so it still sent reps to 358 of them and only
   67% of its deals were rep-caused. Drawn from the whole book, that's 0 and 79%. The holdout has
   to include the leads you'd work anyway, which is exactly what makes it expensive.
2. My first scoring of the briefs said 34% of missing fields were invented. Half were valid
   inferences: a record with 450 monthly app orders does use delivery apps. The real number is 18%.

## Files

| | |
|---|---|
| `world.py` | restaurants, hidden baseline and lift, last quarter's history with the randomized holdout |
| `run.py` | logistic propensity model, T-learner uplift from the holdout, five policies priced on truth |
| `research_eval.py` | 40 incomplete CRM records, two prompts, field-level scoring against the truth |
| `briefs/` | the 80 briefs Claude wrote, cached |

Syed Ahmed Basharat Ali · sabasharat.ali@gmail.com · basharat.net
