#!/usr/bin/env python3
"""Re-score cascade winner selection under alternative severity aggregations.

HISTORY: until 2026-08-11 the cascade ranked clauses by
`severity_score = sum(P(label) for label in severe_labels)` — for v6 that is
P(mutating)+P(destructive), for v7/v8 P(situation-normal)+P(data-critical).
This script is what measured the case for replacing it; the library now ships
`expected_rank` (see `cascade::severity_rank_weights`). Both aggregations are
kept here because every recorded eval artifact was scored under the sum, and
dropping it would make those numbers unreproducible.

That sum is NOT monotone in severity: the
severe set spans two rungs of an ordinal scale, so probability mass moving from
the middle rung to the top rung (a clause getting MORE severe) leaves the sum
unchanged at best and lowers it whenever the informative rung picks up any mass.

Observed live on the deployed v8 daemon 2026-08-11:

    psql -f 001_init.sql   inf 0.0702  sn 0.6002  dc 0.3296   sum 0.9298
    psql -f drop_all.sql   inf 0.1701  sn 0.2297  dc 0.6002   sum 0.8299

The drop_all clause is correctly scored nearly twice as data-critical, and the
cascade ranks it BELOW a schema init.

`expected_rank` is the obvious monotone alternative: the expectation of the
ordinal label index, sum(rank(label) * P(label)), which is strictly increasing
as mass moves up the scale. This script measures whether that change would have
helped or hurt on the pinned probe set, using the recorded eval artifacts so no
model needs to be loaded.

Nothing here changes library behavior — this is a measurement to inform the
aggregation decision, per `commit-the-scorer`.

Usage:  python3 rescore_aggregation.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

WINNER_BANDS = {'laundering', 'fanout', 'nav_noise', 'severity_tie'}  # a/b/c/e

# Ordinal rank of each native label, per checkpoint vocabulary. Rank is the
# position on the severity scale, NOT the position in the checkpoint's label
# array — v6 stores its labels in a different order than it ranks them.
ORDINAL_RANK = {
    'informative': 0,
    'situation-normal': 1,
    'mutating': 1,          # v6's name for the middle rung
    'data-critical': 2,
    'destructive': 2,       # v6's name for the top rung
}

# Which labels the shipped `severity_score` sums, per vocabulary.
SEVERE = {'situation-normal', 'mutating', 'data-critical', 'destructive'}


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def agg_sum(probs, labels):
    """The shipped aggregation: sum of severe-label probabilities."""
    return sum(p for p, lab in zip(probs, labels) if lab in SEVERE)


def agg_expected_rank(probs, labels):
    """Expectation of the ordinal severity rank."""
    return sum(p * ORDINAL_RANK[lab] for p, lab in zip(probs, labels))


# `sum` was the shipped aggregation until 2026-08-11, when the measurement
# below moved the library to `expected_rank`. It is kept here as the
# comparison baseline — the recorded eval artifacts were all scored under it,
# so removing it would make those numbers unreproducible, which is the exact
# failure `commit-the-scorer` exists to prevent.
AGGS = [('sum (historical)', agg_sum), ('expected_rank (shipped)', agg_expected_rank)]


def winner_under(row, labels, agg):
    """Index of the highest-scoring clause; ties keep input order, as the
    library's ranking does."""
    scores = [agg(c['severity_probs'], labels) for c in row['clauses']]
    return max(range(len(scores)), key=lambda i: scores[i]), scores


def score(raw, probes, tag):
    labels = raw['classifier_labels']
    rows = {r['id']: r for r in raw['rows']}

    print(f"\n=== {tag}  (labels: {', '.join(labels)}) ===")
    results = {}
    for agg_name, agg in AGGS:
        total = correct = 0
        misses = []
        flipped = []
        for pid, p in probes.items():
            if p['band'] not in WINNER_BANDS:
                continue
            gwc = p.get('gold_winning_clause')
            if gwc is None:
                continue
            row = rows.get(pid)
            if row is None:
                continue
            total += 1
            gold_set = gwc if isinstance(gwc, list) else [gwc]
            clauses = [c['clause'] for c in row['clauses']]
            widx, _ = winner_under(row, labels, agg)
            predicted = clauses[widx]
            if predicted in gold_set:
                correct += 1
            else:
                misses.append((pid, predicted, gold_set))
            # Did this aggregation pick a different clause than the shipped
            # one? Record whether the SHIPPED pick was already correct — on
            # severity_tie rows gold accepts either co-winner, so a flip can
            # change the winner without changing correctness. Reporting only
            # "now correct" would overstate the gain.
            if widx != row['winner']:
                flipped.append((pid, clauses[row['winner']], predicted,
                                clauses[row['winner']] in gold_set,
                                predicted in gold_set))
        results[agg_name] = (correct, total, misses, flipped)
        pct = 100 * correct / total if total else float('nan')
        print(f"  {agg_name:16s} winner selection: {correct}/{total} = {pct:.1f}%")

    # Detail only for the alternative, and only where it disagrees.
    _, _, _, flipped = results['expected_rank (shipped)']
    if flipped:
        print(f"  expected_rank changes the winner on {len(flipped)} row(s):")
        for pid, was, now, was_ok, now_ok in flipped:
            if was_ok and now_ok:
                verdict = 'both accepted (tie row) — no scoring change'
            elif now_ok:
                verdict = 'FIXED'
            elif was_ok:
                verdict = 'BROKEN'
            else:
                verdict = 'still wrong'
            print(f"    {pid}: {was!r} -> {now!r}   ({verdict})")
    else:
        print("  expected_rank picks the same winner on every scored row.")
    return results


def monotonicity_report(raw, tag):
    """Count clause PAIRS where the shipped sum contradicts the ordinal
    expectation — i.e. the sum ranks A above B while A is less severe on the
    ordinal scale. These are the cases the live psql example exposed."""
    labels = raw['classifier_labels']
    inversions = 0
    pairs = 0
    examples = []
    for row in raw['rows']:
        cs = row['clauses']
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                pi, pj = cs[i]['severity_probs'], cs[j]['severity_probs']
                si, sj = agg_sum(pi, labels), agg_sum(pj, labels)
                ei, ej = agg_expected_rank(pi, labels), agg_expected_rank(pj, labels)
                pairs += 1
                if (si - sj) * (ei - ej) < 0:
                    inversions += 1
                    if len(examples) < 4:
                        examples.append((row['id'], cs[i]['clause'], cs[j]['clause'],
                                         si, sj, ei, ej))
    pct = 100 * inversions / pairs if pairs else 0.0
    print(f"  sum-vs-ordinal disagreements: {inversions}/{pairs} clause pairs "
          f"({pct:.1f}%) rank in OPPOSITE order")
    for pid, a, b, si, sj, ei, ej in examples:
        print(f"    {pid}: {a!r} sum={si:.4f} rank={ei:.4f}")
        print(f"        {' ' * len(pid)}  {b!r} sum={sj:.4f} rank={ej:.4f}")


def main():
    probes_raw = load('probes.json')
    probes = probes_raw if isinstance(probes_raw, dict) else {}
    if 'probes' in probes:
        probes = probes['probes']
    if isinstance(probes, list):
        probes = {p['id']: p for p in probes}

    for version in ('v6', 'v7', 'v8'):
        raw = load(f'eval_{version}_router_raw.json')
        score(raw, probes, version)
        monotonicity_report(raw, version)


if __name__ == '__main__':
    main()
