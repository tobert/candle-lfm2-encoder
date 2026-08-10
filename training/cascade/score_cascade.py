import json, statistics as st

probes = {p['id']: p for p in json.load(open('training/cascade/probes.json'))['probes']}

def load_raw(path):
    d = json.load(open(path))
    return {r['id']: r for r in d['rows']}, d

raw6, meta6 = load_raw('training/cascade/eval_v6_router_raw.json')
raw7, meta7 = load_raw('training/cascade/eval_v7_router_raw.json')
raw8, meta8 = load_raw('training/cascade/eval_v8_router_raw.json')

label_map6 = meta6['provenance']['label_mapping_no_relabel_hypothesis']  # v6label -> goldname
inv_map6 = {v: k for k, v in label_map6.items()}
labels6 = meta6['classifier_labels']  # destructive, informative, mutating
labels7 = meta7['classifier_labels']  # informative, situation-normal, data-critical
labels8 = meta8['classifier_labels']  # informative, situation-normal, data-critical (v8 coverage batch)

WINNER_BANDS = {'laundering', 'fanout', 'nav_noise', 'severity_tie'}  # a/b/c/e


def gold_label6_index(gold_name):
    v6label = inv_map6[gold_name]
    return labels6.index(v6label)


def gold_label7_index(gold_name):
    return labels7.index(gold_name)


def predicted_label(raw_row, i, labels, to_gold):
    probs = raw_row['clauses'][i]['severity_probs']
    idx = max(range(len(probs)), key=lambda k: probs[k])
    native = labels[idx]
    return to_gold(native), probs


def score(raw, to_gold, labels, tag):
    print(f"\n=== {tag} ===")
    # 1. winner selection over bands a/b/c/e (36 rows)
    winner_total = 0
    winner_correct = 0
    winner_misses = []
    for pid, p in probes.items():
        if p['band'] not in WINNER_BANDS:
            continue
        gwc = p['gold_winning_clause']
        if gwc is None:
            continue  # null-winner rows excluded by convention (shouldn't occur in a/b/c/e, but guard anyway)
        r = raw.get(pid)
        if r is None:
            continue
        winner_total += 1
        gold_set = gwc if isinstance(gwc, list) else [gwc]
        clauses = [c['clause'] for c in r['clauses']]
        predicted_clause = clauses[r['winner']]
        ok = predicted_clause in gold_set
        if ok:
            winner_correct += 1
        else:
            winner_misses.append((pid, predicted_clause, gold_set))
    print(f"winner selection: {winner_correct}/{winner_total} = {100*winner_correct/winner_total:.1f}%")
    if winner_misses:
        print("  misses:", winner_misses)

    # 2. per-clause severity vs gold, overall + per gold class
    total = 0
    correct = 0
    per_class_total = {}
    per_class_correct = {}
    per_clause_errors = []
    for pid, p in probes.items():
        if pid == 'f9':
            continue
        r = raw.get(pid)
        if r is None:
            continue
        gold_clauses = p['gold_severity']
        for i, gc in enumerate(gold_clauses):
            gold_name = gc['severity']
            pred_name, probs = predicted_label(r, i, labels, to_gold)
            total += 1
            per_class_total[gold_name] = per_class_total.get(gold_name, 0) + 1
            ok = pred_name == gold_name
            if ok:
                correct += 1
                per_class_correct[gold_name] = per_class_correct.get(gold_name, 0) + 1
            else:
                per_clause_errors.append((pid, i, gc['clause'], gold_name, pred_name))
    print(f"per-clause severity overall: {correct}/{total} = {100*correct/total:.1f}%")
    for cls in ['informative', 'situation-normal', 'data-critical']:
        c = per_class_correct.get(cls, 0)
        t = per_class_total.get(cls, 0)
        print(f"  gold {cls}: {c}/{t} = {100*c/t:.1f}%" if t else f"  gold {cls}: n=0")

    # 3. calibration separation: top-1 softmax confidence, escalate_measured (row-level flag broadcast) vs unanimous
    esc_clause = []
    non_clause = []
    esc_row = []
    non_row = []
    for pid, p in probes.items():
        if pid == 'f9':
            continue
        r = raw.get(pid)
        if r is None:
            continue
        e = p['escalate_measured']
        tops = []
        for c in r['clauses']:
            t = max(c['severity_probs'])
            tops.append(t)
            (esc_clause if e else non_clause).append(t)
        winner_top = max(r['clauses'][r['winner']]['severity_probs'])
        (esc_row if e else non_row).append(winner_top)
    clause_sep = st.mean(non_clause) - st.mean(esc_clause)
    row_sep = st.mean(non_row) - st.mean(esc_row)
    print(f"calibration (top-1 softmax conf): unanimous mean {st.mean(non_clause):.4f} (n={len(non_clause)}), "
          f"escalate_measured mean {st.mean(esc_clause):.4f} (n={len(esc_clause)}), clause-level separation {clause_sep:+.4f}")
    print(f"  row-level (winner clause only): unanimous {st.mean(non_row):.4f} (n={len(non_row)}), "
          f"escalate_measured {st.mean(esc_row):.4f} (n={len(esc_row)}), separation {row_sep:+.4f}")

    return dict(winner=(winner_correct, winner_total), overall=(correct, total),
                per_class=(per_class_correct, per_class_total),
                clause_sep=clause_sep, row_sep=row_sep,
                per_clause_errors=per_clause_errors, winner_misses=winner_misses)


print("Recomputing v6 under this script's exact methodology (sanity check against findings_eval.md's committed numbers)")
res6 = score(raw6, lambda n: label_map6[n], labels6, "v6 (recomputed)")
res7 = score(raw7, lambda n: n, labels7, "v7")
res8 = score(raw8, lambda n: n, labels8, "v8")

print("\n=== a6 / c5 / e4 spotlight ===")
for pid in ['a6', 'c5', 'e4']:
    p = probes[pid]
    r6 = raw6[pid]
    r7 = raw7[pid]
    r8 = raw8[pid]
    clauses = [c['clause'] for c in p['gold_severity']]
    print(f"\n{pid} ({p['band']}): {p['statement']!r}")
    print(f"  gold_winning_clause: {p['gold_winning_clause']!r}")
    print(f"  v6 winner: [{r6['winner']}] {clauses[r6['winner']]!r}  scores={[round(c['severity_score'],4) for c in r6['clauses']]}")
    print(f"  v7 winner: [{r7['winner']}] {clauses[r7['winner']]!r}  scores={[round(c['severity_score'],4) for c in r7['clauses']]}")
    print(f"  v8 winner: [{r8['winner']}] {clauses[r8['winner']]!r}  scores={[round(c['severity_score'],4) for c in r8['clauses']]}")
    for i, cl in enumerate(clauses):
        gold_name = p['gold_severity'][i]['severity']
        pn6, _ = predicted_label(r6, i, labels6, lambda n: label_map6[n])
        pn7, _ = predicted_label(r7, i, labels7, lambda n: n)
        pn8, probs8 = predicted_label(r8, i, labels8, lambda n: n)
        print(f"    clause[{i}] {cl!r}: gold={gold_name} v6_pred={pn6} v7_pred={pn7} v8_pred={pn8} "
              f"v8_probs={[round(x,4) for x in probs8]}")

# lane accuracy sanity check (should not change; router untouched)
lane_ids = ['k8s', 'shell', 'network', 'secrets', 'database', 'git', 'package', 'benign']
for tag, raw in [('v7', raw7), ('v8', raw8)]:
    lane_correct = 0
    lane_total = 0
    for pid, p in probes.items():
        if p['band'] not in WINNER_BANDS:
            continue
        if p['gold_winning_clause'] is None:
            continue
        r = raw.get(pid)
        lane_total += 1
        # lanes list order matches meta7/meta8['lanes']; need lane id names
        winner_lane_name = lane_ids[r['winner_lane']]
        if winner_lane_name == p.get('expected_lane'):
            lane_correct += 1
    print(f"\nlane accuracy (winner's lane vs authored expected_lane), {tag} router pass: "
          f"{lane_correct}/{lane_total} = {100*lane_correct/lane_total:.1f}%")
