"""
Baseline Experiment: Beam Search vs SA vs GA vs PSO on Tomita Languages
========================================================================
Runs all four DFA search methods from the **same** RPNI initial automaton
on Tomita languages L3AB, L4, L6, L7 and reports:

  Metric              Description
  ──────────────────  ──────────────────────────────────────────────────
  train_acc           Accuracy on the perturbation-sample training set
  holdout_acc         Accuracy on the held-out test set (model agreement)
  states              Number of DFA states (proxy for DFA complexity)
  time (s)            Wall-clock seconds
  success             Whether the method met accuracy_threshold

Layer structure
  - Outer loop : one language at a time
  - Per language: one build_shared_init() call → four methods

Usage (from project root):
    cd /home/yihua/anchor-llm
    python examples/RPNI/run_baseline_experiment.py
"""

from __future__ import annotations

import csv
import os
import pickle
import random
import sys
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import accuracy_score

# ── Path setup (same as run_tomita.py) ─────────────────────────────────
PROJECT_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SRC_PATH         = os.path.join(PROJECT_ROOT, 'src')
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, 'external_modules')
MODIFIED_MODULES = os.path.join(PROJECT_ROOT, 'modified_modules')
EXPLAINING_FA    = os.path.join(EXTERNAL_MODULES, 'Explaining-FA')
INTERPRETERA_SRC = os.path.join(EXTERNAL_MODULES, 'interpretera', 'src')

for _p in [MODIFIED_MODULES, SRC_PATH, EXTERNAL_MODULES, EXPLAINING_FA, INTERPRETERA_SRC, PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from modified_modules.alibi.explainers.anchors.anchor_tabular import AnchorTabular
from models.sequence_classifier import SequenceClassifier
from search_baselines import build_shared_init, sa_dfa_search, ga_dfa_search, pso_dfa_search
from tee import Tee


# ======================================================================
# Per-language hyperparameter configurations  (mirrors run_tomita.py)
# ======================================================================
LANGUAGES = {
    "L3AB": dict(
        test_instance   = ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'],
        alphabet        = ['a', 'b', 'c', 'd'],
        accuracy_threshold = 0.9,
        state_threshold = 5,
        delta           = 0.01,
        tau             = 0.01,
        batch_size      = 2000,
        beam_size       = 1,
        init_num_samples= 2000,
        edit_distance   = 4,
        select_by       = 'accuracy',
        max_length      = 10, embedding_dim = 16, hidden_dim = 32,
        num_layers      = 1,  dropout = 0,
    ),
    "L4": dict(
        test_instance   = ['a', 'a', 'b', 'b', 'c', 'd'] ,
        alphabet        = ['a', 'b', 'c', 'd'],
        accuracy_threshold = 0.9,
        state_threshold = 5,
        delta           = 0.01,
        tau             = 0.01,
        batch_size      = 500,
        beam_size       = 1,
        init_num_samples= 2000,
        edit_distance   = 5,
        select_by       = 'accuracy',
        max_length      = 10, embedding_dim = 16, hidden_dim = 64,
        num_layers      = 2,  dropout = 0.5,
    ),
    "L6": dict(
        test_instance   = ['a', 'a', 'a', 'c', 'a', 'a', 'b', 'b'],
        alphabet        = ['a', 'b', 'c', 'd'],
        accuracy_threshold = 0.8,
        state_threshold = 5,
        delta           = 0.1,
        tau             = 0.1,
        batch_size      = 500,
        beam_size       = 1,
        init_num_samples= 2000,
        edit_distance   = 3,
        select_by       = 'accuracy',
        max_length      = 10, embedding_dim = 16, hidden_dim = 32,
        num_layers      = 1,  dropout = 0.5,
    ),
    "L7": dict(
        test_instance   = ['a', 'a', 'b', 'a', 'a', 'b'],
        alphabet        = ['a', 'b', 'c', 'd'],
        accuracy_threshold = 0.8,
        state_threshold = 5,
        delta           = 0.01,
        tau             = 0.01,
        batch_size      = 500,
        beam_size       = 1,
        init_num_samples= 2000,
        edit_distance   = 5,
        select_by       = 'accuracy',
        max_length      = 10, embedding_dim = 16, hidden_dim = 32,
        num_layers      = 1,  dropout = 0,
    ),
    "date": dict(
        test_instance   = None,  # Will be selected from training set
        alphabet        = None,  # Will be auto-detected from data
        accuracy_threshold = 0.9,
        state_threshold = 8,
        delta           = 0.1,
        tau             = 0.1,
        batch_size      = 2000,
        beam_size       = 1,
        init_num_samples= 2000,
        edit_distance   = 10,
        select_by       = 'accuracy',
        max_length      = 15, embedding_dim = 16, hidden_dim = 32,
        num_layers      = 1,  dropout = 0,
    ),
    "mnist": dict(
        test_instance   = None,  # Will be selected from training set
        alphabet        = None,  # Will be set to stroke symbols
        accuracy_threshold = 0.8,
        state_threshold = 5,
        delta           = 0.05,
        tau             = 0.05,
        batch_size      = 2000,
        beam_size       = 1,
        init_num_samples= 1000,
        edit_distance   = 10,
        select_by       = 'accuracy',
        max_length      = 20, embedding_dim = 64, hidden_dim = 256,
        num_layers      = 2,  dropout = 0.3,
    ),
}

# Baseline search budgets (kept comparable to a typical beam-search run)
SA_STEPS       = 150
GA_GENERATIONS = 15
GA_POPULATION  = 5
PSO_PARTICLES  = 8
PSO_ITERATIONS = 15
PSO_STEPS      = 2   # refinement steps per particle


# ======================================================================
# Holdout accuracy helper
# ======================================================================
def eval_holdout(dfa, X_test, y_binary_test, learner) -> float:
    """
    DFA accuracy on the held-out test set.

    y_binary_test[i] = 1 if model predicts the same class as test_instance
                       0 otherwise
    """
    if dfa is None or len(X_test) == 0:
        return 0.0
    accepts = np.array([learner.check_path_accepted(dfa, seq) for seq in X_test])
    lbl     = np.asarray(y_binary_test)
    correct = int(np.sum((lbl == 1) & accepts) + np.sum((lbl == 0) & ~accepts))
    return correct / len(X_test)


# ======================================================================
# Single-language experiment
# ======================================================================
def run_one_language(lang_code: str, cfg: dict, output_root: str) -> dict | None:
    print(f"\n{'='*70}")
    print(f"  LANGUAGE: {lang_code}")
    print(f"{'='*70}")

    out_dir = os.path.join(output_root, lang_code)
    os.makedirs(out_dir, exist_ok=True)

    # ── Load model + data ──────────────────────────────────────────────
    model_path = os.path.join(PROJECT_ROOT, "models", f"{lang_code}_classifier_trained.pth")
    split_path = os.path.join(PROJECT_ROOT, "models", f"{lang_code}_train_test_split.pkl")
    if not (os.path.exists(model_path) and os.path.exists(split_path)):
        print(f"  [SKIP] Pre-trained model or data split not found for {lang_code}.")
        return None

    with open(split_path, "rb") as f:
        split = pickle.load(f)
    X_train, y_train = split["X_train"], split["y_train"]
    X_test,  y_test  = split["X_test"],  split["y_test"]

    clf = SequenceClassifier(
        max_len=cfg['max_length'], embedding_dim=cfg['embedding_dim'], device='cpu')
    clf.load(model_path)
    predict_fn = lambda seqs: clf.predict(seqs)

    clf_train_acc = accuracy_score(y_train, predict_fn(X_train))
    clf_test_acc  = accuracy_score(y_test,  predict_fn(X_test))
    print(f"  Classifier  train={clf_train_acc:.4f}  test={clf_test_acc:.4f}")

    # ── Determine test_instance and alphabet based on dataset ──────────
    test_instance = cfg['test_instance']
    alphabet = cfg['alphabet']
    
    # Auto-select test_instance and alphabet if not specified
    if test_instance is None:
        # For date/mnist: select a positive instance from training set randomly
        positive_indices = [i for i, label in enumerate(y_train) if label == 1]
        if positive_indices:
            test_idx = np.random.RandomState(42).choice(positive_indices)
            test_instance = X_train[test_idx]
        else:
            test_instance = X_train[0]
    
    if alphabet is None:
        # Auto-detect alphabet from data
        alphabet = sorted(set(c for seq in X_train for c in (seq if isinstance(seq, (list, tuple)) else str(seq))))
        # Special handling for MNIST (stroke tuples)
        if lang_code == "mnist":
            alphabet = [(0, 1), (-1, 1), (1, 1), (1, 0), (1, -1), (-1, -1), (-1, 0), (0, -1), (0, 0)]
    
    # ── Binary labels for holdout evaluation ──────────────────────────
    target_label  = predict_fn([test_instance])[0]
    y_pred_test   = predict_fn(X_test)
    y_binary_test = (y_pred_test == target_label).astype(int)
    print(f"  Test instance: {test_instance}  →  label={target_label}")
    print(f"  Holdout set: {len(X_test)} samples  "
          f"({int(y_binary_test.sum())} same-class / "
          f"{int((1-y_binary_test).sum())} other-class)")

    # ── Explainer + sampler ────────────────────────────────────────────
    explainer = AnchorTabular(
        predictor=predict_fn,
        feature_names=[f'pos_{i}' for i in range(max(len(s) for s in X_train))],
        categorical_names={},
        seed=42,
    )
    explainer.fit(automaton_type='DFA', train_data=X_train,
                  alphabet=alphabet, disc_perc=None)
    explainer.samplers[0].d_train_data = X_train

    # Set instance on sampler (mirrors what explain() does before calling anchor_beam)
    for sampler in explainer.samplers:
        sampler.set_instance_label(test_instance)
        sampler.set_n_covered(10)
        sampler.edit_distance = cfg['edit_distance']

    sampler_fn = explainer.samplers[0]

    # ── Build shared initial DFA (RPNI, run ONCE) ──────────────────────
    print("\n  [SharedInit] Building initial DFA via RPNI ...")
    t_shared = time.time()
    shared = build_shared_init(
        sampler_fn        = sampler_fn,
        data_type         = "Tabular",
        init_num_samples  = cfg['init_num_samples'],
        batch_size        = cfg['batch_size'],
        output_dir        = os.path.join(out_dir, "shared"),
    )
    t_shared = time.time() - t_shared
    init_states = len(shared.initial_dfa.states)
    init_acc    = float(np.mean(
        [shared.learner.check_path_accepted(shared.initial_dfa, s)
         for s in shared.data]
    ))
    print(f"  [SharedInit] states={init_states}  "
          f"rpni_build_time={t_shared:.1f}s")

    # Shorthand for holdout using the shared learner
    def h(dfa) -> float:
        return eval_holdout(dfa, X_test, y_binary_test, shared.learner)

    results: dict = {
        "lang"              : lang_code,
        "clf_train_acc"     : clf_train_acc,
        "clf_test_acc"      : clf_test_acc,
        "initial_dfa_states": init_states,
        "shared_build_time" : t_shared,
    }

    # ══════════════════════════════════════════════════════════════════
    # Method 1 – Beam Search  (anchor_beam via explain())
    # ══════════════════════════════════════════════════════════════════
    print("\n  ─── Beam Search ─────────────────────────────────────────")
    t0 = time.time()
    beam_expl = explainer.explain(
        type               = 'Tabular',
        automaton_type     = 'DFA',
        alphabet           = cfg['alphabet'],
        X                  = test_instance,
        edit_distance      = cfg['edit_distance'],
        accuracy_threshold = cfg['accuracy_threshold'],
        state_threshold    = cfg['state_threshold'],
        select_by          = cfg['select_by'],
        delta              = cfg['delta'],
        tau                = cfg['tau'],
        beam_size          = cfg['beam_size'],
        batch_size         = cfg['batch_size'],
        init_num_samples   = cfg['init_num_samples'],
        verbose            = False,
        output_dir         = os.path.join(out_dir, "beam"),
        prebuilt_init      = shared,          # ← shared initial DFA
    )
    beam_time = time.time() - t0
    bd = beam_expl.data
    beam_dfa    = bd.get('automata')
    beam_train  = float(np.mean(bd.get('training_accuracy', [0])))
    beam_states = int(bd.get('size', 0) or
                      (len(beam_dfa.states) if beam_dfa else 0))
    beam_holdout = h(beam_dfa)
    results['beam'] = dict(
        train_acc   = beam_train,
        holdout_acc = beam_holdout,
        states      = beam_states,
        time        = beam_time,
        success     = bd.get('success', False),
    )
    print(f"    train={beam_train:.4f}  holdout={beam_holdout:.4f}  "
          f"states={beam_states}  time={beam_time:.1f}s"
          f"  {'✓' if bd.get('success') else '✗'}")

    # ══════════════════════════════════════════════════════════════════
    # Method 2 – Simulated Annealing
    # ══════════════════════════════════════════════════════════════════
    print("\n  ─── Simulated Annealing ─────────────────────────────────")
    t0 = time.time()
    sa_res  = sa_dfa_search(
        sampler_fn         = sampler_fn,
        data_type          = "Tabular",
        accuracy_threshold = cfg['accuracy_threshold'],
        state_threshold    = cfg['state_threshold'],
        select_by          = cfg['select_by'],
        init_num_samples   = cfg['init_num_samples'],
        batch_size         = cfg['batch_size'],
        output_dir         = os.path.join(out_dir, "sa"),
        n_rounds           = 1000,
        n_steps            = SA_STEPS,
        T_max              = 10.0,
        T_min              = 0.001,
        beam_size          = 10,
        shared_init        = shared,
    )
    sa_time   = time.time() - t0
    sa_dfa    = sa_res.get('automata')
    sa_train  = float(sa_res.get('training_accuracy', 0))
    sa_states = int(sa_res.get('size', 0) or
                    (len(sa_dfa.states) if sa_dfa else 0))
    sa_holdout = h(sa_dfa)
    results['sa'] = dict(
        train_acc   = sa_train,
        holdout_acc = sa_holdout,
        states      = sa_states,
        time        = sa_time,
        success     = sa_res.get('success', False),
    )
    print(f"    train={sa_train:.4f}  holdout={sa_holdout:.4f}  "
          f"states={sa_states}  time={sa_time:.1f}s"
          f"  {'✓' if sa_res.get('success') else '✗'}")

    # ══════════════════════════════════════════════════════════════════
    # Method 3 – Genetic Algorithm
    # ══════════════════════════════════════════════════════════════════
    print("\n  ─── Genetic Algorithm ───────────────────────────────────")
    t0 = time.time()
    ga_res  = ga_dfa_search(
        sampler_fn         = sampler_fn,
        data_type          = "Tabular",
        accuracy_threshold = cfg['accuracy_threshold'],
        state_threshold    = cfg['state_threshold'],
        select_by          = cfg['select_by'],
        init_num_samples   = cfg['init_num_samples'],
        batch_size         = cfg['batch_size'],
        output_dir         = os.path.join(out_dir, "ga"),
        n_rounds           = 1000,
        n_generations      = GA_GENERATIONS,
        population_size    = GA_POPULATION,
        mutation_prob      = 0.8,
        beam_size          = 10,
        shared_init        = shared,
    )
    ga_time   = time.time() - t0
    ga_dfa    = ga_res.get('automata')
    ga_train  = float(ga_res.get('training_accuracy', 0))
    ga_states = int(ga_res.get('size', 0) or
                    (len(ga_dfa.states) if ga_dfa else 0))
    ga_holdout = h(ga_dfa)
    results['ga'] = dict(
        train_acc   = ga_train,
        holdout_acc = ga_holdout,
        states      = ga_states,
        time        = ga_time,
        success     = ga_res.get('success', False),
    )
    print(f"    train={ga_train:.4f}  holdout={ga_holdout:.4f}  "
          f"states={ga_states}  time={ga_time:.1f}s"
          f"  {'✓' if ga_res.get('success') else '✗'}")

    # ══════════════════════════════════════════════════════════════════
    # Method 4 – Particle Swarm Optimisation
    # ══════════════════════════════════════════════════════════════════
    print("\n  ─── Particle Swarm Optimisation ─────────────────────────")
    t0 = time.time()
    pso_res  = pso_dfa_search(
        sampler_fn         = sampler_fn,
        data_type          = "Tabular",
        accuracy_threshold = cfg['accuracy_threshold'],
        state_threshold    = cfg['state_threshold'],
        select_by          = cfg['select_by'],
        init_num_samples   = cfg['init_num_samples'],
        batch_size         = cfg['batch_size'],
        output_dir         = os.path.join(out_dir, "pso"),
        n_rounds           = 1000,
        n_particles        = PSO_PARTICLES,
        n_steps_per_particle = PSO_STEPS,
        beam_size          = 10,
        shared_init        = shared,
    )
    pso_time   = time.time() - t0
    pso_dfa    = pso_res.get('automata')
    pso_train  = float(pso_res.get('training_accuracy', 0))
    pso_states = int(pso_res.get('size', 0) or
                     (len(pso_dfa.states) if pso_dfa else 0))
    pso_holdout = h(pso_dfa)
    results['pso'] = dict(
        train_acc   = pso_train,
        holdout_acc = pso_holdout,
        states      = pso_states,
        time        = pso_time,
        success     = pso_res.get('success', False),
    )
    print(f"    train={pso_train:.4f}  holdout={pso_holdout:.4f}  "
          f"states={pso_states}  time={pso_time:.1f}s"
          f"  {'✓' if pso_res.get('success') else '✗'}")

    return results


# ======================================================================
# Summary table + CSV export
# ======================================================================
METHODS = ['beam', 'sa', 'ga', 'pso']
METHOD_LABELS = {'beam': 'BeamSearch', 'sa': 'SA', 'ga': 'GA', 'pso': 'PSO'}

def print_summary(all_results: dict) -> None:
    col = 12
    sep = "+" + "-"*8 + ("+" + "-"*col) * 5 + "+"
    head = (f"| {'Lang':6s} | {'Method':10s} | {'TrainAcc':10s} | "
            f"{'Holdout':10s} | {'States':10s} | {'Time(s)':10s} |")

    print("\n\n" + "=" * 70)
    print("  EXPERIMENT SUMMARY")
    print("=" * 70)

    # Per-language header with classifier baseline
    for lang_code, res in sorted(all_results.items()):
        if res is None:
            print(f"\n  {lang_code}: [NO DATA]")
            continue

        print(f"\n  {lang_code}  "
              f"(clf_train={res['clf_train_acc']:.4f}  "
              f"clf_test={res['clf_test_acc']:.4f}  "
              f"initial_states={res['initial_dfa_states']})")
        print(sep)
        print(head)
        print(sep)
        for m in METHODS:
            r = res.get(m, {})
            ok = '✓' if r.get('success') else '✗'
            print(f"| {lang_code:6s} | {METHOD_LABELS[m]:10s} | "
                  f"{r.get('train_acc', 0):.4f} {ok:1s}     | "
                  f"{r.get('holdout_acc', 0):.4f}      | "
                  f"{r.get('states', 0):10d} | "
                  f"{r.get('time', 0):10.1f} |")
        print(sep)

    # Paper suitability analysis
    print("\n\n  PAPER SUITABILITY ANALYSIS")
    print("  " + "─" * 66)
    print(f"  {'Lang':6s}  {'BeamSearch':>10s}  {'SA':>8s}  "
          f"{'GA':>8s}  {'PSO':>8s}  {'Δ_holdout':>10s}  {'Comment':s}")
    print("  " + "─" * 66)

    for lang_code, res in sorted(all_results.items()):
        if res is None:
            print(f"  {lang_code:6s}  [NO DATA]")
            continue
        beam_h = res.get('beam', {}).get('holdout_acc', 0)
        sa_h   = res.get('sa',   {}).get('holdout_acc', 0)
        ga_h   = res.get('ga',   {}).get('holdout_acc', 0)
        pso_h  = res.get('pso',  {}).get('holdout_acc', 0)
        best_baseline = max(sa_h, ga_h, pso_h)
        delta = beam_h - best_baseline

        # Simple heuristic to flag "interesting" languages
        bean_states = res.get('beam', {}).get('states', 0)
        comment = []
        if delta > 0.05:
            comment.append("beam clearly wins")
        elif delta < -0.05:
            comment.append("baseline beats beam!")
        else:
            comment.append("methods similar")
        if bean_states >= 3:
            comment.append("non-trivial DFA")
        if res['initial_dfa_states'] >= 5:
            comment.append("rich init")

        print(f"  {lang_code:6s}  {beam_h:10.4f}  {sa_h:8.4f}  "
              f"{ga_h:8.4f}  {pso_h:8.4f}  {delta:+10.4f}  "
              f"{', '.join(comment)}")

    print()


def save_csv(all_results: dict, path: str) -> None:
    rows = []
    for lang, res in all_results.items():
        if res is None:
            continue
        for m in METHODS:
            r = res.get(m, {})
            rows.append({
                'language'     : lang,
                'method'       : m,
                'train_acc'    : r.get('train_acc',    ''),
                'holdout_acc'  : r.get('holdout_acc',  ''),
                'states'       : r.get('states',       ''),
                'time_s'       : r.get('time',         ''),
                'success'      : int(r.get('success', False)),
                'clf_train_acc': res.get('clf_train_acc',      ''),
                'clf_test_acc' : res.get('clf_test_acc',       ''),
                'init_states'  : res.get('initial_dfa_states', ''),
            })
    if not rows:
        print("  [CSV] No rows to write.")
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV saved → {path}")


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "test_result", "baseline_experiment")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    log_path = os.path.join(OUTPUT_ROOT, "experiment_log.txt")

    with open(log_path, "w", encoding="utf-8") as log_f:
        _orig_stdout = sys.stdout
        sys.stdout   = Tee(sys.stdout, log_f)

        try:
            all_results: dict = {}
            for lang_code, cfg in LANGUAGES.items():
                try:
                    all_results[lang_code] = run_one_language(
                        lang_code, cfg, OUTPUT_ROOT)
                except Exception as exc:
                    print(f"\n[ERROR] {lang_code}: {exc}")
                    traceback.print_exc()
                    all_results[lang_code] = None

            print_summary(all_results)
            save_csv(all_results, os.path.join(OUTPUT_ROOT, "results.csv"))

        finally:
            sys.stdout = _orig_stdout

    print(f"\nFull log → {log_path}")
