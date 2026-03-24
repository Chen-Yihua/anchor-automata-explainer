#!/usr/bin/env python
"""
Complete comparison of BEAM SEARCH vs SA vs GA vs PSO
with fair parameter settings for reproducibility.
"""

import os
import sys
import random
import numpy as np
import torch
import time
from datetime import datetime

# ── Path setup ─────────────────────────────────────────────────────────
PROJECT_ROOT     = os.path.abspath(os.path.dirname(__file__))
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

print("=" * 80)
print("COMPREHENSIVE COMPARISON: BEAM SEARCH vs SA vs GA vs PSO")
print("=" * 80)

# Test imports
print("\n[1] Testing imports...")
try:
    from modified_modules.alibi.explainers.anchors.anchor_tabular import AnchorTabular
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from search_baselines import build_shared_init, sa_dfa_search, ga_dfa_search, pso_dfa_search
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Create test dataset
print("\n[2] Creating test dataset...")
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                            n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

explainer = AnchorTabular(model.predict, feature_names=[f'f{i+1}' for i in range(10)])
explainer.fit('Tabular', X_train, disc_perc=(25, 50, 75))
print(f"  ✓ Dataset created: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

# Sampler function
def sampler_fn(num_samples=100, compute_labels=False):
    indices = np.random.choice(len(X_train), min(num_samples, len(X_train)), replace=False)
    samples = X_train[indices]
    if compute_labels:
        labels = model.predict(samples)
        return samples.tolist(), labels.tolist()
    return samples.tolist()

# Build shared initialization (used by all methods)
print("\n[3] Building shared initialization...")
shared_init = build_shared_init(
    sampler_fn=sampler_fn,
    data_type="Tabular",
    init_num_samples=200,
    batch_size=100,
    output_dir="test_result/comparison"
)
print(f"  ✓ Shared init: {len(shared_init.initial_dfa.states)} states, "
      f"{len(shared_init.data)} training samples")

# Common parameters
common_params = {
    "accuracy_threshold": 0.9,
    "state_threshold": 15,
    "select_by": "accuracy",
    "shared_init": shared_init,
}

# Fair comparison: all methods get same budget (evaluated candidates)
max_evaluations = 500

results = {}
times = {}

# ============================================================
# BEAM SEARCH
# ============================================================
print("\n" + "=" * 80)
print("[4] Testing BEAM SEARCH")
print("=" * 80)
try:
    t0 = time.time()
    
    beam_result = explainer.anchor_beam(
        type="Tabular",
        automaton_type="DFA",
        accuracy_threshold=common_params["accuracy_threshold"],
        state_threshold=common_params["state_threshold"],
        select_by=common_params["select_by"],
        beam_size=3,
        epsilon=0.1,
        epsilon_stop=0.05,
        delta=0.05,
        batch_size=100,
        init_num_samples=200,
        prebuilt_init=shared_init,
    )
    
    elapsed = time.time() - t0
    times["BEAM"] = elapsed
    results["BEAM"] = beam_result
    
    print(f"\n✓ BEAM SEARCH completed in {elapsed:.2f}s")
    print(f"  States: {beam_result.get('size', 'N/A')}")
    print(f"  Training Accuracy: {beam_result.get('training_accuracy', 'N/A')}")
    print(f"  Success: {beam_result.get('success', False)}")
    
except Exception as e:
    print(f"✗ BEAM SEARCH failed: {e}")
    import traceback
    traceback.print_exc()
    times["BEAM"] = None

# ============================================================
# SA (Simulated Annealing)
# ============================================================
print("\n" + "=" * 80)
print("[5] Testing SA (Simulated Annealing)")
print("=" * 80)
try:
    t0 = time.time()
    
    sa_result = sa_dfa_search(
        sampler_fn=sampler_fn,
        data_type="Tabular",
        init_num_samples=200,
        batch_size=100,
        output_dir="test_result/comparison/sa",
        beam_size=3,
        T_max=10.0,
        T_min=0.001,
        steps=250,  # max_evaluations / n_starts ≈ 500 / 2
        max_evaluations=max_evaluations,
        **common_params,
    )
    
    elapsed = time.time() - t0
    times["SA"] = elapsed
    results["SA"] = sa_result
    
    print(f"\n✓ SA completed in {elapsed:.2f}s")
    print(f"  States: {sa_result.get('size', 'N/A')}")
    print(f"  Training Accuracy: {sa_result.get('training_accuracy', 'N/A')}")
    print(f"  Budget Used: {sa_result.get('budget_used', 'N/A')}")
    print(f"  Success: {sa_result.get('success', False)}")
    
except Exception as e:
    print(f"✗ SA failed: {e}")
    import traceback
    traceback.print_exc()
    times["SA"] = None

# ============================================================
# GA (Genetic Algorithm)
# ============================================================
print("\n" + "=" * 80)
print("[6] Testing GA (Genetic Algorithm)")
print("=" * 80)
try:
    t0 = time.time()
    
    ga_result = ga_dfa_search(
        sampler_fn=sampler_fn,
        data_type="Tabular",
        init_num_samples=200,
        batch_size=100,
        output_dir="test_result/comparison/ga",
        beam_size=3,
        tournament_size=2,
        max_rounds=166,  # max_evaluations / beam_size ≈ 500 / 3
        max_evaluations=max_evaluations,
        **common_params,
    )
    
    elapsed = time.time() - t0
    times["GA"] = elapsed
    results["GA"] = ga_result
    
    print(f"\n✓ GA completed in {elapsed:.2f}s")
    print(f"  States: {ga_result.get('size', 'N/A')}")
    print(f"  Training Accuracy: {ga_result.get('training_accuracy', 'N/A')}")
    print(f"  Budget Used: {ga_result.get('budget_used', 'N/A')}")
    print(f"  Success: {ga_result.get('success', False)}")
    
except Exception as e:
    print(f"✗ GA failed: {e}")
    import traceback
    traceback.print_exc()
    times["GA"] = None

# ============================================================
# PSO (Particle Swarm Optimization)
# ============================================================
print("\n" + "=" * 80)
print("[7] Testing PSO (Particle Swarm Optimization)")
print("=" * 80)
try:
    t0 = time.time()
    
    pso_result = pso_dfa_search(
        sampler_fn=sampler_fn,
        data_type="Tabular",
        init_num_samples=200,
        batch_size=100,
        output_dir="test_result/comparison/pso",
        beam_size=3,
        n_particles=4,
        max_rounds=41,  # max_evaluations / (n_particles × beam_size) ≈ 500 / 12
        max_evaluations=max_evaluations,
        **common_params,
    )
    
    elapsed = time.time() - t0
    times["PSO"] = elapsed
    results["PSO"] = pso_result
    
    print(f"\n✓ PSO completed in {elapsed:.2f}s")
    print(f"  States: {pso_result.get('size', 'N/A')}")
    print(f"  Training Accuracy: {pso_result.get('training_accuracy', 'N/A')}")
    print(f"  Budget Used: {pso_result.get('budget_used', 'N/A')}")
    print(f"  Success: {pso_result.get('success', False)}")
    
except Exception as e:
    print(f"✗ PSO failed: {e}")
    import traceback
    traceback.print_exc()
    times["PSO"] = None

# ============================================================
# SUMMARY COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("FINAL COMPARISON SUMMARY")
print("=" * 80)

comparison_table = []
for method in ["BEAM", "SA", "GA", "PSO"]:
    if method not in results:
        print(f"\n[{method}] - FAILED (no results)")
        continue
    
    result = results[method]
    elapsed = times.get(method, "N/A")
    
    states = result.get('size', 'N/A')
    train_acc = result.get('training_accuracy', 'N/A')
    test_acc = result.get('testing_accuracy', 'N/A')
    budget = result.get('budget_used', 'N/A')
    success = result.get('success', False)
    
    print(f"\n[{method}]")
    print(f"  Time Elapsed:        {elapsed:.2f}s" if isinstance(elapsed, float) else f"  Time Elapsed:        {elapsed}")
    print(f"  Final States:        {states}")
    print(f"  Training Accuracy:   {train_acc:.4f}" if isinstance(train_acc, (int, float)) else f"  Training Accuracy:   {train_acc}")
    print(f"  Testing Accuracy:    {test_acc:.4f}" if isinstance(test_acc, (int, float)) else f"  Testing Accuracy:    {test_acc}")
    print(f"  Budget Used:         {budget}")
    print(f"  Success:             {success}")
    
    if isinstance(states, int) and isinstance(train_acc, (int, float)):
        comparison_table.append({
            "Method": method,
            "Time (s)": elapsed if isinstance(elapsed, float) else 0,
            "States": states,
            "Train Acc": train_acc,
            "Test Acc": test_acc,
            "Budget": budget,
            "Success": success,
        })

# Print comparison table
if comparison_table:
    print("\n" + "=" * 80)
    print("TABLE COMPARISON")
    print("=" * 80)
    print(f"{'Method':<10} {'Time (s)':<12} {'States':<10} {'Train Acc':<12} {'Test Acc':<12} {'Budget':<10} {'Success':<10}")
    print("-" * 90)
    for row in comparison_table:
        print(f"{row['Method']:<10} {row['Time (s)']:<12.2f} {row['States']:<10} "
              f"{row['Train Acc']:<12.4f} {row['Test Acc']:<12.4f} {row['Budget']:<10} {row['Success']:<10}")

print("\n" + "=" * 80)
print(f"Comparison completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
