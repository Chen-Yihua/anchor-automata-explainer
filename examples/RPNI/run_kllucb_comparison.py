"""
Beam Search Comparison: WITH KL-LUCB vs WITHOUT KL-LUCB
=======================================================
Compares beam search performance when using KL-LUCB-based candidate selection
vs simple greedy top-k selection based on training accuracy.

Usage (from project root):
    python examples/RPNI/run_kllucb_comparison.py --accuracy_threshold 0.9
    python examples/RPNI/run_kllucb_comparison.py --accuracy_threshold 0.8

Results will be saved to:
    test_result/kllucb_comparison/accuracy_threshold_{threshold}/
"""

from __future__ import annotations
import argparse
import copy
import os
import pickle
import random
import sys
import time
from typing import Optional
from collections import namedtuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score

# ── Path setup ─────────────────────────────────────────────────────────
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
from tee import Tee


# ======================================================================
# Shared initialization container
# ======================================================================
PrebuiltInit = namedtuple('PrebuiltInit', ['learner', 'data', 'labels', 'initial_dfa'])


# ======================================================================
# Language configurations
# ======================================================================
def get_languages_config(accuracy_threshold):
    """Get language configurations with parameterized accuracy threshold."""
    return {
        "L3AB": dict(
            test_instance   = ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'],
            alphabet        = ['a', 'b', 'c', 'd'],
            accuracy_threshold = accuracy_threshold,
            state_threshold = 10,
            delta           = 0.01,
            tau             = 0.05,
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
            accuracy_threshold = accuracy_threshold,
            state_threshold = 5,
            delta           = 0.01,
            tau             = 0.05,
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
            accuracy_threshold = accuracy_threshold,
            state_threshold = 5,
            delta           = 0.01,
            tau             = 0.05,
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
            accuracy_threshold = accuracy_threshold,
            state_threshold = 5,
            delta           = 0.01,
            tau             = 0.05,
            batch_size      = 500,
            beam_size       = 1,
            init_num_samples= 2000,
            edit_distance   = 5,
            select_by       = 'accuracy',
            max_length      = 10, embedding_dim = 16, hidden_dim = 32,
            num_layers      = 1,  dropout = 0,
        ),
        "mnist": dict(
            test_instance   = None,  # Will be selected from training set
            alphabet        = [(0, 1), (-1, 1), (1, 1), (1, 0), (1, -1), (-1, -1), (-1, 0), (0, -1), (0, 0)],
            accuracy_threshold = accuracy_threshold,
            state_threshold = 5,
            delta           = 0.01,
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


# ======================================================================
# Single-language comparison
# ======================================================================
def run_one_language(lang_code: str, cfg: dict, output_root: str) -> dict | None:
    """Run beam search comparison for one language."""
    print(f"\n{'='*80}")
    print(f"  LANGUAGE: {lang_code}")
    print(f"{'='*80}")

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
        if lang_code == "mnist":
            test_instance = X_train[23]
        else:
            positive_indices = [i for i, label in enumerate(y_train) if label == 1]
            test_instance = X_train[positive_indices[2]]  # Fallback to first instance if no positives

    # ── Create explainer ───────────────────────────────────────────────
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

    results = {}

    # ─────────────────────────────────────────────────────────────────
    # METHOD 1: Beam Search WITH KL-LUCB (from same initial automaton)
    # ─────────────────────────────────────────────────────────────────
    print("\n  Beam Search WITH KL-LUCB")
    t0 = time.time()
    try:
        beam_with_kllucb = explainer.explain(
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
            output_dir         = os.path.join(out_dir, "beam_with_kllucb"),
            use_kllucb         = True,  # WITH KL-LUCB
            prebuilt_init      = None,  # Use shared initial automaton
        )
    except Exception as e:
        print(f"  [ERROR] Beam Search WITH KL-LUCB failed: {e}")
        import traceback
        traceback.print_exc()
        return None  # Skip this language if first method fails
    
    time_with_kllucb = time.time() - t0
    init_time_with = getattr(beam_with_kllucb, 'init_automaton_time', 0.0)
    beam_time_with = max(0.0, time_with_kllucb - init_time_with)
    
    # ─── Extract shared initialization from explainer.mab ───
    print("  Extracting shared initialization...")
    sys.stdout.flush()
    
    initial_state = getattr(beam_with_kllucb, 'initial_state')
    initial_train = getattr(beam_with_kllucb, 'initial_training_accuracy')
    initial_validation = getattr(beam_with_kllucb, 'initial_validation_accuracy')
    final_train  = getattr(beam_with_kllucb, 'final_training_accuracy')
    final_validation = getattr(beam_with_kllucb, 'final_validation_accuracy')
    final_state = getattr(beam_with_kllucb, 'final_state')
    budget_used = getattr(beam_with_kllucb, 'budget_used', None)
    success = getattr(beam_with_kllucb, 'success', False)
    mab = getattr(explainer, 'mab', None)
    automatas = getattr(mab, 'automatas', None) or []
    initial_dfa = automatas[0].copy()
    validation_data = list(getattr(mab, 'validation_data'))
    validation_labels = np.asarray(getattr(mab, 'validation_labels'))

    print(f"  Extracted {len(validation_data)} initial samples, DFA with {initial_dfa.size} states")
    sys.stdout.flush()
    
    # Create a learner instance for PrebuiltInit (will be used in second explain call)
    from learner.dfa_learner import DFALearner
    learner_instance = DFALearner()
    
    # Create PrebuiltInit for sharing with second method
    prebuilt = PrebuiltInit(
        learner=learner_instance,
        data=validation_data,
        labels=validation_labels,
        initial_dfa=initial_dfa
    )
    sys.stdout.flush()

    results['with_kllucb'] = {
        'initial_state': initial_state,
        'initial_training_accuracy': initial_train,
        'final_validation_accuracy': initial_validation,
        'shared_init_used': True,  # Mark that shared initialization was used
        'final_state': final_state,
        'final_training_accuracy': final_train,
        'final_validation_accuracy': final_validation,
        'time_total': time_with_kllucb,
        'init_time': init_time_with,
        'beam_time': beam_time_with,
        'success': success,
        'budget_used': budget_used,
    }
    
    print(f"    WITH KL-LUCB:  train_acc={results['with_kllucb']['final_training_accuracy']:.4f}, "
          f"states={results['with_kllucb']['final_state']}, "
          f"beam_time={beam_time_with:.1f}s")

    # ─────────────────────────────────────────────────────────────────
    # METHOD 2: Beam Search WITHOUT KL-LUCB (from same initial automaton)
    # ─────────────────────────────────────────────────────────────────
    print("\n  Beam Search WITHOUT KL-LUCB (using shared initial automaton)...")
    sys.stdout.flush()
    
    # Reset sampler state for second explain call
    sys.stdout.flush()
    for sampler in explainer.samplers:
        sampler.set_instance_label(test_instance)
        sampler.set_n_covered(10)
        sampler.edit_distance = cfg['edit_distance']
    sys.stdout.flush()
    
    t0 = time.time()
    try:
        beam_no_kllucb = explainer.explain(
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
            output_dir         = os.path.join(out_dir, "beam_no_kllucb"),
            use_kllucb         = False,  # NO KL-LUCB
            prebuilt_init      = prebuilt,  # Use shared PrebuiltInit object
        )
        sys.stdout.flush()
    except Exception as e:
        print(f"  [ERROR] Beam Search WITHOUT KL-LUCB failed: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None  # Skip this language if second method fails
    
    time_no_kllucb = time.time() - t0
    init_time_no = getattr(beam_no_kllucb, 'init_automaton_time', 0.0)
    beam_time_no = max(0.0, time_no_kllucb - init_time_no)

    initial_state = getattr(beam_no_kllucb, 'initial_state')
    initial_train = getattr(beam_no_kllucb, 'initial_training_accuracy')
    initial_validation = getattr(beam_no_kllucb, 'initial_validation_accuracy')
    final_train  = getattr(beam_no_kllucb, 'final_training_accuracy')
    final_validation = getattr(beam_no_kllucb, 'final_validation_accuracy')
    final_state = getattr(beam_no_kllucb, 'final_state')
    budget_used = getattr(beam_no_kllucb, 'budget_used', None)
    success = getattr(beam_no_kllucb, 'success', False)

    results['no_kllucb'] = {
        'initial_state': initial_state,  # Same for both methods
        'initial_training_accuracy': initial_train,
        'final_validation_accuracy': initial_validation,
        'shared_init_used': True,
        'final_state': final_state,
        'final_training_accuracy': final_train,
        'final_validation_accuracy': final_validation,
        'time_total': time_no_kllucb,
        'init_time': init_time_no,
        'beam_time': beam_time_no,
        'success': success,
        'budget_used': budget_used,
    }
    
    print(f"    NO KL-LUCB:     train_acc={results['no_kllucb']['final_training_accuracy']:.4f}, "
          f"states={results['no_kllucb']['final_state']}, "
          f"beam_time={beam_time_no:.1f}s")

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Comparison & Analysis
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print(f"  COMPARISON for {lang_code} (starting from SAME shared initial automaton):")
    print("─"*80)
    
    print(f"\n  Initial Automaton (RPNI baseline - SHARED by both methods):")
    print(f"    States:       {initial_state}")
    print(f"    Train Acc:    {initial_train:.4f}")
    
    acc_with = results['with_kllucb']['final_training_accuracy']
    acc_no = results['no_kllucb']['final_training_accuracy']
    state_with = results['with_kllucb']['final_state']
    state_no = results['no_kllucb']['final_state']
    time_with = results['with_kllucb']['beam_time']
    time_no = results['no_kllucb']['beam_time']

    print(f"\n  Final Training Accuracy:")
    print(f"    WITH KL-LUCB: {acc_with:.4f} (Δ {acc_with - initial_train:+.4f})")
    print(f"    NO KL-LUCB:   {acc_no:.4f} (Δ {acc_no - initial_train:+.4f})")
    print(f"    Difference:   {abs(acc_with - acc_no):.4f} {'✓ Similar' if abs(acc_with - acc_no) < 0.01 else '✗ Different'}")

    print(f"\n  Final State Count:")
    print(f"    WITH KL-LUCB: {state_with} (Δ {state_with - initial_state:+d})")
    print(f"    NO KL-LUCB:   {state_no} (Δ {state_no - initial_state:+d})")
    print(f"    Difference:   {abs(state_with - state_no)} {'✓ Same' if abs(state_with - state_no) == 0 else '✗ Different'}")

    print(f"\n  Beam Search Time (excluding RPNI):")
    print(f"    WITH KL-LUCB: {time_with:.2f}s")
    print(f"    NO KL-LUCB:   {time_no:.2f}s")
    print(f"    Speedup:      {time_with/max(time_no, 0.01):.2f}x {'(KL-LUCB slower - uses statistical testing)' if time_with > time_no else '(No KL-LUCB faster - simpler)' }")

    print(f"\n  Validation Accuracy:")
    print(f"    WITH KL-LUCB: {results['with_kllucb']['final_validation_accuracy']:.4f}")
    print(f"    NO KL-LUCB:   {results['no_kllucb']['final_validation_accuracy']:.4f}")

    return results


# ======================================================================
# Summary Report
# ======================================================================
def print_summary_report(all_results: dict, accuracy_threshold: float) -> None:
    """Print summary comparison across all languages."""
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY: KL-LUCB Comparison (accuracy_threshold={accuracy_threshold})")
    print(f"{'='*80}\n")

    print(f"{'Language':<12} {'Metric':<20} {'Initial':<12} {'WITH KL':<12} {'WITHOUT KL':<12}")
    print("─" * 80)

    for lang_code, results in sorted(all_results.items()):
        if results is None:
            continue

        with_r = results['with_kllucb']
        no_r = results['no_kllucb']
        init_state = with_r['initial_state']
        init_acc = with_r['initial_training_accuracy']

        # Row 1: Final Accuracy
        print(f"{lang_code:<12} {'Final Train Acc':<20} {init_acc:<12.4f} "
              f"{with_r['final_training_accuracy']:<12.4f} {no_r['final_training_accuracy']:<12.4f}")

        # Row 2: Final States
        print(f"{'':<12} {'Final States':<20} {init_state:<12} "
              f"{with_r['final_state']:<12} {no_r['final_state']:<12}")

        # Row 3: Beam Time
        print(f"{'':<12} {'Beam Time (s)':<20} {'-':<12} "
              f"{with_r['beam_time']:<12.2f} {no_r['beam_time']:<12.2f}")

        # Row 4: Validation Acc
        print(f"{'':<12} {'Validation Acc':<20} {'-':<12} "
              f"{with_r['final_validation_accuracy']:<12.4f} {no_r['final_validation_accuracy']:<12.4f}")

        print("─" * 80)

    print("\nConclusion:")
    print("  • Shared initial automaton (via prebuilt_init) eliminates initialization variance")
    print("  • Differences in final results are purely due to Beam Search strategy, not RPNI randomness")
    print("  • If final results are similar → KL-LUCB overhead may not be justified")
    print("  • If WITH KL-LUCB achieves higher accuracy → statistical testing is beneficial")
    print("  • If WITHOUT KL-LUCB is much faster → consider simpler greedy selection")


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare beam search WITH and WITHOUT KL-LUCB")
    parser.add_argument("--accuracy_threshold", type=float, default=0.9,
                        help="Accuracy threshold for all languages (default: 0.9)")
    parser.add_argument("--languages", type=str, default="L3AB,L4,L6,L7,mnist",
                        help="Comma-separated list of languages to run (default: L3AB,L4,L6,L7,mnist)")
    args = parser.parse_args()

    accuracy_threshold = args.accuracy_threshold
    languages_to_run = [l.strip() for l in args.languages.split(",")]

    output_root = os.path.join(PROJECT_ROOT, "test_result", "kllucb_comparison",
                               f"accuracy_threshold_{accuracy_threshold}")
    os.makedirs(output_root, exist_ok=True)

    # Redirect output to file + console
    log_path = os.path.join(output_root, "comparison.log")
    print(f"\n{'='*80}")
    print(f"  Logging to: {log_path}")
    print(f"{'='*80}\n")
    
    # Use Tee for logging, with automatic flushing
    tee = Tee(log_path)
    
    try:
        all_results = {}
        
        cfg_dict = get_languages_config(accuracy_threshold)
        for lang_code in languages_to_run:
            if lang_code not in cfg_dict:
                print(f"\n[SKIP] Unknown language: {lang_code}")
                sys.stdout.flush()  # Force flush
                continue

            print(f"\n[STARTING] Processing {lang_code}...")
            sys.stdout.flush()
            
            try:
                result = run_one_language(lang_code, cfg_dict[lang_code], output_root)
                if result is not None:
                    all_results[lang_code] = result
                    print(f"[COMPLETED] {lang_code} succeeded")
                else:
                    print(f"[FAILED] {lang_code} returned None")
            except Exception as e:
                print(f"[ERROR] {lang_code} failed with exception: {e}")
                import traceback
                traceback.print_exc()
            
            sys.stdout.flush()  # Force flush after each language

        # Print summary
        print_summary_report(all_results, accuracy_threshold)
        sys.stdout.flush()

        # Save detailed results to pickle
        results_path = os.path.join(output_root, "kllucb_comparison_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\nDetailed results saved to: {results_path}")
        print(f"Main log saved to: {log_path}")
        sys.stdout.flush()

    except Exception as e:
        print(f"\n[FATAL ERROR] Unexpected exception: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    
    finally:
        tee.close()
        print("[SCRIPT COMPLETED]")
