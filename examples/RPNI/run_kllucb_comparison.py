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
# Helper function to safely convert values to scalar
# ======================================================================
def to_scalar(value):
    """Convert value to scalar, handling lists, arrays, and other types."""
    if value is None:
        return 0
    # If it's a list or array, take the first element
    if isinstance(value, (list, tuple)):
        return value[0] if value else 0
    # If it's a numpy array, convert to Python scalar
    if isinstance(value, np.ndarray):
        return value.item()
    # Otherwise return as-is
    return value


# ======================================================================
# Shared initialization container
# ======================================================================
# Contains: learner, initial_dfa (shared), validation_data/labels (for selection)
# Training data is NOT included — anchor_beam will resample it independently
PrebuiltInit = namedtuple('PrebuiltInit', ['learner', 'initial_dfa', 'validation_data', 'validation_labels'])


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
            tau             = 0.1,
            batch_size      = 1000,
            beam_size       = 1,
            init_num_samples= 2000,
            edit_distance   = 4,
            select_by       = 'accuracy',
            max_length      = 10, embedding_dim = 16, hidden_dim = 32,
            num_layers      = 1,  dropout = 0,
        ),
        "L4": dict(
            test_instance   = ['a', 'a', 'b', 'b', 'c', 'd', 'c'] ,
            alphabet        = ['a', 'b', 'c', 'd'],
            accuracy_threshold = accuracy_threshold,
            state_threshold = 5,
            delta           = 0.01,
            tau             = 0.1,
            batch_size      = 1000,
            beam_size       = 1,
            init_num_samples= 2000,
            edit_distance   = 5,
            select_by       = 'accuracy',
            max_length      = 10, embedding_dim = 16, hidden_dim = 64,
            num_layers      = 2,  dropout = 0.5,
        ),
        "L6": dict(
            test_instance   = ['a', 'a', 'a', 'c', 'a', 'a'],
            alphabet        = ['a', 'b', 'c', 'd'],
            accuracy_threshold = accuracy_threshold,
            state_threshold = 5,
            delta           = 0.01,
            tau             = 0.1,
            batch_size      = 1000,
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
            tau             = 0.1,
            batch_size      = 1000,
            beam_size       = 1,
            init_num_samples= 1000,
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
            tau             = 0.1,
            batch_size      = 1000,
            beam_size       = 1,
            init_num_samples= 800,
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
    
    # Convert all values to scalar (handles lists, arrays, and numpy types)
    initial_state = int(to_scalar(getattr(beam_with_kllucb, 'initial_state')))
    initial_train = float(to_scalar(getattr(beam_with_kllucb, 'initial_training_accuracy')))
    initial_validation = float(to_scalar(getattr(beam_with_kllucb, 'initial_validation_accuracy')))
    final_train  = float(to_scalar(getattr(beam_with_kllucb, 'final_training_accuracy')))
    final_validation = float(to_scalar(getattr(beam_with_kllucb, 'final_validation_accuracy')))
    final_state = int(to_scalar(getattr(beam_with_kllucb, 'final_state')))
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
        initial_dfa=initial_dfa,
        validation_data=validation_data,
        validation_labels=validation_labels
    )
    sys.stdout.flush()

    results['with_kllucb'] = {
        'initial_state': initial_state,
        'initial_training_accuracy': initial_train,
        'initial_validation_accuracy': initial_validation,
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

    # Convert all values to scalar (handles lists, arrays, and numpy types)
    initial_state_no = int(to_scalar(getattr(beam_no_kllucb, 'initial_state')))
    initial_train_no = float(to_scalar(getattr(beam_no_kllucb, 'initial_training_accuracy')))
    initial_validation_no = float(to_scalar(getattr(beam_no_kllucb, 'initial_validation_accuracy')))
    final_train_no  = float(to_scalar(getattr(beam_no_kllucb, 'final_training_accuracy')))
    final_validation_no = float(to_scalar(getattr(beam_no_kllucb, 'final_validation_accuracy')))
    final_state_no = int(to_scalar(getattr(beam_no_kllucb, 'final_state')))
    budget_used = getattr(beam_no_kllucb, 'budget_used', None)
    success = getattr(beam_no_kllucb, 'success', False)

    results['no_kllucb'] = {
        'initial_state': initial_state_no,  # Same for both methods
        'initial_training_accuracy': initial_train_no,
        'initial_validation_accuracy': initial_validation_no,
        'shared_init_used': True,
        'final_state': final_state_no,
        'final_training_accuracy': final_train_no,
        'final_validation_accuracy': final_validation_no,
        'time_total': time_no_kllucb,
        'init_time': init_time_no,
        'beam_time': beam_time_no,
        'success': success,
        'budget_used': budget_used,
    }
    
    print(f"    NO KL-LUCB:     train_acc={results['no_kllucb']['final_training_accuracy']:.4f}, "
          f"states={results['no_kllucb']['final_state']}, "
          f"beam_time={beam_time_no:.1f}s")

    return results


# ======================================================================
# Summary Report
# ======================================================================
def print_summary_report(all_results: dict, accuracy_threshold: float) -> None:
    """Print summary comparison across all languages."""
    print(f"\n\n{'='*145}")
    print(f"  SUMMARY: KL-LUCB Comparison (accuracy_threshold={accuracy_threshold})")
    print(f"{'='*145}\n")

    # Header row
    col_width = 15
    print(f"{'Language':<12} {'Method':<18} {'Init Train':<12} {'Final Train':<12} {'Δ Train':<12} "
          f"{'Init Val':<12} {'Final Val':<12} {'Δ Val':<12} {'Init States':<12} {'Final States':<12} {'Δ States':<12} {'Time(s)':<12}")
    print("─" * 145)

    for lang_code, results in sorted(all_results.items()):
        if results is None:
            continue

        with_r = results['with_kllucb']
        no_r = results['no_kllucb']
        
        # Extract all values and convert to scalar (handles lists, arrays, and numpy types)
        init_state = int(to_scalar(with_r.get('initial_state', 0)))
        init_train_acc = float(to_scalar(with_r.get('initial_training_accuracy', 0.0)))
        init_val_acc = float(to_scalar(with_r.get('initial_validation_accuracy', 0.0)))
        
        final_train_with = float(to_scalar(with_r.get('final_training_accuracy', 0.0)))
        final_train_no = float(to_scalar(no_r.get('final_training_accuracy', 0.0)))
        delta_train_with = final_train_with - init_train_acc
        delta_train_no = final_train_no - init_train_acc
        
        final_val_with = float(to_scalar(with_r.get('final_validation_accuracy', 0.0)))
        final_val_no = float(to_scalar(no_r.get('final_validation_accuracy', 0.0)))
        delta_val_with = final_val_with - init_val_acc
        delta_val_no = final_val_no - init_val_acc
        
        final_state_with = int(to_scalar(with_r.get('final_state', 0)))
        final_state_no = int(to_scalar(no_r.get('final_state', 0)))
        delta_state_with = final_state_with - init_state
        delta_state_no = final_state_no - init_state
        
        time_with = float(to_scalar(with_r.get('beam_time', 0.0)))
        time_no = float(to_scalar(no_r.get('beam_time', 0.0)))

        # WITH KL-LUCB row
        print(f"{lang_code:<12} {'WITH KL-LUCB':<18} {init_train_acc:<12.4f} {final_train_with:<12.4f} {delta_train_with:+12.4f} "
              f"{init_val_acc:<12.4f} {final_val_with:<12.4f} {delta_val_with:+12.4f} {init_state:<12d} {final_state_with:<12d} {delta_state_with:+12d} {time_with:<12.2f}")

        # WITHOUT KL-LUCB row
        print(f"{'':<12} {'WITHOUT KL-LUCB':<18} {init_train_acc:<12.4f} {final_train_no:<12.4f} {delta_train_no:+12.4f} "
              f"{init_val_acc:<12.4f} {final_val_no:<12.4f} {delta_val_no:+12.4f} {init_state:<12d} {final_state_no:<12d} {delta_state_no:+12d} {time_no:<12.2f}")

        print("─" * 145)


# ======================================================================
# Entry point
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare beam search WITH and WITHOUT KL-LUCB")
    parser.add_argument("--accuracy_threshold", type=float, default=0.8,
                        help="Accuracy threshold for all languages (default: 0.8)")
    parser.add_argument("--languages", type=str, default="L3AB,L4,L6,L7,mnist",
                        help="Comma-separated list of languages to run (default: L3AB,L4,L6,L7,mnist)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="(optional) Override per-language batch_size for quick tests")
    parser.add_argument("--init_num_samples", type=int, default=None,
                        help="(optional) Override per-language init_num_samples for quick tests")
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
        # Optionally override per-language batch_size / init_num_samples for quick testing
        if args.batch_size is not None or args.init_num_samples is not None:
            for lc, cfg in cfg_dict.items():
                if args.batch_size is not None:
                    cfg['batch_size'] = args.batch_size
                if args.init_num_samples is not None:
                    cfg['init_num_samples'] = args.init_num_samples
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

        # Print detailed results at the end
        print("\n\n" + "="*95)
        print(f"  DETAILED RESULTS")
        print("="*95)
        
        for lang_code, results in sorted(all_results.items()):
            if results is None:
                print(f"\n[{lang_code}] SKIPPED (failed or not found)")
                continue
            
            with_r = results['with_kllucb']
            no_r = results['no_kllucb']
            
            print(f"\n[{lang_code}]")
            print(f"  WITH KL-LUCB:")
            print(f"    Initial: {int(to_scalar(with_r.get('initial_state', 0)))} states, {float(to_scalar(with_r.get('initial_training_accuracy', 0.0))):.4f} train acc, {float(to_scalar(with_r.get('initial_validation_accuracy', 0.0))):.4f} val acc")
            print(f"    Final:   {int(to_scalar(with_r.get('final_state', 0)))} states, {float(to_scalar(with_r.get('final_training_accuracy', 0.0))):.4f} train acc, {float(to_scalar(with_r.get('final_validation_accuracy', 0.0))):.4f} val acc")
            print(f"    Time:    {float(to_scalar(with_r.get('beam_time', 0.0))):.2f}s")
            print(f"    Success: {with_r.get('success', False)}")
            
            print(f"  WITHOUT KL-LUCB:")
            print(f"    Initial: {int(to_scalar(no_r.get('initial_state', 0)))} states, {float(to_scalar(no_r.get('initial_training_accuracy', 0.0))):.4f} train acc, {float(to_scalar(no_r.get('initial_validation_accuracy', 0.0))):.4f} val acc")
            print(f"    Final:   {int(to_scalar(no_r.get('final_state', 0)))} states, {float(to_scalar(no_r.get('final_training_accuracy', 0.0))):.4f} train acc, {float(to_scalar(no_r.get('final_validation_accuracy', 0.0))):.4f} val acc")
            print(f"    Time:    {float(to_scalar(no_r.get('beam_time', 0.0))):.2f}s")
            print(f"    Success: {no_r.get('success', False)}")
        
        print("\n" + "="*95)
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
