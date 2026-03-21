"""
Baseline Experiment: Parameterized by accuracy_threshold
========================================================
Runs all four DFA search methods with configurable accuracy_threshold.

Usage (from project root):
    python examples/RPNI/run_baseline_experiment_parameterized.py --accuracy_threshold 0.8
    python examples/RPNI/run_baseline_experiment_parameterized.py --accuracy_threshold 0.9
    
Results will be saved to:
    test_result/baseline_experiment/accuracy_threshold_{threshold}/
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import pickle
import random
import sys
import time
import traceback
from typing import Optional

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
from search_baselines import sa_dfa_search, ga_dfa_search, pso_dfa_search, SharedInit
from tee import Tee


# ======================================================================
# Per-language hyperparameter configurations  (mirrors run_tomita.py)
# ======================================================================
def get_languages_config(accuracy_threshold):
    """
    Get language configurations.
    
    Parameters
    ----------
    accuracy_threshold_override : float, optional
        If provided, override all language thresholds with this value.
        If None, use per-language defaults (L3AB/L4=0.9, L6/L7/mnist=0.8)
    
    Returns
    -------
    dict : Language configurations
    """
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

# ═══════════════════════════════════════════════════════════════════════════
# Search strategy: Fair comparison with unified budget
# ═══════════════════════════════════════════════════════════════════════════
# All methods (SA, GA, PSO) use same budget as Beam Search: measure actual candidates generated
#
# MAX_EVALUATIONS_DEFAULT = 500  # Fallback if beam search stats unavailable
SA_STEPS         = 100  # SA steps per start (full cooling schedule)
SA_BEAM_SIZE     = 1    # SA: single neighbor per step (sequential search)
PSO_PARTICLES    = 20   # PSO particle count
PSO_BEAM_SIZE    = 1    # PSO: single candidate per particle (swarm provides diversity)
GA_BEAM_SIZE     = 5    # GA: 5 candidates per mutation (tournament selection)


# ======================================================================
# Accuracy extraction helpers
# ======================================================================
def _extract_accuracy(value) -> float:
    """
    Extract scalar accuracy from various formats returned by search methods.
    
    Handles:
    - Scalar float/int: return as float
    - List (from anchor_base.py): return first element or mean
    - None/zero: return 0.0
    """
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        return float(np.mean(value)) if len(value) > 1 else float(value[0])
    return float(value)


def eval_validation_acc(dfa, val_data, val_labels, learner) -> float:
    """
    DFA accuracy on the validation set (initial DFA training samples).

    val_labels[i] = 1 if model predicts positive class
                    0 if model predicts negative class
    """
    if dfa is None or len(val_data) == 0:
        return 0.0
    accepts = np.array([learner.check_path_accepted(dfa, seq) for seq in val_data])
    lbl     = np.asarray(val_labels)
    correct = int(np.sum((lbl == 1) & accepts) + np.sum((lbl == 0) & ~accepts))
    return correct / len(val_data)


def load_beam_results_from_disk(output_dir: str) -> Optional[dict]:
    """
    Load beam search results from disk (previously saved).
    
    Looks for lang_code/shared/beam_results.pkl
    
    Returns
    -------
    dict with keys: 'initial_state', 'initial_training_accuracy', 'final_state', etc.
    OR None if file does not exist or load failed
    """
    if output_dir is None:
        return None

    shared_dir = os.path.join(output_dir, "shared")
    beam_results_path = os.path.join(shared_dir, "beam_results.pkl")

    if not os.path.exists(beam_results_path):
        print(f"  [BeamResults] No cached beam results found at {beam_results_path}.")
        return None
    try:
        with open(beam_results_path, 'rb') as f:
            beam_res = pickle.load(f)
        
        if not isinstance(beam_res, dict):
            print(f"  [BeamResults] WARNING: Loaded object is not dict type.")
            return None
        
        print(f"  [BeamResults] Loaded from disk: train_acc={beam_res['final_training_accuracy']:.4f}, "
              f"validation_acc={beam_res['final_validation_accuracy']:.4f}, states={beam_res['final_state']}")
        return beam_res
    except Exception as exc:
        print(f"  [BeamResults] Failed to load from disk: {exc}")
        return None


def load_shared_init_from_disk(output_dir: str) -> Optional[SharedInit]:
    """
    Load shared initialization from disk (previously saved by beam search).
    
    Looks for lang_code/shared/shared_init.pkl
    
    Returns
    -------
    SharedInit object with initial_dfa, learner, validation_data, validation_labels
    OR None if file does not exist or load failed
    """
    if output_dir is None:
        return None

    shared_dir = os.path.join(output_dir, "shared")
    shared_init_path = os.path.join(shared_dir, "shared_init.pkl")

    if not os.path.exists(shared_init_path):
        print(f"  [SharedInit] No cached shared init found at {shared_init_path}.")
        return None

    try:
        with open(shared_init_path, 'rb') as f:
            shared_init = pickle.load(f)
        
        if not isinstance(shared_init, SharedInit):
            print(f"  [SharedInit] WARNING: Loaded object is not SharedInit type.")
            return None
        
        print(f"  [SharedInit] Loaded from disk: {len(shared_init.initial_dfa.states)} states, "
              f"{len(shared_init.validation_data)} test samples")
        return shared_init
    except Exception as exc:
        print(f"  [SharedInit] Failed to load from disk: {exc}")
        return None


def save_beam_results_to_disk(beam_results: dict, output_dir: str = None) -> bool:
    """
    Save beam search results to disk.
    
    Parameters
    ----------
    beam_results : dict with keys 'train_acc', 'validation_acc', 'states', 'time', 'success'
    output_dir   : language directory (lang_code/)
    
    Returns
    -------
    True if saved successfully, False otherwise
    """
    if output_dir is None or beam_results is None:
        return False

    shared_dir = os.path.join(output_dir, "shared")
    try:
        os.makedirs(shared_dir, exist_ok=True)
        beam_results_path = os.path.join(shared_dir, "beam_results.pkl")
        with open(beam_results_path, 'wb') as f:
            pickle.dump(beam_results, f)
        print(f"  [BeamResults] Saved to {beam_results_path}")
        return True
    except Exception as exc:
        print(f"  [WARNING] Could not save beam results: {exc}")
        return False


def build_shared_init_from_beam(explainer, output_dir: str = None) -> Optional[SharedInit]:
    """
    Extract shared initialization data from beam search result and save to disk.
    
    Parameters
    ----------
    explainer       : AnchorTabular object with mab (multi-armed bandit) set after explain()
    output_dir      : language directory (lang_code/) to save shared/ subfolder
    
    Returns
    -------
    SharedInit object with: initial_dfa, learner, validation_data, validation_labels
    OR None if extraction failed
    """
    mab = getattr(explainer, 'mab', None)
    if mab is None:
        print("  [ERROR] Beam search mab not available.")
        return None

    # Extract initial DFA (best automaton from beam search)
    automatas = getattr(mab, 'automatas', None) or []
    if not automatas:
        print("  [ERROR] Beam search produced no automata.")
        return None
    
    initial_dfa = automatas[0]
    if hasattr(initial_dfa, 'copy'):
        initial_dfa = initial_dfa.copy()
    else:
        initial_dfa = copy.deepcopy(initial_dfa)

    # Extract validation data and labels (these are validation samples from beam)
    validation_data = list(getattr(mab, 'validation_data'))
    validation_labels = np.asarray(getattr(mab, 'validation_labels'))
    
    if len(validation_data) == 0:
        print("  [ERROR] Beam search validation_data missing.")
        return None

    # Get learner instance from mab
    from learner.dfa_learner import DFALearner
    learner = DFALearner()

    # Create SharedInit object
    shared_init = SharedInit(
        initial_dfa=initial_dfa,
        learner=learner,
        validation_data=validation_data,
        validation_labels=validation_labels
    )

    # Save to disk under lang_code/shared/
    if output_dir is not None:
        shared_dir = os.path.join(output_dir, "shared")
        try:
            os.makedirs(shared_dir, exist_ok=True)
            shared_init_path = os.path.join(shared_dir, "shared_init.pkl")
            with open(shared_init_path, 'wb') as f:
                pickle.dump(shared_init, f)
            print(f"  [SharedInit] Saved to {shared_init_path}")
            print(f"    ├─ initial_dfa: {len(initial_dfa.states)} states")
            print(f"    ├─ validation_data: {len(validation_data)} samples")
            print(f"    └─ validation_labels: {len(validation_labels)} labels")
        except Exception as exc:
            print(f"  [WARNING] Could not save shared init: {exc}")

    return shared_init


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
        if lang_code == "mnist":
            test_instance = X_train[23]  # 23 is a '7' with good stroke variety
        else:
            positive_indices = [i for i, label in enumerate(y_train) if label == 1]
            test_instance = X_train[positive_indices[2]]  # Fallback to first instance if no positives
   
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

    # ══════════════════════════════════════════════════════════════════
    # Method 1 – Beam Search  (anchor_beam via explain())
    # Check for cached shared_init and beam_results first; if not found, run beam search
    # ══════════════════════════════════════════════════════════════════
    print("\n  ─── Beam Search (First - to create/load shared_init) ─────────────────────────")

    # Check if both shared_init and beam_results are cached
    shared = load_shared_init_from_disk(out_dir)
    beam_cached = load_beam_results_from_disk(out_dir)

    if shared is not None and beam_cached is not None:
        # Both cached - use them, skip explainer.explain
        print("    [CACHED] Using previously saved shared_init and beam results.")
        initial_state = beam_cached['initial_state']
        initial_train = beam_cached['initial_training_accuracy']
        initial_validation = beam_cached['initial_validation_accuracy']
        final_train = beam_cached['final_training_accuracy']
        final_validation = beam_cached['final_validation_accuracy']
        final_state = beam_cached['final_state']
        beam_time = beam_cached['time']
        beam_success = beam_cached['success']
        budget_used = beam_cached['budget_used']
    else:
        # No cached data - need to run beam search
        if shared is not None:
            print("    [INFO] Found shared_init but no beam results; running beam search.")
        else:
            print("    [INFO] No cached data found; running beam search to generate initial DFA.")

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
        )
        beam_time_total = time.time() - t0
        init_time = getattr(beam_expl, 'init_automaton_time', 0.0)
        beam_time = max(0.0, beam_time_total - init_time)

        initial_state = getattr(beam_expl, 'initial_state')
        initial_train = getattr(beam_expl, 'initial_training_accuracy')
        initial_validation = getattr(beam_expl, 'initial_validation_accuracy')
        final_train  = getattr(beam_expl, 'final_training_accuracy')
        final_validation = getattr(beam_expl, 'final_validation_accuracy')
        final_state = getattr(beam_expl, 'final_state')
        beam_success = getattr(beam_expl, 'success')
        budget_used = getattr(beam_expl, 'budget_used', None)
        print(f"    train={final_train:.4f}  states={final_state}  time={beam_time:.1f}s"
              f"  {'✓' if beam_success else '✗'}")

        # Save shared_init from beam search
        shared = build_shared_init_from_beam(explainer, out_dir)

        # Save beam results to disk for future runs
        beam_results_dict = {
            'initial_state': initial_state,
            'initial_training_accuracy': initial_train,
            'initial_validation_accuracy': initial_validation,
            'final_training_accuracy': final_train,
            'final_validation_accuracy': final_validation,
            'final_state': final_state,
            'time': beam_time,
            'success': beam_success,
            'budget_used': budget_used,
        }
        save_beam_results_to_disk(beam_results_dict, out_dir)

    if shared is None:
        print("  [SKIP] Could not extract shared_init from beam search.")
        return None
    
    # Use beam search's actual budget for SA/GA/PSO (fair comparison)
    print(f"  [Budget] Setting SA/GA/PSO max_evaluations={budget_used} beam search)")
    
    results: dict = {
        "lang"              : lang_code,
        "clf_train_acc"     : clf_train_acc,
        "clf_test_acc"      : clf_test_acc,
        "initial_dfa_states": initial_state,
        "initial_train_acc" : initial_train,  # Training accuracy from beam search on perturbation samples
        "initial_validation_acc": 1.0,  # Validation accuracy on validation_data
        "shared_build_time" : 0.0,  # Extracted from beam search
    }

    results['beam'] = dict(
        train_acc   = final_train,
        validation_acc = final_validation,
        final_state = final_state,
        time        = beam_time,
        success     = beam_success,
    )

    # ══════════════════════════════════════════════════════════════════════════════════════════════
    # Method 2 – Simulated Annealing
    # ══════════════════════════════════════════════════════════════════════════════════════════════
    print("\n  ─── Simulated Annealing ───────────────────────────────")
    t0 = time.time()
    sa_res  = sa_dfa_search(
        sampler_fn         = sampler_fn,
        data_type          = "Tabular",
        shared_init        = shared,
        accuracy_threshold = cfg['accuracy_threshold'],
        state_threshold    = cfg['state_threshold'],
        select_by          = cfg['select_by'],
        init_num_samples   = cfg['init_num_samples'],
        batch_size         = cfg['batch_size'],
        output_dir         = os.path.join(out_dir, "sa"),
        beam_size          = SA_BEAM_SIZE,  # Single neighbor per step
        steps              = SA_STEPS,
        T_max              = 10.0,
        T_min              = 0.001,
        max_evaluations    = budget_used,  # Budget limit: same as Beam Search
        instance           = test_instance,
    )
    sa_time   = time.time() - t0
    sa_dfa    = sa_res.get('automata')
    # training_accuracy: accuracy on perturbation samples from search_baselines
    sa_train  = _extract_accuracy(sa_res.get('training_accuracy', 0))
    # validation_accuracy: accuracy on validation samples from beam search (shared.validation_data/validation_labels)
    sa_validation = eval_validation_acc(sa_dfa, shared.validation_data, shared.validation_labels, shared.learner)
    sa_states = int(sa_res.get('size', 0) or
                    (len(sa_dfa.states) if sa_dfa else 0))
    results['sa'] = dict(
        train_acc   = sa_train,
        validation_acc = sa_validation,
        states      = sa_states,
        time        = sa_time,
        success     = sa_res.get('success', False),
    )
    print(f"    train={sa_train:.4f}  validation={sa_validation:.4f}  "
          f"states={sa_states}  time={sa_time:.1f}s"
          f"  {'✓' if sa_res.get('success') else '✗'}")

    # ══════════════════════════════════════════════════════════════════
    # Method 3 – Genetic Algorithm
    # ══════════════════════════════════════════════════════════════════
    print("\n  ─── Genetic Algorithm ────────────────────────────────")
    t0 = time.time()
    ga_res  = ga_dfa_search(
        sampler_fn         = sampler_fn,
        data_type          = "Tabular",
        shared_init        = shared,
        accuracy_threshold = cfg['accuracy_threshold'],
        state_threshold    = cfg['state_threshold'],
        select_by          = cfg['select_by'],
        init_num_samples   = cfg['init_num_samples'],
        batch_size         = cfg['batch_size'],
        output_dir         = os.path.join(out_dir, "ga"),
        beam_size          = GA_BEAM_SIZE,  # 5 candidates per mutation for diversity
        tournament_size    = 2,  # Tournament selection: pick best from 2 random
        max_evaluations    = budget_used,  # Budget limit: same as Beam Search
        instance           = test_instance,
    )
    ga_time   = time.time() - t0
    ga_dfa    = ga_res.get('automata')
    # training_accuracy: accuracy on perturbation samples from search_baselines
    ga_train  = _extract_accuracy(ga_res.get('training_accuracy', 0))
    # validation_accuracy: accuracy on validation samples from beam search (shared.validation_data/validation_labels)
    ga_validation = eval_validation_acc(ga_dfa, shared.validation_data, shared.validation_labels, shared.learner)
    ga_states = int(ga_res.get('size', 0) or
                    (len(ga_dfa.states) if ga_dfa else 0))
    results['ga'] = dict(
        train_acc   = ga_train,
        validation_acc = ga_validation,
        states      = ga_states,
        time        = ga_time,
        success     = ga_res.get('success', False),
    )
    print(f"    train={ga_train:.4f}  validation={ga_validation:.4f}  "
          f"states={ga_states}  time={ga_time:.1f}s"
          f"  {'✓' if ga_res.get('success') else '✗'}")

    # ══════════════════════════════════════════════════════════════════
    # Method 4 – Particle Swarm Optimisation
    # ══════════════════════════════════════════════════════════════════
    print("\n  ─── Particle Swarm Optimisation ─────────────────────")
    t0 = time.time()
    pso_res  = pso_dfa_search(
        sampler_fn         = sampler_fn,
        data_type          = "Tabular",
        shared_init        = shared,
        accuracy_threshold = cfg['accuracy_threshold'],
        state_threshold    = cfg['state_threshold'],
        select_by          = cfg['select_by'],
        init_num_samples   = cfg['init_num_samples'],
        batch_size         = cfg['batch_size'],
        output_dir         = os.path.join(out_dir, "pso"),
        n_particles        = PSO_PARTICLES,
        beam_size          = PSO_BEAM_SIZE,  # 1 candidate per particle (swarm provides diversity)
        max_evaluations    = budget_used,  # Budget limit: same as Beam Search
        instance           = test_instance,
    )
    pso_time   = time.time() - t0
    pso_dfa    = pso_res.get('automata')
    pso_train  = _extract_accuracy(pso_res.get('training_accuracy', 0))
    pso_validation = eval_validation_acc(pso_dfa, shared.validation_data, shared.validation_labels, shared.learner)
    pso_states = int(pso_res.get('size', 0) or (len(pso_dfa.states) if pso_dfa else 0))
    results['pso'] = dict(
        train_acc   = pso_train,
        validation_acc = pso_validation,
        states      = pso_states,
        time        = pso_time,
        success     = pso_res.get('success', False),
    )
    print(f"    train={pso_train:.4f}  validation={pso_validation:.4f}  "
          f"states={pso_states}  time={pso_time:.1f}s"
          f"  {'✓' if pso_res.get('success') else '✗'}")

    return results


# ======================================================================
# Summary table + CSV export
# ======================================================================
METHODS = ['beam', 'sa', 'ga', 'pso']
METHOD_LABELS = {'beam': 'BeamSearch', 'sa': 'SA', 'ga': 'GA', 'pso': 'PSO'}

def print_summary(all_results: dict, accuracy_threshold: float = 0.9) -> None:
    print("\n\n" + "=" * 100)
    print(f"  EXPERIMENT SUMMARY (threshold={accuracy_threshold:.1f}) - WITH INITIAL & FINAL METRICS")
    print("=" * 100)

    # Per-language results
    for lang_code, res in sorted(all_results.items()):
        if res is None:
            print(f"\n  {lang_code}: [NO DATA]")
            continue

        print(f"\n  {lang_code}  "
              f"(clf_train={res['clf_train_acc']:.4f}  "
              f"clf_test={res['clf_test_acc']:.4f}  "
              f"initial_states={res['initial_dfa_states']})")
        
        # Header for initial/final accuracies
        init_train = res.get('initial_train_acc', 0)
        init_val   = res.get('initial_validation_acc', 0)
        
        print(f"\n  Initial (RPNI):  train={init_train:.4f}  validation={init_val:.4f}")
        print("  " + "─" * 96)
        print(f"  | {'Method':12s} | {'Train (Init→Final)':20s} | {'Validation (Init→Final)':24s} | "
              f"{'States':10s} | {'Time(s)':10s} |")
        print("  " + "─" * 96)
        
        for m in METHODS:
            r = res.get(m, {})
            ok = '✓' if r.get('success') else '✗'
            train_init = res.get('initial_train_acc', 0)
            train_final = r.get('train_acc', 0)
            val_init = res.get('initial_validation_acc', 0)
            val_final = r.get('validation_acc', 0)
            
            print(f"  | {METHOD_LABELS[m]:12s} | "
                  f"{train_init:.4f}→{train_final:.4f} {ok:1s}       | "
                  f"{val_init:.4f}→{val_final:.4f}       | "
                  f"{r.get('states', 0):10d} | "
                  f"{r.get('time', 0):10.1f} |")
        print("  " + "─" * 96)

    # Paper suitability analysis
    print("\n\n  PAPER SUITABILITY ANALYSIS")
    print("  " + "─" * 66)
    print(f"  {'Lang':6s}  {'BeamSearch':>10s}  {'SA':>8s}  "
          f"{'GA':>8s}  {'PSO':>8s}  {'Δ_validation':>10s}  {'Comment':s}")
    print("  " + "─" * 66)

    for lang_code, res in sorted(all_results.items()):
        if res is None:
            print(f"  {lang_code:6s}  [NO DATA]")
            continue
        beam_h = res.get('beam', {}).get('validation_acc', 0)
        sa_h   = res.get('sa',   {}).get('validation_acc', 0)
        ga_h   = res.get('ga',   {}).get('validation_acc', 0)
        pso_h  = res.get('pso',  {}).get('validation_acc', 0)
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
                'language'              : lang,
                'method'                : m,
                'initial_train_acc'     : res.get('initial_train_acc', ''),
                'final_train_acc'       : r.get('train_acc', ''),
                'initial_validation_acc': res.get('initial_validation_acc', ''),
                'final_validation_acc'  : r.get('validation_acc', ''),
                'states'                : r.get('states', ''),
                'time_s'                : r.get('time', ''),
                'success'               : int(r.get('success', False)),
                'clf_train_acc'         : res.get('clf_train_acc', ''),
                'clf_test_acc'          : res.get('clf_test_acc', ''),
                'init_states'           : res.get('initial_dfa_states', ''),
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
    parser = argparse.ArgumentParser(
        description="Run baseline experiment with specified accuracy_threshold"
    )
    parser.add_argument(
        "--accuracy_threshold",
        type=float,
        default=0.9,
        help="Accuracy threshold for all languages (default: 0.9)"
    )
    args = parser.parse_args()

    accuracy_threshold = args.accuracy_threshold
    LANGUAGES = get_languages_config(accuracy_threshold)

    OUTPUT_ROOT = os.path.join(
        PROJECT_ROOT, "test_result", 
        f"baseline_experiment_threshold_{accuracy_threshold}"
    )
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    log_path = os.path.join(OUTPUT_ROOT, "experiment_log_rerun.txt")

    print(f"\n{'='*70}")
    print(f"  Running experiment with accuracy_threshold={accuracy_threshold}")
    print(f"  Output directory: {OUTPUT_ROOT}")
    print(f"{'='*70}\n")

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
                    print(f"[TRACEBACK]")
                    traceback.print_exc()
                    all_results[lang_code] = None

            print_summary(all_results, accuracy_threshold)
            save_csv(all_results, os.path.join(OUTPUT_ROOT, "results_rerun.csv"))

        finally:
            sys.stdout = _orig_stdout

    print(f"\nFull log → {log_path}")
