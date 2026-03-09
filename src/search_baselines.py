"""
DFA Search Baselines: SA, GA, PSO

Each baseline follows the same pipeline as the main anchor-based system:
  1. Same perturbation samples + RPNI initial DFA
  2. Same propose_automata() for candidate generation (DELETE / MERGE / DELTA)
  3. Same training-accuracy scoring
  4. Same final-selection logic as anchor_base.anchor_beam()

Libraries:
  from deap import base, creator, tools, algorithms
  from simanneal import Annealer
  from pyswarms.single.global_best import GlobalBestPSO
"""

from __future__ import annotations

import gc
import random
import itertools
import os
from collections import defaultdict
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from deap import base, creator, tools, algorithms
from simanneal import Annealer
from pyswarms.single.global_best import GlobalBestPSO
from modified_modules.alibi.utils.distributions import kl_bernoulli

# -----------------------------------------------------------------------
# Module-level reference to the learner instance (mirrors anchor_base.py)
# -----------------------------------------------------------------------
_AUTO_INSTANCE = None  # set by _common_init() or build_shared_init()


# ======================================================================
# Shared initialisation object
# ======================================================================

class SharedInit(NamedTuple):
    """
    Carries the artefacts produced by a single RPNI run so that every
    search method (beam search, SA, GA, PSO) can start from the exact
    same initial DFA and the same training data.

    Fields
    ------
    initial_dfa : object       – the RPNI-generated DFA (call .copy() before use)
    data        : list         – training samples drawn for RPNI
    labels      : np.ndarray  – corresponding labels
    learner     : DFALearner  – the learner instance (holds alphabet_map etc.)
    """
    initial_dfa: object
    data: list
    labels: np.ndarray
    learner: object


def build_shared_init(sampler_fn: Callable,
                      data_type: str,
                      init_num_samples: int = 1000,
                      batch_size: int = 100,
                      output_dir: str = "test_result/shared") -> SharedInit:
    """
    Build the shared initial DFA **once** and return a :class:`SharedInit`
    that can be passed to :func:`sa_dfa_search`, :func:`ga_dfa_search`,
    :func:`pso_dfa_search`, and (via ``prebuilt_init=``) to
    ``AnchorBaseBeam.anchor_beam()``.

    Parameters
    ----------
    sampler_fn       : the same sampler already attached to AnchorBaseBeam
    data_type        : "Tabular" / "Text" / "Image"
    init_num_samples : number of samples to draw for RPNI (same as anchor_beam)
    batch_size       : used only for state-dict prealloc sizing
    output_dir       : directory for any graphviz output

    Returns
    -------
    SharedInit
    """
    global _AUTO_INSTANCE

    from learner.dfa_learner import DFALearner
    _AUTO_INSTANCE = DFALearner()

    os.makedirs(output_dir, exist_ok=True)

    data, labels = sampler_fn(num_samples=init_num_samples, compute_labels=True)
    labels = np.array(labels)

    positive_samples = [x for x, y in zip(data, labels) if y == 1]
    negative_samples = [x for x, y in zip(data, labels) if y == 0]
    initial_dfa = _AUTO_INSTANCE.create_init_automata(data_type, positive_samples, negative_samples)

    print(f"[SharedInit] Initial DFA: {len(initial_dfa.states)} states")
    print(f"[SharedInit] Training data: {len(data)} samples "
          f"({int(np.sum(labels == 1))} pos / {int(np.sum(labels == 0))} neg)")

    return SharedInit(
        initial_dfa=initial_dfa,
        data=list(data),
        labels=labels,
        learner=_AUTO_INSTANCE,
    )


# ======================================================================
# Shared helpers
# ======================================================================

def _compute_accuracy(dfa, data: list, labels: np.ndarray) -> float:
    """Training accuracy of *dfa* on the labelled data."""
    if len(data) == 0:
        return 0.0
    accepts = np.array([_AUTO_INSTANCE.check_path_accepted(dfa, p) for p in data])
    lbl = np.asarray(labels)
    correct = int(np.sum((lbl == 1) & accepts) + np.sum((lbl == 0) & ~accepts))
    return correct / len(lbl)


def _rank_candidates(candidates: list, data: list, labels: np.ndarray) -> list:
    """
    Rank candidates by accuracy (descending).
    Since propose_automata naturally reduces states, we only sort by accuracy.
    
    Returns sorted list (best accuracy first): [{'dfa': ..., 'accuracy': ..., 'states': ...}, ...]
    """
    scored = []
    for dfa in candidates:
        acc = _compute_accuracy(dfa, data, labels)
        scored.append({
            'dfa': dfa,
            'accuracy': acc,
            'states': len(dfa.states),
        })
    
    # Sort by accuracy descending (best first)
    scored.sort(key=lambda x: x['accuracy'], reverse=True)
    return scored


def _dlow_bernoulli(p: np.ndarray, level: np.ndarray, n_iter: int = 10) -> np.ndarray:
    """
    Compute lower accuracy bound using KL divergence (from anchor_base.py).
    
    Parameters
    ----------
    p       : accuracy values
    level   : beta / nb of samples
    n_iter  : number of bisection iterations
    
    Returns
    -------
    Lower bound on accuracy
    """
    um = p.copy()
    lm = np.maximum(np.maximum(p - np.sqrt(level / 2.), 0.0), 0.0)
    
    for j in range(1, n_iter):
        qm = (um + lm) / 2.
        kl_gt_idx = kl_bernoulli(p, qm) > level
        kl_lt_idx = np.logical_not(kl_gt_idx)
        lm[kl_gt_idx] = qm[kl_gt_idx]
        um[kl_lt_idx] = qm[kl_lt_idx]
    
    return lm


def _dup_bernoulli(p: np.ndarray, level: np.ndarray, n_iter: int = 10) -> np.ndarray:
    """
    Compute upper accuracy bound using KL divergence (from anchor_base.py).
    
    Parameters
    ----------
    p       : accuracy values
    level   : beta / nb of samples
    n_iter  : number of bisection iterations
    
    Returns
    -------
    Upper bound on accuracy
    """
    lm = p.copy()
    um = np.minimum(np.minimum(p + np.sqrt(level / 2.), 1.0), 1.0)
    
    for j in range(1, n_iter):
        qm = (um + lm) / 2.
        kl_gt_idx = kl_bernoulli(p, qm) > level
        kl_lt_idx = np.logical_not(kl_gt_idx)
        um[kl_gt_idx] = qm[kl_gt_idx]
        lm[kl_lt_idx] = qm[kl_lt_idx]
    
    return um


def _kl_lucb_adaptive_sampling(sampler_fn: Callable,
                               current_beam: list,
                               data: list,
                               labels: np.ndarray,
                               state: dict,
                               accuracy_threshold: float = 0.95,
                               delta: float = 0.05,
                               epsilon: float = 0.1,
                               batch_size: int = 100,
                               max_samples: int = 5000) -> Tuple[list, np.ndarray]:
    """
    KL-LUCB adaptive sampling: add samples if confidence bound doesn't meet threshold.
    
    This matches the strategy in anchor_base.anchor_beam():
    - Compute accuracy and lower bound on current best DFA
    - If lower bound < (target_accuracy - epsilon), draw more samples
    - Stop when: lower bound meets target, or max_samples reached
    
    Parameters
    ----------
    sampler_fn           : function to draw samples
    current_beam         : list of best DFAs from current round
    data                 : current training data (list)
    labels               : current labels (ndarray)
    state                : state dict for propose_automata
    accuracy_threshold   : target accuracy for final automaton
    delta                : confidence parameter (default 0.05)
    epsilon              : tolerance margin (default 0.1)
    batch_size           : samples to draw per batch
    max_samples          : maximum total samples to collect
    
    Returns
    -------
    (updated_data, updated_labels) tuple
    """
    if not current_beam or len(current_beam) == 0:
        return data, np.array(labels)
    
    # Get best DFA from current beam
    best_dfa = current_beam[0]
    
    # Compute current accuracy
    current_acc = _compute_accuracy(best_dfa, data, labels)
    beta = np.log(1.0 / delta)
    total_samples = len(labels)
    
    # Compute lower bound
    mean = np.array([current_acc])
    lb = _dlow_bernoulli(mean, np.array([beta / total_samples]))[0]
    
    # Check if we need more samples
    if lb >= accuracy_threshold - epsilon or total_samples >= max_samples:
        return data, np.array(labels)
    
    print(f"  [KL-LUCB] Current accuracy: {current_acc:.4f}, lower bound: {lb:.4f}, target: {accuracy_threshold - epsilon:.4f}")
    print(f"  [KL-LUCB] Sampling more data (current: {total_samples}, max: {max_samples})")
    
    # Draw more samples in batches until convergence
    new_data = list(data)
    new_labels = np.array(labels)
    
    while total_samples < max_samples:
        try:
            # Draw batch of samples
            batch_samples, batch_labels = sampler_fn(num_samples=batch_size, compute_labels=True)
            batch_labels = np.array(batch_labels)
            
            # Append to data
            new_data.extend(batch_samples)
            new_labels = np.concatenate([new_labels, batch_labels])
            total_samples = len(new_labels)
            
            # Recompute accuracy with new data
            current_acc = _compute_accuracy(best_dfa, new_data, new_labels)
            mean = np.array([current_acc])
            lb = _dlow_bernoulli(mean, np.array([beta / total_samples]))[0]
            
            print(f"    [KL-LUCB] After {total_samples} samples: acc={current_acc:.4f}, lb={lb:.4f}")
            
            # Check convergence
            if lb >= accuracy_threshold - epsilon:
                print(f"  [KL-LUCB] Convergence reached at {total_samples} samples")
                break
        except Exception as e:
            print(f"  [KL-LUCB] Sampling error: {e}, stopping early")
            break
    
    return new_data, new_labels


def _is_valid_dfa(dfa) -> bool:
    """True if *dfa* has >= 2 states and at least one accepting state."""
    if not hasattr(dfa, 'states'):
        return False
    states = list(dfa.states)
    return len(states) >= 2 and any(s.is_accepting for s in states)


def _add_to_history(all_history: list, seen_ids: set, dfa, accuracy: float) -> None:
    """Append a valid, not-yet-seen DFA to *all_history*."""
    dfa_id = id(dfa)
    if dfa_id in seen_ids:
        return
    if not _is_valid_dfa(dfa):
        return
    seen_ids.add(dfa_id)
    all_history.append({
        "automata": dfa,
        "accuracy": accuracy,
        "states": len(dfa.states),
    })


def _get_candidates(dfa, state: dict, iteration: int, output_dir: str,
                    beam_size: int = 10) -> list:
    """Wrapper around propose_automata for a single-DFA neighbourhood."""
    return _AUTO_INSTANCE.propose_automata(
        [dfa], state, iteration, [dfa], output_dir, beam_size
    )


def _common_init(sampler_fn: Callable,
                 data_type: str,
                 init_num_samples: int,
                 batch_size: int,
                 output_dir: str,
                 shared_init: Optional[SharedInit] = None) -> Tuple[object, list, np.ndarray, dict]:
    """
    Identical initialisation to :py:meth:`anchor_base.AnchorBaseBeam.anchor_beam`:
      1. Draw *init_num_samples* from *sampler_fn*  (skipped when *shared_init* is given)
      2. Build initial DFA via RPNI               (skipped when *shared_init* is given)
      3. Initialise the state dict (same schema as AnchorBaseBeam._init_state)
      4. Call propose_automata(iteration=0) to register initial DFA metrics

    Parameters
    ----------
    shared_init : SharedInit, optional
        When provided the RPNI step is skipped; the learner, initial DFA, data,
        and labels are taken directly from *shared_init* (a fresh .copy() of the
        DFA is used so each method has its own independent copy).

    Returns
    -------
    initial_dfa, data (list), labels (np.ndarray), state (dict)
    """
    global _AUTO_INSTANCE

    os.makedirs(output_dir, exist_ok=True)

    if shared_init is not None:
        # ---- Use pre-built shared initialisation ----
        _AUTO_INSTANCE = shared_init.learner
        initial_dfa    = shared_init.initial_dfa.copy()   # independent copy
        data           = list(shared_init.data)
        labels         = shared_init.labels.copy()
        print(f"[Init] Using shared initial DFA ({len(initial_dfa.states)} states, "
              f"{len(data)} training samples)")
    else:
        # ---- Build from scratch (original behaviour) ----
        from learner.dfa_learner import DFALearner
        _AUTO_INSTANCE = DFALearner()

        # Draw initial training samples (same size as anchor_beam)
        data, labels = sampler_fn(num_samples=init_num_samples, compute_labels=True)
        labels = np.array(labels)

        # RPNI: build the initial DFA
        positive_samples = [x for x, y in zip(data, labels) if y == 1]
        negative_samples = [x for x, y in zip(data, labels) if y == 0]
        initial_dfa = _AUTO_INSTANCE.create_init_automata(data_type, positive_samples, negative_samples)

    # Build state dict (mirrors AnchorBaseBeam._init_state)
    prealloc_size = batch_size * 10_000
    state: dict = {
        't_coverage':       defaultdict(lambda: 0.),
        't_coverage_idx':   defaultdict(set),
        't_covered_true':   defaultdict(None),
        't_covered_false':  defaultdict(None),
        't_idx':            defaultdict(set),
        't_nsamples':       defaultdict(lambda: 0.),
        't_accepted':       defaultdict(lambda: 0.),
        't_order':          defaultdict(list),
        't_positives':      defaultdict(lambda: 0.),
        't_negatives':      defaultdict(lambda: 0.),
        'prealloc_size':    prealloc_size,
        'data':             list(data),
        'labels':           labels,
        'current_idx':      len(data),
    }
    state['t_order'][()] = ()

    # Register initial DFA metrics (propose_automata at iteration=0 is a no-op for candidates)
    _AUTO_INSTANCE.propose_automata([initial_dfa], state, 0, [], output_dir, beam_size=1)

    return initial_dfa, list(data), labels, state


def _select_final(all_history: list,
                  select_by: str,
                  accuracy_threshold: float,
                  state_threshold: int,
                  automaton_type: str,
                  initial_dfa,
                  output_dir: str,
                  get_metadata_fn: Callable,
                  test_data: Optional[list] = None,
                  test_labels: Optional[np.ndarray] = None) -> dict:
    """
    Exact same final-selection logic as anchor_base.anchor_beam().

    Modes
    -----
    "accuracy"  – among candidates with accuracy >= accuracy_threshold, pick fewest states.
    "state"     – among candidates with states <= state_threshold, pick highest accuracy.
    Fallback    – best-effort (highest accuracy) when no candidate meets the threshold.
    
    Parameters
    ----------
    test_data   : optional holdout test data for computing testing_accuracy
    test_labels : optional holdout test labels
    """
    def _cleanup(best_record: dict, success: bool, label: str = "") -> dict:
        automata = best_record["automata"]
        if automaton_type.upper() == "DFA":
            from learner.dfa_learner import remove_unreachable_states
            remove_unreachable_states(automata)
            _AUTO_INSTANCE.automaton_to_graphviz(
                automata, filename="final_automata", output_dir=output_dir
            )
        return get_metadata_fn(automata, success=success, test_data=test_data, test_labels=test_labels)

    print(f"\n[SELECT] select_by='{select_by}', "
          f"accuracy_threshold={accuracy_threshold}, state_threshold={state_threshold}")
    print(f"  Total candidates in history: {len(all_history)}")

    if not all_history:
        print("  [SELECT] No candidates – returning initial DFA.")
        return get_metadata_fn(initial_dfa, success=False)

    if select_by == "accuracy":
        qualified = [r for r in all_history if r["accuracy"] >= accuracy_threshold]
        if qualified:
            best = min(qualified, key=lambda x: x["states"])
            print(f"  [accuracy mode] {len(qualified)} candidate(s) meet accuracy >= {accuracy_threshold}.")
            print(f"  Selected: states={best['states']}, accuracy={best['accuracy']:.4f}")
            return _cleanup(best, success=True, label="qualified(accuracy)")
        else:
            best = max(all_history, key=lambda x: x["accuracy"])
            print(f"  [accuracy mode] No candidate meets accuracy threshold.")
            print(f"  Best-effort: states={best['states']}, accuracy={best['accuracy']:.4f}")
            return _cleanup(best, success=False, label="best-effort(accuracy)")

    elif select_by == "state":
        under = [r for r in all_history if r["states"] <= state_threshold]
        if under:
            best = max(under, key=lambda x: x["accuracy"])
            print(f"  [state mode] {len(under)} candidate(s) have states <= {state_threshold}.")
            print(f"  Selected: states={best['states']}, accuracy={best['accuracy']:.4f}")
            return _cleanup(best, success=True, label="qualified(state)")
        else:
            best = max(all_history, key=lambda x: x["accuracy"])
            print(f"  [state mode] No candidate has states <= {state_threshold}.")
            print(f"  Best-effort: states={best['states']}, accuracy={best['accuracy']:.4f}")
            return _cleanup(best, success=False, label="best-effort(state)")

    else:
        raise ValueError(f"Unknown select_by='{select_by}'. Use 'accuracy' or 'state'.")


def _simple_metadata(dfa, data: list, labels: np.ndarray, success: bool,
                     test_data: Optional[list] = None, test_labels: Optional[np.ndarray] = None) -> dict:
    """
    Lightweight metadata dict (mirrors the fields returned by anchor_beam).
    
    Parameters
    ----------
    dfa             : DFA to evaluate
    data            : training data (for training_accuracy)
    labels          : training labels
    success         : whether selection was successful
    test_data       : optional holdout/test data (for testing_accuracy)
    test_labels     : optional holdout/test labels
    """
    train_acc = _compute_accuracy(dfa, data, labels)
    
    # Compute testing accuracy if holdout data is provided
    if test_data is not None and test_labels is not None:
        test_acc = _compute_accuracy(dfa, test_data, test_labels)
    else:
        test_acc = train_acc  # fallback to training accuracy if no test set
    
    return {
        'automata':          dfa,
        'training_accuracy': train_acc,
        'testing_accuracy':  test_acc,
        'size':              len(dfa.states) if hasattr(dfa, 'states') else 0,
        'coverage':          [],
        'examples':          [],
        'success':           success,
        'false_accept':      [],
        'true_reject':       [],
    }


# ======================================================================
# Simulated Annealing with simanneal library
# ======================================================================

class DFAAnnealer(Annealer):
    """
    Simulated Annealing solver for DFA optimization.
    
    Uses temperature-based energy acceptance to escape local optima.
    Energy = 1 - accuracy (minimize this)
    """
    
    def __init__(self, initial_dfa, data: list, labels: np.ndarray, 
                 state: dict, learner, iteration: int, output_dir: str,
                 beam_size: int, all_history: list, seen_ids: set,
                 T_max: float = 10.0, T_min: float = 0.001, n_steps: int = 100):
        self.current_dfa = initial_dfa.copy()
        self.best_dfa = initial_dfa.copy()
        self.best_energy = 1.0 - _compute_accuracy(initial_dfa, data, labels)
        
        self.data = data
        self.labels = labels
        self.state = state
        self.learner = learner
        self.round_idx = iteration  # Round number (for tracking)
        self.local_iteration = 0    # Local iteration counter (incremented in each move())
        self.output_dir = output_dir
        self.beam_size = beam_size
        self.all_history = all_history
        self.seen_ids = seen_ids
        
        # Annealer parameters
        self.Tmax = T_max          # Initial temperature
        self.Tmin = T_min          # Final temperature
        self.steps = n_steps       # Number of iterations
        self.updates = max(1, n_steps // 10)   # Progress updates
        
        # Required by simanneal.Annealer: pass initial_state to parent __init__
        # (actual DFA state stored in self.current_dfa, initial_state is just a placeholder)
        super().__init__(initial_state=0)
    
    def move(self):
        """Generate a neighboring DFA via propose_automata."""
        try:
            # Use alternating local_iteration for DELETE/MERGE (even) and DELTA (odd) operations
            # Round * 2 ensures even iterations at the start of each round (enables DELETE/MERGE)
            effective_iteration = (self.round_idx - 1) * 2 + self.local_iteration
            self.local_iteration += 1
            
            candidates = _AUTO_INSTANCE.propose_automata(
                [self.current_dfa], self.state, effective_iteration, 
                [self.current_dfa], self.output_dir, self.beam_size
            )
            if candidates:
                self.current_dfa = random.choice(candidates)
        except Exception as e:
            # If generation fails, keep current
            pass
    
    def energy(self):
        """Compute energy = 1 - accuracy (lower is better)."""
        acc = _compute_accuracy(self.current_dfa, self.data, self.labels)
        energy = 1.0 - acc
        
        # Track best found
        if energy < self.best_energy:
            self.best_energy = energy
            self.best_dfa = self.current_dfa.copy()
            _add_to_history(self.all_history, self.seen_ids, self.best_dfa, 1.0 - energy)
        
        # Also add to history for exploration tracking
        _add_to_history(self.all_history, self.seen_ids, self.current_dfa, acc)
        
        return energy


def sa_dfa_search(sampler_fn: Callable,
                  data_type: str,
                  *,
                  accuracy_threshold: float = 1.0,
                  state_threshold: int = 5,
                  select_by: str = "accuracy",
                  init_num_samples: int = 1000,
                  batch_size: int = 100,
                  output_dir: str = "test_result/sa",
                  beam_size: int = 3,
                  n_rounds: int = 10,
                  T_max: float = 10.0,
                  T_min: float = 0.001,
                  n_steps: int = 100,
                  shared_init: Optional[SharedInit] = None,
                  test_data: Optional[list] = None,
                  test_labels: Optional[np.ndarray] = None,
                  **kwargs) -> dict:
    """
    Simulated Annealing for DFA search using simanneal library.
    
    Each round uses SA to explore neighborhood of current best DFA.
    Temperature cooling schedule: Tmax → Tmin over n_steps iterations.
    Probabilistic acceptance: exp(-ΔE/T) determines if worse moves are accepted.
    
    Parameters
    ----------
    beam_size       : int       – candidates per SA round (for propose_automata)
    n_rounds        : int       – number of SA rounds (each gets fresh temperature schedule)
    T_max           : float     – initial temperature
    T_min           : float     – final temperature
    n_steps         : int       – number of SA iterations per round
    shared_init     : SharedInit, optional – pre-built DFA and data
    
    Returns
    -------
    dict with same keys as anchor_beam()
    """
    print("=" * 70)
    print("[SA-SIMANNEAL] Initialising …")
    initial_dfa, data, labels, state = _common_init(
        sampler_fn, data_type, init_num_samples, batch_size, output_dir, shared_init
    )

    all_history: List[dict] = []
    seen_ids: set = set()
    min_states = 2

    # Seed history and beam with initial DFA
    init_acc = _compute_accuracy(initial_dfa, data, labels)
    _add_to_history(all_history, seen_ids, initial_dfa, init_acc)
    current_beam = [initial_dfa]
    
    round_idx = 0
    while round_idx < n_rounds:
        round_idx += 1
        print(f"\n[SA-SIMANNEAL] Round {round_idx}/{n_rounds}")
        
        # Run SA from each DFA in current beam
        best_from_round = None
        best_acc_round = -1
        
        for beam_idx, seed_dfa in enumerate(current_beam):
            print(f"  [SA Particle {beam_idx}] Starting from {len(seed_dfa.states)}-state DFA")
            
            # Create annealer instance
            annealer = DFAAnnealer(
                seed_dfa, data, labels, state, _AUTO_INSTANCE, round_idx,
                output_dir, beam_size, all_history, seen_ids,
                T_max=T_max, T_min=T_min, n_steps=n_steps
            )
            
            # Run SA (returns placeholder state and energy, use annealer.best_dfa instead)
            _, best_energy = annealer.anneal()
            best_dfa = annealer.best_dfa
            best_acc_sa = 1.0 - annealer.best_energy
            
            print(f"    SA converged: {len(best_dfa.states)} states, accuracy={best_acc_sa:.4f}")
            
            # Track best from this round
            if best_acc_sa > best_acc_round:
                best_acc_round = best_acc_sa
                best_from_round = best_dfa
        
        # [KL-LUCB] Adaptive sampling
        enable_kl_lucb = kwargs.get('enable_kl_lucb', True)
        if enable_kl_lucb and round_idx < n_rounds:
            # Get top candidates from history for this round
            round_candidates = [h for h in all_history[-100:]]  # Recent candidates
            if round_candidates:
                best_recent = max(round_candidates, key=lambda x: x['accuracy'])
                data, labels = _kl_lucb_adaptive_sampling(
                    sampler_fn, [best_recent['automata']], data, labels, state,
                    accuracy_threshold=accuracy_threshold,
                    delta=kwargs.get('delta', 0.05),
                    epsilon=kwargs.get('epsilon', 0.1),
                    batch_size=batch_size,
                    max_samples=kwargs.get('max_samples', 5000)
                )
                print(f"  [SA] Training data size after KL-LUCB: {len(data)} samples")
        
        # Check stopping condition
        if all_history:
            best_states = min(h['states'] for h in all_history)
            if best_states <= min_states:
                print(f"  Minimum states {min_states} reached. Stopping.")
                break
        
        # Update beam: keep top accuracy DFAs from history
        if all_history:
            sorted_hist = sorted(all_history, key=lambda x: x['accuracy'], reverse=True)
            current_beam = [h['automata'] for h in sorted_hist[:beam_size]]
            top_k = min(beam_size, len(sorted_hist))
            top_info = [(sorted_hist[i]['accuracy'], sorted_hist[i]['states']) for i in range(top_k)]
            print(f"  [Top {beam_size}] Best accuracies (states): {top_info}")
    
    print(f"\n[SA-SIMANNEAL] Completed {round_idx} rounds, {len(all_history)} candidates total")
    gc.collect()

    return _select_final(
        all_history, select_by, accuracy_threshold, state_threshold,
        "DFA", initial_dfa, output_dir,
        lambda dfa, success, test_data=test_data, test_labels=test_labels: _simple_metadata(dfa, data, labels, success, test_data, test_labels),
        test_data=test_data, test_labels=test_labels
    )


# ======================================================================
# Genetic Algorithm (Round-based, Comparable to Beam Search)
# ======================================================================

# DEAP requires global creator registration; guard against double-registration.
if not hasattr(creator, 'DFAFitness'):
    creator.create("DFAFitness", base.Fitness, weights=(1.0,))
if not hasattr(creator, 'DFAIndividual'):
    creator.create("DFAIndividual", list, fitness=creator.DFAFitness)


def ga_dfa_search(sampler_fn: Callable,
                  data_type: str,
                  *,
                  accuracy_threshold: float = 1.0,
                  state_threshold: int = 5,
                  select_by: str = "accuracy",
                  init_num_samples: int = 1000,
                  batch_size: int = 100,
                  output_dir: str = "test_result/ga",
                  beam_size: int = 3,
                  n_rounds: int = 10,
                  population_size: int = 10,
                  n_generations: int = 5,
                  mutation_prob: float = 0.8,
                  tournament_size: int = 2,
                  shared_init: Optional[SharedInit] = None,
                  test_data: Optional[list] = None,
                  test_labels: Optional[np.ndarray] = None,
                  **kwargs) -> dict:
    """
    Genetic Algorithm for DFA search using DEAP library.
    
    Each round:
    1. Create population from top candidates of previous round (elite preservation)
    2. Evolve for n_generations with mutation via propose_automata()
    3. Rank by accuracy
    4. Keep top beam_size for next round
    5. Stop when: no new candidates OR min_states=2
    
    Parameters
    ----------
    beam_size       : int       – beam width (elite size for next round)
    n_rounds        : int       – number of evolution rounds
    population_size : int       – population size per generation
    n_generations   : int       – generations per round
    mutation_prob   : float     – probability of mutation per individual
    tournament_size : int       – tournament selection size
    shared_init     : SharedInit, optional – pre-built DFA and data
    
    Returns
    -------
    dict with same keys as anchor_beam()
    """
    print("=" * 70)
    print("[GA-DEAP] Initialising …")
    initial_dfa, data, labels, state = _common_init(
        sampler_fn, data_type, init_num_samples, batch_size, output_dir, shared_init
    )

    all_history: List[dict] = []
    seen_ids: set = set()
    min_states = 2
    iteration_counter = [0]  # Mutable container for tracking iterations across mutations

    # Seed history with initial DFA
    init_acc = _compute_accuracy(initial_dfa, data, labels)
    _add_to_history(all_history, seen_ids, initial_dfa, init_acc)

    # -------- DEAP toolbox --------
    toolbox = base.Toolbox()

    def _create_individual(dfa_seed):
        """Create a DEAP Individual from a DFA seed."""
        return creator.DFAIndividual([dfa_seed.copy()])

    def _evaluate(individual):
        """Evaluate fitness = accuracy."""
        dfa = individual[0]
        acc = _compute_accuracy(dfa, data, labels)
        _add_to_history(all_history, seen_ids, dfa, acc)
        return (acc,)

    def _mutate(individual, round_idx):
        """Mutation operator: apply propose_automata to generate neighbor."""
        dfa = individual[0]
        try:
            # Use alternating iteration for DELETE/MERGE (even) and DELTA (odd) operations
            effective_iteration = (round_idx - 1) * 2 + iteration_counter[0]
            iteration_counter[0] += 1
            
            candidates = _get_candidates(dfa, state, effective_iteration, output_dir, beam_size)
            if candidates:
                individual[0] = random.choice(candidates)
        except Exception as e:
            pass  # Keep current if generation fails
        del individual.fitness.values
        return (individual,)

    toolbox.register("evaluate", _evaluate)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    current_beam = [initial_dfa]
    
    round_idx = 0
    while round_idx < n_rounds:
        round_idx += 1
        print(f"\n[GA-DEAP] Round {round_idx}/{n_rounds}, beam size: {len(current_beam)}")
        
        # ---- Initialize population from elite (current_beam) ----
        pop = []
        for dfa in current_beam:
            pop.append(_create_individual(dfa))
        
        # Fill population via mutation of elite
        while len(pop) < population_size:
            parent = toolbox.select(pop, 1)[0]
            child = _create_individual(parent[0].copy())
            pop.append(child)
        
        # Evaluate initial population
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(ind) for ind in invalid]
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit
        
        print(f"  [GA] Population initialized: {len(pop)} individuals")
        
        # ---- Multi-generation evolution ----
        for gen in range(n_generations):
            # Select next generation
            offspring = toolbox.select(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]
            
            # Apply mutation
            mutated_count = 0
            for ind in offspring:
                if random.random() < mutation_prob:
                    _mutate(ind, round_idx)
                    mutated_count += 1
            
            # Evaluate offspring with invalid fitness
            invalid_offspring = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [toolbox.evaluate(ind) for ind in invalid_offspring]
            for ind, fit in zip(invalid_offspring, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            pop[:] = offspring
            
            # Statistics
            fits = [ind.fitness.values[0] for ind in pop]
            gen_best = max(fits)
            print(f"    [Gen {gen + 1}] Mutated: {mutated_count}/{len(pop)}, "
                  f"Best accuracy: {gen_best:.4f}")
        
        # ---- KL-LUCB Adaptive sampling ----
        enable_kl_lucb = kwargs.get('enable_kl_lucb', True)
        if enable_kl_lucb and round_idx < n_rounds:
            # Get best DFA from population for sampling decision
            best_pop = max(pop, key=lambda x: x.fitness.values[0])
            data, labels = _kl_lucb_adaptive_sampling(
                sampler_fn, [best_pop[0]], data, labels, state,
                accuracy_threshold=accuracy_threshold,
                delta=kwargs.get('delta', 0.05),
                epsilon=kwargs.get('epsilon', 0.1),
                batch_size=batch_size,
                max_samples=kwargs.get('max_samples', 5000)
            )
            print(f"  [GA] Training data size after KL-LUCB: {len(data)} samples")
        
        # ---- Check stopping condition ----
        if all_history:
            best_states = min(h['states'] for h in all_history)
            if best_states <= min_states:
                print(f"  Minimum states {min_states} reached. Stopping.")
                break
        
        # ---- Elite selection: keep top beam_size for next round ----
        sorted_hist = sorted(all_history, key=lambda x: x['accuracy'], reverse=True)
        current_beam = [h['automata'] for h in sorted_hist[:beam_size]]
        top_k = min(beam_size, len(sorted_hist))
        top_info = [(sorted_hist[i]['accuracy'], sorted_hist[i]['states']) for i in range(top_k)]
        print(f"  [Elite] Best accuracies (states): {top_info}")
    
    print(f"\n[GA-DEAP] Completed {round_idx} rounds, {len(all_history)} candidates total")
    gc.collect()

    return _select_final(
        all_history, select_by, accuracy_threshold, state_threshold,
        "DFA", initial_dfa, output_dir,
        lambda dfa, success, test_data=test_data, test_labels=test_labels: _simple_metadata(dfa, data, labels, success, test_data, test_labels),
        test_data=test_data, test_labels=test_labels
    )


# ======================================================================
# Particle Swarm Optimisation (PSO) using Multi-Round Particle Search
# ======================================================================

class DFAParticle:
    """
    PSO particle for DFA optimization.
    
    Each particle maintains:
    - current DFA position
    - personal best DFA and accuracy
    - exploration history
    """
    
    def __init__(self, initial_dfa, data: list, labels: np.ndarray, 
                 state: dict, iteration: int, output_dir: str,
                 beam_size: int, all_history: list, seen_ids: set):
        self.current_dfa = initial_dfa.copy()
        self.best_dfa = initial_dfa.copy()
        self.best_accuracy = _compute_accuracy(initial_dfa, data, labels)
        
        self.data = data
        self.labels = labels
        self.state = state
        self.round_idx = iteration  # Round number (for tracking)
        self.local_iteration = 0    # Local iteration counter (incremented in each step)
        self.output_dir = output_dir
        self.beam_size = beam_size
        self.all_history = all_history
        self.seen_ids = seen_ids
    
    def step(self, global_best_dfa: object, n_steps: int = 3) -> None:
        """
        Execute PSO particle step: explore neighborhood and update position/best.
        
        Parameters
        ----------
        global_best_dfa : best DFA found by swarm
        n_steps         : refinement steps to explore
        """
        for step_idx in range(n_steps):
            try:
                # Use alternating local_iteration for DELETE/MERGE (even) and DELTA (odd) operations
                # Round * 2 ensures even iterations at the start of each round (enables DELETE/MERGE)
                effective_iteration = (self.round_idx - 1) * 2 + self.local_iteration
                self.local_iteration += 1
                
                # Generate candidates via propose_automata
                candidates = _AUTO_INSTANCE.propose_automata(
                    [self.current_dfa], self.state, effective_iteration,
                    [self.current_dfa], self.output_dir, self.beam_size
                )
                
                if not candidates:
                    break
                
                # PSO decision: blend personal best, global best, and exploration
                # Create candidate pool: current, neighbors, personal best, global best
                candidate_pool = [self.current_dfa] + candidates
                if global_best_dfa is not None:
                    candidate_pool.append(global_best_dfa)
                candidate_pool.append(self.best_dfa)
                
                # Rank candidates
                scored = _rank_candidates(candidate_pool, self.data, self.labels)
                
                # Probabilistic selection with weights (PSO: tendency towards better solutions)
                top_k = min(5, len(scored))
                weights = [1.0 / (i + 1) for i in range(top_k)]
                selected_idx = random.choices(list(range(top_k)), weights=weights, k=1)[0]
                self.current_dfa = scored[selected_idx]['dfa']
                
                # Update to history
                _add_to_history(self.all_history, self.seen_ids, self.current_dfa,
                              _compute_accuracy(self.current_dfa, self.data, self.labels))
                
            except Exception as e:
                # If step fails, keep current position
                pass
        
        # Update personal best
        current_acc = _compute_accuracy(self.current_dfa, self.data, self.labels)
        if current_acc > self.best_accuracy:
            self.best_accuracy = current_acc
            self.best_dfa = self.current_dfa.copy()
            _add_to_history(self.all_history, self.seen_ids, self.best_dfa, current_acc)


def pso_dfa_search(sampler_fn: Callable,
                   data_type: str,
                   *,
                   accuracy_threshold: float = 1.0,
                   state_threshold: int = 5,
                   select_by: str = "accuracy",
                   init_num_samples: int = 1000,
                   batch_size: int = 100,
                   output_dir: str = "test_result/pso",
                   beam_size: int = 3,
                   n_rounds: int = 10,
                   n_particles: int = 10,
                   n_steps_per_particle: int = 3,
                   shared_init: Optional[SharedInit] = None,
                   test_data: Optional[list] = None,
                   test_labels: Optional[np.ndarray] = None,
                   **kwargs) -> dict:
    """
    Particle Swarm Optimisation for DFA search using multi-particle exploration.
    
    Each round:
    1. Each particle explores neighborhood of its current position
    2. Particles track personal best and are influenced by global best
    3. Top candidates become seeds for next round
    4. Stop when: min_states=2 reached or max rounds
    
    Parameters
    ----------
    n_particles           : int  – number of particles in swarm
    n_steps_per_particle  : int  – refinement steps per particle per round
    n_rounds              : int  – max number of rounds
    beam_size             : int  – top candidates to seed next round
    shared_init           : SharedInit, optional – pre-built DFA and data
    
    Returns
    -------
    dict with same keys as anchor_beam()
    """
    print("=" * 70)
    print("[PSO-PYSWARMS] Initialising …")
    initial_dfa, data, labels, state = _common_init(
        sampler_fn, data_type, init_num_samples, batch_size, output_dir, shared_init
    )

    all_history: List[dict] = []
    seen_ids: set = set()
    min_states = 2

    # Seed history with initial DFA
    init_acc = _compute_accuracy(initial_dfa, data, labels)
    _add_to_history(all_history, seen_ids, initial_dfa, init_acc)
    
    # Initialize swarm particles
    particles = [
        DFAParticle(initial_dfa, data, labels, state, 0, output_dir,
                   beam_size, all_history, seen_ids)
        for _ in range(n_particles)
    ]
    
    # Global best tracking
    global_best_dfa = initial_dfa.copy()
    global_best_accuracy = init_acc
    
    round_idx = 0
    while round_idx < n_rounds:
        round_idx += 1
        print(f"\n[PSO-PYSWARMS] Round {round_idx}/{n_rounds}, swarm size: {n_particles}")
        
        # Execute PSO step for each particle
        for particle_idx, particle in enumerate(particles):
            particle.iteration = round_idx
            particle.state = state
            particle.data = data
            particle.labels = labels
            
            print(f"  [Particle {particle_idx}] Exploring from {len(particle.current_dfa.states)}-state DFA")
            
            # PSO step: explore and update position/best
            particle.step(global_best_dfa, n_steps=n_steps_per_particle)
            
            # Update global best if particle found better
            if particle.best_accuracy > global_best_accuracy:
                global_best_accuracy = particle.best_accuracy
                global_best_dfa = particle.best_dfa.copy()
                print(f"    Global best improved: accuracy={global_best_accuracy:.4f}, "
                      f"states={len(global_best_dfa.states)}")
        
        # [KL-LUCB] Adaptive sampling
        enable_kl_lucb = kwargs.get('enable_kl_lucb', True)
        if enable_kl_lucb and round_idx < n_rounds:
            # Use global best for sampling decision
            data, labels = _kl_lucb_adaptive_sampling(
                sampler_fn, [global_best_dfa], data, labels, state,
                accuracy_threshold=accuracy_threshold,
                delta=kwargs.get('delta', 0.05),
                epsilon=kwargs.get('epsilon', 0.1),
                batch_size=batch_size,
                max_samples=kwargs.get('max_samples', 5000)
            )
            print(f"  [PSO] Training data size after KL-LUCB: {len(data)} samples")
        
        # Check stopping condition
        if all_history:
            best_states = min(h['states'] for h in all_history)
            if best_states <= min_states:
                print(f"  Minimum states {min_states} reached. Stopping.")
                break
        
        # Reinitialize particles at end of round (optional: seed from top candidates)
        # This helps maintain diversity while leveraging best solutions
        if round_idx < n_rounds:
            sorted_hist = sorted(all_history, key=lambda x: x['accuracy'], reverse=True)
            top_candidates = [h['automata'] for h in sorted_hist[:beam_size]]
            
            # Keep particles at their current positions but give them new starting points occasionally
            for particle_idx, particle in enumerate(particles):
                # With probability, reinitialize particle from top candidates
                if random.random() < 0.3 and top_candidates:
                    seed_dfa = random.choice(top_candidates)
                    particle.current_dfa = seed_dfa.copy()
                    particle.best_dfa = seed_dfa.copy()
                    particle.best_accuracy = _compute_accuracy(seed_dfa, data, labels)
            
            top_k = min(beam_size, len(sorted_hist))
            top_info = [(sorted_hist[i]['accuracy'], sorted_hist[i]['states']) for i in range(top_k)]
            print(f"  [Top {beam_size}] Best accuracies (states): {top_info}")
    
    print(f"\n[PSO-PYSWARMS] Completed {round_idx} rounds, {len(all_history)} candidates total")
    gc.collect()

    return _select_final(
        all_history, select_by, accuracy_threshold, state_threshold,
        "DFA", initial_dfa, output_dir,
        lambda dfa, success, test_data=test_data, test_labels=test_labels: _simple_metadata(dfa, data, labels, success, test_data, test_labels),
        test_data=test_data, test_labels=test_labels
    )
