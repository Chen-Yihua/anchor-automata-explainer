"""
DFA Search Baselines: SA, GA, PSO

Each baseline follows the same pipeline as the main anchor-based system:
  1. Same perturbation samples + RPNI initial DFA
  2. Same propose_automata() for candidate generation (DELETE / MERGE / DELTA)
  3. Same training-accuracy scoring
  4. Same final-selection logic as anchor_base.anchor_beam()

Libraries:
  from deap import base, creator, tools
  from simanneal import Annealer
  from pyswarms.single.global_best import GlobalBestPSO
"""

from __future__ import annotations

import copy
import gc
import random
import os
from collections import defaultdict
from typing import Callable, List, NamedTuple, Tuple

import numpy as np
from deap import base, creator, tools
from simanneal import Annealer
from pyswarms.single.global_best import GlobalBestPSO

from automaton.utils import plot_beam_stats

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
    learner     : DFALearner  – the learner instance (holds alphabet_map etc.)
    validation_data : list       – validation samples (from beam search)
    validation_labels : np.ndarray – validation labels (from beam search)
    """
    initial_dfa: object
    learner: object
    validation_data: list
    validation_labels: np.ndarray

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


def _rank_candidates(candidates: list, training_data: list, training_labels: np.ndarray,
                    validation_data: list, validation_labels: np.ndarray) -> list:
    """
    Rank candidates by training accuracy (descending).
    Also compute validation accuracy for each candidate.
    
    Returns sorted list (best training accuracy first):
    [{'dfa': ..., 'training_accuracy': ..., 'validation_accuracy': ..., 'states': ...}, ...]
    """
    scored = []
    for dfa in candidates:
        train_acc = _compute_accuracy(dfa, training_data, training_labels)
        val_acc = _compute_accuracy(dfa, validation_data, validation_labels)
        scored.append({
            'dfa': dfa,
            'training_accuracy': train_acc,
            'validation_accuracy': val_acc,
            'states': len(dfa.states),
        })
    
    # Sort by training accuracy descending (best first)
    scored.sort(key=lambda x: x['training_accuracy'], reverse=True)
    return scored


def _is_valid_dfa(dfa) -> bool:
    """True if *dfa* has >= 2 states and at least one accepting state."""
    if not hasattr(dfa, 'states'):
        return False
    states = list(dfa.states)
    return len(states) >= 2 and any(s.is_accepting for s in states)


def _add_to_history(all_history: list, seen_ids: set, dfa, training_accuracy: float,
                    validation_accuracy: float = None) -> None:
    """Append a valid, not-yet-seen DFA to *all_history*."""
    dfa_id = id(dfa)
    if dfa_id in seen_ids:
        return
    if not _is_valid_dfa(dfa):
        return
    seen_ids.add(dfa_id)
    all_history.append({
        "automata": dfa,
        "training_accuracy": training_accuracy,
        "validation_accuracy": validation_accuracy,
        "states": len(dfa.states),
    })


def _get_candidates(dfa, state: dict, iteration: int, output_dir: str,
                    beam_size: int = 10) -> list:
    """
    生成候選 DFA 的包裝函數。
    
    根據 propose_automata 的邏輯：
    - iteration == 0：初始化指標，返回 dfas 本身
    - iteration > 0：從 previous_best 生成新侯選
    
    Parameters
    ----------
    dfa : 單一 DFA
    iteration : 迭代計數
    
    Returns
    -------
    list of candidate DFAs
    """
    # Debug: check types before proceeding
    if not isinstance(state, dict):
        raise TypeError(f"[ERROR] state must be dict, got {type(state).__name__}: {state}")
    if 'current_idx' not in state:
        raise KeyError(f"[ERROR] state missing 'current_idx' key. Available keys: {list(state.keys())}")
    
    if iteration == 0:
        # 初始化：dfas 被用來初始化指標，previous_best 被忽略
        _AUTO_INSTANCE.propose_automata(
            dfas=[dfa],          # 用來初始化指標
            state=state,
            iteration=0,
            previous_best=[],    # 被忽略
            output_dir=output_dir,
            beam_size=beam_size
        )
        return [dfa]  # iteration=0 時直接返回 dfas
    else:
        # 候選生成：dfas 被忽略，previous_best 用來生成新候選
        return _AUTO_INSTANCE.propose_automata(
            dfas=[],             # 被忽略
            state=state,
            iteration=iteration,
            previous_best=[dfa], # 用來生成新侯選
            output_dir=output_dir,
            beam_size=beam_size
        )


def _common_init(shared_init: SharedInit,
                 batch_size: int,
                 output_dir: str,
                 sampler_fn: Callable) -> Tuple[object, int, list, np.ndarray, dict, list, np.ndarray]:
    """
    Initialize from SharedInit object prepared by beam search.
    
    Parameters
    ----------
    shared_init : SharedInit
        Required. Contains initial_dfa, learner, validation_data, validation_labels.
    batch_size : int
        Used for pre-allocation of state dict
    output_dir : str
        Output directory for any files
    sampler_fn : Callable
        Sampler function to generate perturbation training samples (shared across all baselines)
    
    Returns
    -------
    initial_dfa, initial_states (int), validation_data (list), validation_labels (ndarray), state (dict), 
    training_data (list), training_labels (ndarray)
    """
    global _AUTO_INSTANCE

    os.makedirs(output_dir, exist_ok=True)

    # Extract from SharedInit and SET GLOBAL LEARNER INSTANCE
    _AUTO_INSTANCE = shared_init.learner  # ← CRITICAL: Set the global learner instance!
    initial_dfa = shared_init.initial_dfa.copy() if hasattr(shared_init.initial_dfa, 'copy') else copy.deepcopy(shared_init.initial_dfa)
    initial_states = len(initial_dfa.states) if hasattr(initial_dfa, 'states') else 0
    validation_data = list(shared_init.validation_data)
    validation_labels = np.array(shared_init.validation_labels)
    
    print(f"[Init] Using shared init: {initial_states} states, {len(validation_data)} validation samples")

    # Generate shared perturbation training samples (ONCE, used by all baselines)
    training_data, training_labels = sampler_fn(num_samples=batch_size, compute_labels=True)
    training_labels = np.array(training_labels)
    print(f"[Init] Generated {len(training_data)} shared training samples from perturbation")

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
        'data':             list(validation_data),  # Pre-populate with validation data
        'labels':           np.zeros(prealloc_size, dtype=np.float64),
        'current_idx':      len(validation_data),
    }
    # Initialize labels with actual data
    state['t_order'][()] = ()

    return initial_dfa, initial_states, validation_data, validation_labels, state, training_data, training_labels


def _select_final(all_history: list,
                  select_by: str,
                  accuracy_threshold: float,
                  state_threshold: int,
                  automaton_type: str,
                  initial_dfa,
                  initial_states: int,
                  output_dir: str) -> dict:
    """
    Select final DFA from all_history based on select_by mode.
    Uses pre-computed accuracies from all_history items (no re-computation).

    Modes
    -----
    "accuracy"  – among candidates with accuracy >= accuracy_threshold, pick fewest states.
    "state"     – among candidates with states <= state_threshold, pick highest accuracy.
    Fallback    – best-effort (highest accuracy) when no candidate meets the threshold.
    """
    def _cleanup(best_record: dict, success: bool) -> dict:
        automata = best_record["automata"]
        if automaton_type.upper() == "DFA":
            from learner.dfa_learner import remove_unreachable_states
            remove_unreachable_states(automata)
            _AUTO_INSTANCE.automaton_to_graphviz(
                automata, filename="final_automata", output_dir=output_dir
            )
        # Use pre-computed accuracies from best_record
        return {
            'automata':          automata,
            'training_accuracy': best_record.get("training_accuracy"),
            'validation_accuracy':  best_record.get("validation_accuracy"),
            'size':              best_record.get("states"),
            'coverage':          [],
            'examples':          [],
            'success':           success,
            'false_accept':      [],
            'true_reject':       [],
        }

    print(f"\n[SELECT] select_by='{select_by}', "
          f"accuracy_threshold={accuracy_threshold}, state_threshold={state_threshold}")
    print(f"  Total candidates in history: {len(all_history)}")
    print(f"  Initial states (from shared_init): {initial_states}")

    if not all_history:
        print("  [SELECT] No candidates – returning initial DFA.")
        return {
            'automata':          initial_dfa,
            'training_accuracy': 0.0,
            'validation_accuracy':  0.0,
            'size':              len(initial_dfa.states) if hasattr(initial_dfa, 'states') else 0,
            'initial_states':    initial_states,
            'coverage':          [],
            'examples':          [],
            'success':           False,
            'false_accept':      [],
            'true_reject':       [],
        }

    if select_by == "accuracy":
        qualified = [r for r in all_history if r["training_accuracy"] >= accuracy_threshold]
        if qualified:
            best = min(qualified, key=lambda x: x["states"])
            print(f"  [accuracy mode] {len(qualified)} candidate(s) meet training_accuracy >= {accuracy_threshold}.")
            print(f"  Selected: states={best['states']}, training_accuracy={best['training_accuracy']:.4f}, validation_accuracy={best['validation_accuracy']:.4f}")
            return _cleanup(best, success=True)
        else:
            best = max(all_history, key=lambda x: x["training_accuracy"])
            print(f"  [accuracy mode] No candidate meets accuracy threshold.")
            print(f"  Best-effort: states={best['states']}, training_accuracy={best['training_accuracy']:.4f}, validation_accuracy={best['validation_accuracy']:.4f}")
            return _cleanup(best, success=False)

    elif select_by == "state":
        under = [r for r in all_history if r["states"] <= state_threshold]
        if under:
            best = max(under, key=lambda x: x["training_accuracy"])
            print(f"  [state mode] {len(under)} candidate(s) have states <= {state_threshold}.")
            print(f"  Selected: states={best['states']}, training_accuracy={best['training_accuracy']:.4f}, validation_accuracy={best['validation_accuracy']:.4f}")
            return _cleanup(best, success=True)
        else:
            best = max(all_history, key=lambda x: x["training_accuracy"])
            print(f"  [state mode] No candidate has states <= {state_threshold}.")
            print(f"  Best-effort: states={best['states']}, training_accuracy={best['training_accuracy']:.4f}, validation_accuracy={best['validation_accuracy']:.4f}")
            return _cleanup(best, success=False)

    else:
        raise ValueError(f"Unknown select_by='{select_by}'. Use 'accuracy' or 'state'.")

# ======================================================================
# Simulated Annealing Baseline (using simanneal.Annealer)
# ======================================================================

class DFAAnnealer(Annealer):
    """Simulated Annealing for DFA optimization."""
    
    def __init__(self, initial_dfa, training_data, training_labels, 
                 validation_data, validation_labels, state, 
                 output_dir, beam_size, max_evaluations):
        self.training_data = training_data
        self.training_labels = training_labels
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.propose_state = state  # ← 改名為 propose_state，避免與 simanneal 的 self.state 衝突
        self.output_dir = output_dir
        self.beam_size = beam_size
        self.max_evaluations = max_evaluations
        self.evaluations_count = 0
        self.iteration_count = 0
        self.all_history = []
        self.seen_ids = set()
        self._initialized = False  # Track if initialization has been done
        
        # Make a copy of initial DFA
        self.current_dfa = initial_dfa.copy() if hasattr(initial_dfa, 'copy') else copy.deepcopy(initial_dfa)
        self.best_dfa = self.current_dfa
        
        # Initialize parent Annealer with initial_state as argument (required!)
        super().__init__(initial_state=self.current_dfa)
        self.Tmax = 10.0
        self.Tmin = 0.001
        self.steps = 100
    
    def move(self):
        """
        Generate a single neighbor DFA via propose_automata.
        
        simanneal's Annealer handles Metropolis criterion:
        - Accepts if new_energy < current_energy
        - Accepts with prob exp(-dE/T) otherwise
        - Reverts if rejected
        """
        self.iteration_count += 1
        
        # First-time initialization (iteration == 0 logic)
        if not self._initialized:
            _get_candidates(
                self.current_dfa, self.propose_state, 0,
                self.output_dir, self.beam_size
            )
            self._initialized = True
            return  # Return None on first initialization (simanneal handles it)
        
        if self.evaluations_count >= self.max_evaluations:
            return  # Stop moving when budget exhausted
        
        # Generate a SINGLE neighbor candidate (standard SA: one neighbor move)
        candidates = _get_candidates(
            self.current_dfa, self.propose_state, self.iteration_count,
            self.output_dir, beam_size=1  # ← Generate only 1 candidate
        )
        self.evaluations_count += len(candidates)
        
        if candidates:
            # Randomly select from candidates (or just use the single one)
            candidate = random.choice(candidates) if len(candidates) > 1 else candidates[0]
            
            # Try to accept this candidate (simanneal's Metropolis will make final decision)
            candidate_acc = _compute_accuracy(candidate, self.training_data, self.training_labels)
            
            # Track in history
            _add_to_history(
                self.all_history, self.seen_ids,
                candidate, candidate_acc,
                _compute_accuracy(candidate, self.validation_data, self.validation_labels)
            )
            
            # Update current DFA (simanneal decides accept/reject after energy() call)
            self.current_dfa = candidate
            
            # Update best (track best seen)
            current_best_acc = _compute_accuracy(self.best_dfa, self.training_data, self.training_labels)
            if candidate_acc > current_best_acc:
                self.best_dfa = self.current_dfa
                print(f"  [SA-iter{self.iteration_count}] NEW best: acc={candidate_acc:.4f}, states={len(candidate.states)}, evals: {self.evaluations_count}/{self.max_evaluations}")
    
    def energy(self):
        """Return negative accuracy (we minimize energy)."""
        acc = _compute_accuracy(self.current_dfa, self.training_data, self.training_labels)
        return -acc  # Negative because Annealer minimizes energy


def sa_dfa_search(sampler_fn: Callable,
                  data_type: str,
                  shared_init: SharedInit,
                  *,
                  accuracy_threshold: float = 1.0,
                  state_threshold: int = 5,
                  select_by: str = "accuracy",
                  init_num_samples: int = 1000,
                  batch_size: int = 100,
                  output_dir: str = "test_result/sa",
                  beam_size: int = 1,
                  steps: int = 500,  # 增加至500（從100），每個step調用_get_candidates
                  T_max: float = 10.0,
                  T_min: float = 0.001,
                  max_evaluations: int = 500,
                  **kwargs) -> dict:
    """
    Simulated Annealing for DFA search using simanneal.Annealer.
    
    Requires SharedInit from beam search containing initial DFA and validation data.
    
    Parameters
    ----------
    shared_init       : SharedInit – from beam search (required)
    steps             : int – SA steps
    T_max, T_min      : float – temperature range
    max_evaluations   : int – max propose_automata() calls (budget limit)
    
    Returns
    -------
    dict with same keys as anchor_beam()
    """
    print("=" * 70)
    print("[SA] Initialising Simulated Annealing…")
    
    if shared_init is None:
        raise ValueError("[SA] FATAL: shared_init is required (from beam search)")
    
    initial_dfa, initial_states, validation_data, validation_labels, state, training_data, training_labels = _common_init(
        shared_init, batch_size, output_dir, sampler_fn
    )

    # Create annealer
    annealer = DFAAnnealer(
        initial_dfa, training_data, training_labels,
        validation_data, validation_labels, state,
        output_dir, beam_size, max_evaluations
    )
    annealer.Tmax = T_max
    annealer.Tmin = T_min
    annealer.steps = steps
    
    # Run SA
    print(f"[SA] Running {steps} steps with T_max={T_max}, T_min={T_min}")
    print(f"[SA] Each step calls _get_candidates (max {steps} times until budget exhausted)")
    best_dfa, best_energy = annealer.anneal()
    
    # Prepare all_history with collected candidates
    all_history = annealer.all_history
    initial_train_acc = _compute_accuracy(initial_dfa, training_data, training_labels)
    initial_val_acc = _compute_accuracy(initial_dfa, validation_data, validation_labels)
    _add_to_history(all_history, annealer.seen_ids, initial_dfa, initial_train_acc, initial_val_acc)
    
    print(f"[SA] Completed {annealer.evaluations_count} evaluations")
    print(f"[SA] Best DFA state: {len(best_dfa.states)}")
    print(f"[SA] Best accuracy: {-best_energy:.4f}")
    
    # Note: SA doesn't collect per-iteration stats, so skip plot generation
    
    gc.collect()

    result = _select_final(
        all_history, select_by, accuracy_threshold, state_threshold,
        "DFA", initial_dfa, initial_states, output_dir,
    )
    result['initial_states'] = initial_states  # Add missing initial_states to result
    return result


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
                  shared_init: SharedInit,
                  *,
                  accuracy_threshold: float = 1.0,
                  state_threshold: int = 5,
                  select_by: str = "accuracy",
                  init_num_samples: int = 1000,
                  batch_size: int = 100,
                  output_dir: str = "test_result/ga",
                  beam_size: int = 1,
                  population_size: int = 20,
                  tournament_size: int = 2,
                  max_evaluations: int = 500,
                  **kwargs) -> dict:
    """
    Original Genetic Algorithm for DFA search (no crossover, mutation-only).
    
    Since DFA crossover is not well-defined, uses only selection + mutation.
    
    Algorithm:
    1. Initialize population with initial_dfa
    2. While budget allows:
       a) Fill population via tournament selection to population_size
       b) Always mutate via propose_automata (100% mutation rate)
       c) Evaluate and add to population
    3. Return best individual from all history
    
    Requires SharedInit from beam search containing initial DFA and validation data.
    
    Parameters
    ----------
    shared_init       : SharedInit – from beam search (required)
    population_size   : int – fixed population size
    tournament_size   : int – tournament selection size
    max_evaluations   : int – max candidates generated (budget limit)
    
    Returns
    -------
    dict with same keys as anchor_beam()
    """
    print("=" * 70)
    print("[GA-Original] Initialising Genetic Algorithm (mutation-only, no crossover)…")
    
    if shared_init is None:
        raise ValueError("[GA] FATAL: shared_init is required (from beam search)")
    
    initial_dfa, initial_states, validation_data, validation_labels, state, training_data, training_labels = _common_init(
        shared_init, batch_size, output_dir, sampler_fn
    )

    all_history: List[dict] = []
    seen_ids: set = set()
    evaluations_count = 0  # Track propose_automata() calls

    # Training data is now shared from _common_init

    # Seed history with initial DFA
    initial_train_acc = _compute_accuracy(initial_dfa, training_data, training_labels)
    initial_val_acc = 1.0
    _add_to_history(all_history, seen_ids, initial_dfa, initial_train_acc, initial_val_acc)

    # -------- DEAP toolbox --------
    toolbox = base.Toolbox()

    def _create_individual(dfa_seed):
        return creator.DFAIndividual([dfa_seed.copy()])

    def _evaluate(individual):
        dfa = individual[0]
        train_acc = _compute_accuracy(dfa, training_data, training_labels)
        val_acc = _compute_accuracy(dfa, validation_data, validation_labels)
        _add_to_history(all_history, seen_ids, dfa, train_acc, val_acc)
        return (train_acc,)

    toolbox.register("evaluate", _evaluate)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # Initialize population: seed with initial DFA
    population = [_create_individual(initial_dfa)]
    
    # Evaluate initial population
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    print(f"[GA] Initialized population: {len(population)} individual(s)")
    print(f"[GA] Population size per generation: {population_size} (always mutate each individual)")
    
    # Data collection for visualization
    generation_stats = []
    gen = 0
    prev_evals = 0  # Track progress - to detect stalled loops
    no_progress_count = 0  # Consecutive generations with no new evaluations
    
    # Evolution loop: controlled by evaluations_count AND progress detection
    while evaluations_count < max_evaluations:
        gen += 1
        print(f"\n[GA-Gen] Generation {gen}, evals: {evaluations_count}/{max_evaluations}")
        
        #  If no new evaluations since last generation, stop
        if evaluations_count == prev_evals:
            no_progress_count += 1
            print(f"    [GA STALL] No progress for {no_progress_count} consecutive generations")
            if no_progress_count >= 3:
                print(f"[GA] Stopping: {no_progress_count} generations with no progress (population likely converged)")
                break
        else:
            no_progress_count = 0
        prev_evals = evaluations_count
        
        # Fill population via tournament selection
        while len(population) < population_size:
            # Break if budget exhausted
            if evaluations_count >= max_evaluations:
                print(f"    [GA Budget exhausted] Stopping population fill at {len(population)}/{population_size}")
                break
            
            # Select parent via tournament selection
            if len(population) > 0:
                parent = toolbox.select(population, 1)[0]
            else:
                parent = _create_individual(initial_dfa)
            
            # Always mutate (no random copy of parent)
            try:
                dfa = parent[0]
                iteration = gen * population_size
                candidates = _get_candidates(dfa, state, iteration, output_dir, beam_size)
                evaluations_count += len(candidates)
                
                if candidates:
                    # Pick best candidate from this mutation
                    scored = _rank_candidates(candidates, training_data, training_labels,
                                             validation_data, validation_labels)
                    child = _create_individual(scored[0]['dfa'])
                    child.fitness.values = toolbox.evaluate(child)
                    population.append(child)
                else:
                    # No candidates, copy parent as fallback
                    child = _create_individual(parent[0])
                    child.fitness.values = toolbox.evaluate(child)
                    population.append(child)
            except Exception as exc:
                print(f"    [Warning] Mutation failed: {exc}")
                child = _create_individual(parent[0])
                child.fitness.values = toolbox.evaluate(child)
                population.append(child)
        
        # Budget check before keeping population
        if evaluations_count >= max_evaluations:
            print(f"[GA] Budget exhausted. Stopping after generation {gen}")
            break
        
        # Keep population at fixed size
        population = population[:population_size]
        
        # Log statistics
        fitnesses = [ind.fitness.values[0] for ind in population]
        best_acc = max(fitnesses)
        avg_acc = sum(fitnesses) / len(fitnesses)
        sizes = [len(ind[0].states) if hasattr(ind[0], 'states') else 0 for ind in population]
        
        print(f"  [GA-Gen] Best acc this gen: {best_acc:.4f}, Avg: {avg_acc:.4f}, Avg states: {np.mean(sizes):.1f}")
        
        generation_stats.append({
            'generation': gen,
            'best_accuracy': best_acc,
            'avg_accuracy': avg_acc,
            'best_states': min(sizes) if sizes else 0,
            'avg_states': np.mean(sizes) if sizes else 0,
        })
    
    print(f"\n[GA-Original] Completed {gen} generations, {len(all_history)} candidates total, final evals: {evaluations_count}/{max_evaluations}")
    
    gc.collect()

    result = _select_final(
        all_history, select_by, accuracy_threshold, state_threshold,
        "DFA", initial_dfa, initial_states, output_dir,
    )
    result['initial_states'] = initial_states  # Add missing initial_states to result
    return result


# ======================================================================
# Particle Swarm Optimisation (using pyswarms.GlobalBestPSO)
# ======================================================================

def pso_dfa_search(sampler_fn: Callable,
                   data_type: str,
                   shared_init: SharedInit,
                   *,
                   accuracy_threshold: float = 1.0,
                   state_threshold: int = 5,
                   select_by: str = "accuracy",
                   init_num_samples: int = 1000,
                   batch_size: int = 100,
                   output_dir: str = "test_result/pso",
                   beam_size: int = 1,
                   n_particles: int = 10,
                   max_evaluations: int = 500,
                   **kwargs) -> dict:
    """
    Original Particle Swarm Optimisation for DFA search using pyswarms.GlobalBestPSO.
    
    Simple and pure PSO: particles search the DFA space via propose_automata calls.
    Budget controlled by max_evaluations: iters × n_particles ≈ max_evaluations.
    
    Requires SharedInit from beam search containing initial DFA and validation data.
    
    Parameters
    ----------
    shared_init       : SharedInit – from beam search (required)
    n_particles       : int – number of particles in swarm
    beam_size         : int – beam_size for candidate generation (fixed)
    max_evaluations   : int – max candidates generated (budget limit, controls iters)
    
    Returns
    -------
    dict with same keys as anchor_beam()
    """
    print("=" * 70)
    print("[PSO-Original] Initialising Particle Swarm Optimisation (original pure PSO)…")
    
    if shared_init is None:
        raise ValueError("[PSO] FATAL: shared_init is required (from beam search)")
    
    initial_dfa, initial_states, validation_data, validation_labels, state, training_data, training_labels = _common_init(
        shared_init, batch_size, output_dir, sampler_fn
    )

    all_history: List[dict] = []
    seen_ids: set = set()
    evaluations_count = [0]  # Use list for mutable reference
    
    # Add initial DFA to history
    initial_train_acc = _compute_accuracy(initial_dfa, training_data, training_labels)
    initial_val_acc = _compute_accuracy(initial_dfa, validation_data, validation_labels)
    _add_to_history(all_history, seen_ids, initial_dfa, initial_train_acc, initial_val_acc)
    
    best_dfa = initial_dfa
    best_accuracy = initial_train_acc
    iteration_count = [0]
    
    def objective_function(position_array):
        """
        Objective function for PSO (GlobalBestPSO) with Operation Strength. Used for computing fitness of particles in each iteration.
        
        Simple original PSO: each iteration evaluates all n_particles.
        Budget control: stop generating new candidates when evaluations_count >= max_evaluations.
        
        position_array: shape (n_particles, dimensions)
            pos[0] ∈ [0, 1] → operation strength (0=conservative: generate fewer candidates, 1=aggressive: generate more candidates)
        
        Returns: shape (n_particles,) with fitness for each particle
        """
        nonlocal best_dfa, best_accuracy
        
        # Handle single particle case
        if position_array.ndim == 1:
            position_array = position_array.reshape(1, -1)
        
        fitness_values = np.zeros(len(position_array))
        
        for particle_idx, pos in enumerate(position_array):
            if evaluations_count[0] >= max_evaluations:
                fitness_values[particle_idx] = best_accuracy
                continue
            
            try:
                # Scheme 1: Operation Strength [0, 1] → beam_size [1, 10]
                strength = np.clip(pos[0], 0, 1)
                search_beam_size = max(1, int(1 + strength * 9))  # Maps [0,1] to [1,10]
                iteration = iteration_count[0] * n_particles + particle_idx
                
                candidates = _get_candidates(best_dfa, state, iteration, output_dir, beam_size=search_beam_size)
                
                if not candidates:
                    fitness_values[particle_idx] = best_accuracy
                    continue
                
                evaluations_count[0] += len(candidates)
                
                # Rank and select best candidate
                scored = _rank_candidates(
                    candidates, training_data, training_labels,
                    validation_data, validation_labels
                )
                
                top_candidate = scored[0]
                acc = top_candidate['training_accuracy']
                
                # Track in history
                _add_to_history(all_history, seen_ids, top_candidate['dfa'], acc, 
                               top_candidate['validation_accuracy'])
                
                # Update best
                best_accuracy = acc
                best_dfa = top_candidate['dfa']
                print(f"  [PSO-iter{iteration_count[0]}-p{particle_idx}] "
                      f"NEW best: acc={acc:.4f}, states={top_candidate['states']}, "
                      f"evals: {evaluations_count[0]}/{max_evaluations}")
                
                fitness_values[particle_idx] = acc
                
            except Exception as exc:
                print(f"[ERROR] PSO particle {particle_idx}: {exc}")
                import traceback
                traceback.print_exc()
                fitness_values[particle_idx] = best_accuracy
        
        iteration_count[0] += 1
        return fitness_values
    
    # Create and configure PSO
    print(f"[PSO] Running PSO with {n_particles} particles, baseline beam_size={beam_size}")
    
    # iters × n_particles ≈ max_evaluations
    iters = max(10, (max_evaluations + n_particles - 1) // n_particles)
    print(f"[PSO] Iters={iters} (each iteration evaluates {n_particles} particles, "
          f"total ~{iters * n_particles} evaluations)")
    
    # PSO options
    options = {
        'c1': 0.5,      # Cognitive parameter
        'c2': 1.5,      # Social parameter
        'w': 0.9,       # Inertia weight
        'k': n_particles,
        'p': 2          # Minkowski norm dimension
    }
    
    # Initialize PSO optimizer
    # dimensions=1: particle position is operation strength [0, 1]
    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=1,
        options=options,
        bounds=(np.array([0.0]), np.array([1.0])),
        velocity_clamp=(-1, 1)
    )
    
    # Run optimization (returns best_cost, best_position)
    best_cost, _ = optimizer.optimize(
        objective_function,
        iters=iters,
        verbose=False
    )
    
    print(f"[PSO] PSO optimization completed")
    print(f"[PSO] Best cost (accuracy): {best_cost:.4f}")
    
    print(f"\n[PSO] Completed {evaluations_count[0]} evaluations in {iteration_count[0]} iterations")
    print(f"[PSO] Best accuracy: {best_accuracy:.4f}")
    
    # Note: PSO doesn't collect per-iteration stats in the same way, so skip plot generation
    
    gc.collect()

    result = _select_final(
        all_history, select_by, accuracy_threshold, state_threshold,
        "DFA", initial_dfa, initial_states, output_dir,
    )
    result['initial_states'] = initial_states  # Add missing initial_states to result
    return result
