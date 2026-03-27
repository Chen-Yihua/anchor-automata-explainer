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
from learner.pso_optimizer import PSOAutomataOptimizer

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
    def _cleanup(best_record: dict, success: bool, reason: str = "") -> dict:
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
            'reason':            reason,  # Explain why this result was selected
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
            'reason':            'No candidates generated. Returning initial DFA only.',
            'false_accept':      [],
            'true_reject':       [],
        }

    if select_by == "accuracy":
        qualified = [r for r in all_history if r["training_accuracy"] >= accuracy_threshold]
        if qualified:
            best = min(qualified, key=lambda x: x["states"])
            print(f"  [accuracy mode] {len(qualified)} candidate(s) meet training_accuracy >= {accuracy_threshold}.")
            print(f"  Selected: states={best['states']}, training_accuracy={best['training_accuracy']:.4f}, validation_accuracy={best['validation_accuracy']:.4f}")
            return _cleanup(best, success=True, reason="Found candidate meeting accuracy threshold")
        else:
            best = max(all_history, key=lambda x: x["training_accuracy"])
            print(f"  [accuracy mode] No candidate meets accuracy threshold.")
            print(f"  Best-effort: states={best['states']}, training_accuracy={best['training_accuracy']:.4f}, validation_accuracy={best['validation_accuracy']:.4f}")
            return _cleanup(best, success=False, reason=f"No candidate meets accuracy >= {accuracy_threshold}. Returning best-effort with highest accuracy.")

    elif select_by == "state":
        under = [r for r in all_history if r["states"] <= state_threshold]
        if under:
            best = max(under, key=lambda x: x["training_accuracy"])
            print(f"  [state mode] {len(under)} candidate(s) have states <= {state_threshold}.")
            print(f"  Selected: states={best['states']}, training_accuracy={best['training_accuracy']:.4f}, validation_accuracy={best['validation_accuracy']:.4f}")
            return _cleanup(best, success=True, reason="Found candidate meeting state threshold")
        else:
            best = max(all_history, key=lambda x: x["training_accuracy"])
            print(f"  [state mode] No candidate has states <= {state_threshold}.")
            print(f"  Best-effort: states={best['states']}, training_accuracy={best['training_accuracy']:.4f}, validation_accuracy={best['validation_accuracy']:.4f}")
            return _cleanup(best, success=False, reason=f"No candidate has states <= {state_threshold}. Returning best-effort with highest accuracy.")

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
        
        Stops immediately when evaluations_count >= max_evaluations.
        """
        # Stop if budget exhausted (check FIRST before anything else)
        if self.evaluations_count >= self.max_evaluations:
            # Signal early termination by returning - simanneal will exit gracefully
            print(f"[SA] Budget exhausted: {self.evaluations_count} >= {self.max_evaluations}")
            self.user_exit = True
            return
        
        self.iteration_count += 1
        
        # First-time initialization (iteration == 0 logic)
        if not self._initialized:
            _get_candidates(
                self.current_dfa, self.propose_state, 0,
                self.output_dir, self.beam_size
            )
            self._initialized = True
            return  # Return None on first initialization (simanneal handles it)
        
        # Generate a SINGLE neighbor candidate (standard SA: one neighbor move)
        candidates = _get_candidates(
            self.current_dfa, self.propose_state, self.iteration_count,
            self.output_dir, beam_size=1  # ← Generate only 1 candidate
        )
        self.evaluations_count += len(candidates)
        
        if candidates:
            # Randomly select from candidates (or just use the single one)
            candidate = random.choice(candidates) if len(candidates) > 1 else candidates[0]

            # Compute candidate accuracy (for logging/history)
            candidate_acc = _compute_accuracy(candidate, self.training_data, self.training_labels)

            # Track in history
            _add_to_history(
                self.all_history, self.seen_ids,
                candidate, candidate_acc,
                _compute_accuracy(candidate, self.validation_data, self.validation_labels)
            )

            # Update Annealer state to the candidate so simanneal can evaluate energy() and revert if needed
            # simanneal uses self.state for its internal bookkeeping so we must assign there.
            try:
                self.state = candidate
            except Exception:
                # Fallback: keep a separate reference but energy() will rely on self.state
                self.state = candidate

            # Also keep current_dfa reference in this wrapper for debugging/printing convenience
            self.current_dfa = candidate

            # Update best (track best seen so far)
            current_best_acc = _compute_accuracy(self.best_dfa, self.training_data, self.training_labels)
            if candidate_acc > current_best_acc:
                self.best_dfa = copy.deepcopy(candidate)
                print(f"  [SA-iter{self.iteration_count}] NEW best: acc={candidate_acc:.4f}, states={len(candidate.states)}, evals: {self.evaluations_count}/{self.max_evaluations}")
            
            # Early stopping: if we reach 2 states, stop the search
            if len(candidate.states) == 2:
                print(f"  [SA] Early stopping: reached 2 states at iteration {self.iteration_count}")
                self.user_exit = True
    
    def energy(self):
        """Return negative accuracy (we minimize energy)."""
        # Use simanneal's canonical state (`self.state`) for energy calculation so revert works correctly
        dfa_to_eval = getattr(self, 'state', None) or self.current_dfa
        acc = _compute_accuracy(dfa_to_eval, self.training_data, self.training_labels)
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
    Standard Genetic Algorithm for DFA search (no crossover, mutation-only).
    
    Since DFA crossover is not well-defined, uses only selection + mutation.
    
    Algorithm:
    1. Initialize population with initial_dfa
    2. While budget allows:
       a) Generate population_size new offspring via tournament selection + mutation
       b) Evaluate all offspring
       c) Replace entire population with new offspring (generational replacement)
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
    
    # Evolution loop: Standard GA - generate entire new generation each iteration
    while evaluations_count < max_evaluations:
        gen += 1
        print(f"\n[GA-Gen] Generation {gen}, evals: {evaluations_count}/{max_evaluations}")
        
        # Generate new offspring for this generation
        offspring = []
        
        # Create population_size new individuals via mutation
        for individual_idx in range(population_size):
            # Break if budget exhausted
            if evaluations_count >= max_evaluations:
                print(f"    [GA Budget exhausted] Generated {individual_idx}/{population_size} offspring")
                break
            
            try:
                # Select parent via tournament selection
                parent = toolbox.select(population, tournament_size)[0]
                dfa = parent[0]
                
                # Mutate parent to generate candidates
                iteration = gen * population_size + individual_idx
                candidates = _get_candidates(dfa, state, iteration, output_dir, beam_size)
                evaluations_count += len(candidates)
                
                if candidates:
                    # Pick best candidate from this mutation
                    scored = _rank_candidates(candidates, training_data, training_labels,
                                             validation_data, validation_labels)
                    child = _create_individual(scored[0]['dfa'])
                    child.fitness.values = toolbox.evaluate(child)
                    offspring.append(child)
                else:
                    # No candidates, copy parent as fallback
                    child = _create_individual(parent[0])
                    child.fitness.values = toolbox.evaluate(child)
                    offspring.append(child)
                    
            except Exception as exc:
                print(f"    [Warning] Generating offspring {individual_idx} failed: {exc}")
                # Fallback: copy best individual from current population
                if len(population) > 0:
                    best_ind = max(population, key=lambda x: x.fitness.values[0])
                    child = _create_individual(best_ind[0])
                    child.fitness.values = best_ind.fitness.values
                    offspring.append(child)
                else:
                    # Last resort: use initial DFA
                    child = _create_individual(initial_dfa)
                    child.fitness.values = toolbox.evaluate(child)
                    offspring.append(child)
        
        # Replace population with new offspring (standard GA replacement strategy)
        population[:] = offspring
        
        print(f"    [GA-Gen] Generated {len(offspring)} offspring, total population size: {len(population)}")
        
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
        
        # Early stopping: if any individual has 2 states, stop the search
        if min(sizes) == 2:
            print(f"  [GA] Early stopping: reached 2 states at generation {gen}")
            break
    
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
    Particle Swarm Optimisation for DFA search using PSOAutomataOptimizer.
    
    Uses the PSOAutomataOptimizer from pso_optimizer.py for state minimization
    while maintaining accuracy above a threshold.
    
    Requires SharedInit from beam search containing initial DFA and validation data.
    
    Parameters
    ----------
    shared_init       : SharedInit – from beam search (required)
    n_particles       : int – number of particles in swarm
    beam_size         : int – beam_size for candidate generation (not used, kept for compatibility)
    max_evaluations   : int – controls max iterations
    
    Returns
    -------
    dict with same keys as anchor_beam()
    """
    print("=" * 70)
    print("[PSO] Initialising Particle Swarm Optimisation (PSOAutomataOptimizer)…")
    
    if shared_init is None:
        raise ValueError("[PSO] FATAL: shared_init is required (from beam search)")
    
    initial_dfa, initial_states, validation_data, validation_labels, state, training_data, training_labels = _common_init(
        shared_init, batch_size, output_dir, sampler_fn
    )

    all_history: List[dict] = []
    seen_ids: set = set()
    
    # Add initial DFA to history
    initial_train_acc = _compute_accuracy(initial_dfa, training_data, training_labels)
    initial_val_acc = _compute_accuracy(initial_dfa, validation_data, validation_labels)
    _add_to_history(all_history, seen_ids, initial_dfa, initial_train_acc, initial_val_acc)
        
    # Calculate iterations based on max_evaluations and n_particles
    # Each PSO iteration evaluates n_particles candidates, so:
    # n_iterations ≈ max_evaluations / n_particles
    # Clamp between [5, 100] to ensure reasonable bounds
    n_iterations = max_evaluations // n_particles
    
    print(f"[PSO] Configuration:")
    print(f"  Particles: {n_particles}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Threshold: {accuracy_threshold:.2f}")
    print(f"  Training samples: {len(training_data)}")
    print(f"  Initial states: {initial_states}")
    
    # Create PSOAutomataOptimizer
    try:
        optimizer = PSOAutomataOptimizer(
            initial_dfa=initial_dfa,
            threshold=accuracy_threshold,
            data=training_data,
            labels=training_labels,            
            validation_data=validation_data,
            validation_labels=validation_labels,            
            learner=shared_init.learner,
            n_particles=n_particles,
            n_iterations=n_iterations,
            w=0.7,
            c1=1.5,
            c2=1.5,
            verbose=True,
            max_evaluations=max_evaluations,
            slot_beam_size=max(1, beam_size)
        )
    except Exception as e:
        print(f"[PSO ERROR] Failed to initialize PSOAutomataOptimizer: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return initial DFA
        result = _select_final(
            all_history, select_by, accuracy_threshold, state_threshold,
            "DFA", initial_dfa, initial_states, output_dir,
        )
        result['initial_states'] = initial_states
        return result
    
    # Run PSO optimization
    print(f"\n[PSO] Starting optimization...")
    try:
        pso_result = optimizer.optimize(n_particles=n_particles, n_iterations=n_iterations, save_trajectory=True)
        
        print(f"\n[PSO] Optimization completed successfully")
        print(f"  Best states: {pso_result['best_states']}")
        print(f"  Best accuracy: {pso_result['best_accuracy']:.4f}")
        print(f"  Best validation accuracy: {pso_result['best_validation_accuracy']:.4f}")
        print(f"  Best loss: {pso_result['best_loss']:.4f}")
        
        # Collect all candidates from PSO history into all_history (use _add_to_history for consistency)
        if 'all_history' in pso_result:
            for candidate in pso_result['all_history']:
                try:
                    _add_to_history(all_history, seen_ids, candidate['automata'],
                                    candidate.get('training_accuracy', 0.0),
                                    candidate.get('validation_accuracy', 0.0))
                except Exception:
                    continue
        # Optionally report how many evaluations PSO performed
        if 'evaluations' in pso_result:
            print(f"  [PSO] Evaluations used: {pso_result['evaluations']} / {pso_result.get('max_evaluations')}")
        
        # Check if we have a 2-state solution from PSO
        min_states = min([record['states'] for record in all_history] if all_history else [float('inf')])
        if min_states == 2:
            print(f"  [PSO] Early stopping: reached 2 states")
        
    except Exception as e:
        print(f"[PSO WARNING] Optimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n[PSO] Total candidates collected: {len(all_history)}")
    
    gc.collect()

    result = _select_final(
        all_history, select_by, accuracy_threshold, state_threshold,
        "DFA", initial_dfa, initial_states, output_dir,
    )
    result['initial_states'] = initial_states  # Add missing initial_states to result
    return result
