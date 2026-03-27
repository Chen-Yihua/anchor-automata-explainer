"""
PSO-based DFA Optimizer

This module implements a Particle Swarm Optimization (PSO) approach for optimizing
Deterministic Finite Automata (DFA) by minimizing the number of states while
maintaining training accuracy above a specified threshold.

Key Components:
1. Objective function: Penalizes low accuracy, rewards state reduction
2. Discrete space mapping: PSO particle positions → DFA modifications
3. Integration with existing DFA operators: _propose_delete, _propose_merge, _propose_delta
4. State sanitation: Applies remove_unreachable_states after each DFA modification
"""

import gc
import math
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from copy import deepcopy
import random

try:
    from pyswarms.single.global_best import GlobalBestPSO
    PSO_AVAILABLE = True
except ImportError:
    PSO_AVAILABLE = False
    print("[WARNING] pyswarms not found. PSO optimizer will not be available.")

from .dfa_learner import remove_unreachable_states


class PSOAutomataOptimizer:
    """
    Particle Swarm Optimization-based DFA optimizer that searches for the minimal
    DFA satisfying accuracy constraints while minimizing state count.
    
    Core Concept:
    - Particles represent sequences of DFA modifications (DELETE, MERGE, DELTA)
    - Particle positions are continuous vectors mapped to discrete DFA operations
    - Fitness is computed based on: Accuracy (must satisfy THRESHOLD) + State Count
    - PSO's pbest and gbest guide particles toward better DFAs
    
    Attributes:
        initial_dfa: Starting DFA for optimization
        threshold: Minimum required accuracy (0.0 to 1.0)
        data: Training sequences
        labels: Training labels (binary)
        learner: DFALearner instance for accessing _propose_delete, _propose_merge, _propose_delta
        state_metrics: Dictionary for tracking DFA evaluation metrics
        n_particles: Number of particles in the swarm
        n_iterations: Number of PSO iterations
        w: Inertia weight for PSO
        c1: Cognitive coefficient for PSO
        c2: Social coefficient for PSO
        verbose: Enable detailed logging
    """
    
    def __init__(self, 
                 initial_dfa,
                 threshold: float = 0.8,
                 data: List = None,
                 labels: np.ndarray = None,
                 learner = None,
                 validation_data: List = None,
                 validation_labels: np.ndarray = None,
                 n_particles: int = 10,
                 n_iterations: int = 20,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 verbose: bool = True,
                 max_evaluations: Optional[int] = None,
                 slot_beam_size: Optional[int] = None,
                 max_ops_per_iteration: Optional[int] = None):
        """
        Initialize PSO optimizer.
        
        Parameters
        ----------
        initial_dfa : Dfa
            Starting DFA for optimization
        threshold : float
            Minimum required training accuracy (default: 0.8)
        data : list
            Training sequences
        labels : np.ndarray
            Training labels (1 for accept, 0 for reject)
        learner : DFALearner
            DFALearner instance with access to _propose_delete, _propose_merge, _propose_delta
        validation_data : list, optional
            Validation sequences for evaluating DFA quality (default: None)
        validation_labels : np.ndarray, optional
            Validation labels (default: None)
        n_particles : int
            Number of particles in swarm (default: 10)
        n_iterations : int
            Number of PSO iterations (default: 20)
        w : float
            Inertia weight (default: 0.7)
        c1 : float
            Cognitive coefficient (default: 1.5)
        c2 : float
            Social coefficient (default: 1.5)
        verbose : bool
            Enable logging (default: True)
        """
        if not PSO_AVAILABLE:
            raise ImportError("pyswarms is required for PSOAutomataOptimizer. Install with: pip install pyswarms")
        
        self.initial_dfa = initial_dfa
        self.threshold = threshold
        self.data = data or []
        self.labels = labels if labels is not None else np.array([])
        self.validation_data = validation_data or []
        self.validation_labels = validation_labels if validation_labels is not None else np.array([])
        self.learner = learner
        self.verbose = verbose
        
        # PSO edit-space configuration
        self.slot_count = 4                 # Fixed number of operation slots per particle
        self.slot_width = 3                 # (operation_type, target_signature, probability)
        self.max_ops_per_iteration = max(1, max_ops_per_iteration or 3)
        slot_beam_default = slot_beam_size if slot_beam_size is not None else 5
        self.slot_beam_size = max(1, slot_beam_default)
        self.dimensions = self.slot_count * self.slot_width

        # PSO hyperparameters
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # State tracking
        self.state_metrics: Dict[int, Dict] = {}
        self._init_state_metrics()
        
        # DFA sequence for each particle: list of (operation_type, candidate_idx)
        # operation_type: 0=DELETE, 1=MERGE, 2=DELTA
        self.particle_dfas: Dict[int, Any] = {}  # particle_id -> current_dfa
        self.particle_histories: Dict[int, List[Tuple[str, int]]] = {}  # particle_id -> list of operations
        
        # Global best tracking
        self.gbest_dfa = None
        self.gbest_fitness = float('inf')
        self.gbest_accuracy = 0.0
        self.gbest_val_accuracy = 0.0
        self.gbest_states = float('inf')
        
        # All candidates history (for comparison with other methods)
        self.all_history: List[Dict] = []
        self.seen_ids: set = set()
        # Evaluation budget tracking
        self.evaluations_count: int = 0
        self.max_evaluations: Optional[int] = max_evaluations
        
        # Evaluate initial DFA and set as initial global best
        initial_accuracy, initial_states, initial_loss, initial_val_acc = self._evaluate_and_cache(self.initial_dfa)
        self._update_gbest(self.initial_dfa, initial_accuracy, initial_states, initial_loss, initial_val_acc)
        self._add_to_history(self.initial_dfa, initial_accuracy, initial_val_acc)
        
        if self.verbose:
            val_str = f", val_acc={initial_val_acc:.4f}" if len(self.validation_data) > 0 else ""
            print(f"[PSO Init] Initial DFA: {len(self.initial_dfa.states)} states, accuracy={initial_accuracy:.4f}{val_str}")
    
    def _init_state_metrics(self):
        """Initialize state metrics dictionary (used by learner's update_state_metrics)"""
        self.state_metrics = {
            't_idx': {},           # Row indices where automaton applies
            't_nsamples': {},      # Total number of samples
            't_accepted': {},      # Number of accepted samples
            't_positives': {},     # True positives (correctly accepted)
            't_negatives': {},     # True negatives (correctly rejected)
            't_order': {},         # Operation history
            'current_idx': 0,      # Current index
        }
    
    def _compute_validation_accuracy(self, dfa) -> float:
        """Compute validation accuracy on validation dataset."""
        if self.learner is None or len(self.validation_data) == 0:
            return 0.0
        
        accepts = np.array([self.learner.check_path_accepted(dfa, p) for p in self.validation_data])
        lbl = np.asarray(self.validation_labels)
        correct = int(np.sum((lbl == 1) & accepts) + np.sum((lbl == 0) & ~accepts))
        return correct / len(lbl)
    
    def _add_to_history(self, dfa, training_accuracy: float, validation_accuracy: float) -> None:
        """Track DFA in history for final comparison."""
        dfa_id = id(dfa)
        if dfa_id in self.seen_ids:
            return
        self.seen_ids.add(dfa_id)
        self.all_history.append({
            "automata": deepcopy(dfa),
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy,
            "states": len(dfa.states),
        })
    
    def _evaluate_and_cache(self, dfa):
        """
        Evaluate DFA and cache metrics.
        
        Parameters
        ----------
        dfa : Dfa
            DFA to evaluate
            
        Returns
        -------
        tuple
            (accuracy, num_states, loss, validation_accuracy)
        """
        dfa_id = id(dfa)
        
        # Skip if already cached
        if dfa_id in self.state_metrics['t_nsamples']:
            accuracy = self._compute_accuracy(dfa_id)
            num_states = len(dfa.states)
            loss = self._compute_loss(accuracy, num_states)
            val_acc = self._compute_validation_accuracy(dfa)
            return accuracy, num_states, loss, val_acc
        
        # Evaluate using learner's path checking method
        if self.learner and len(self.data) > 0 and len(self.labels) > 0:
            accepts = np.array([self.learner.check_path_accepted(dfa, p) for p in self.data])
            true_accept = np.sum((self.labels == 1) & (accepts == True))
            false_reject = np.sum((self.labels == 0) & (accepts == False))
        else:
            true_accept = 0
            false_reject = 0
            accepts = np.array([False] * len(self.data))
        
        # Cache metrics
        self.state_metrics['t_nsamples'][dfa_id] = float(len(self.data))
        self.state_metrics['t_accepted'][dfa_id] = float(np.sum(accepts))
        self.state_metrics['t_positives'][dfa_id] = float(true_accept)
        self.state_metrics['t_negatives'][dfa_id] = float(false_reject)
        self.state_metrics['t_order'][dfa_id] = []
        
        accuracy = self._compute_accuracy(dfa_id)
        num_states = len(dfa.states)
        loss = self._compute_loss(accuracy, num_states)
        val_acc = self._compute_validation_accuracy(dfa)
        
        if self.verbose:
            val_str = f", val_acc={val_acc:.4f}" if len(self.validation_data) > 0 else ""
            print(f"  [EVAL] Accuracy: {accuracy:.4f}, States: {num_states}, Loss: {loss:.4f}{val_str}")
        
        return accuracy, num_states, loss, val_acc
    
    def _compute_accuracy(self, dfa_id: int) -> float:
        """Compute accuracy from cached metrics."""
        n_samples = self.state_metrics['t_nsamples'].get(dfa_id, 1)
        if n_samples == 0:
            return 0.0
        true_pos = self.state_metrics['t_positives'].get(dfa_id, 0)
        true_neg = self.state_metrics['t_negatives'].get(dfa_id, 0)
        accuracy = (true_pos + true_neg) / n_samples
        return float(accuracy)
    
    def _compute_loss(self, accuracy: float, num_states: int) -> float:
        """
        Compute loss function as per specification:
        - If accuracy < THRESHOLD: Loss = 1000 + (THRESHOLD - accuracy) × 100 (penalty for low accuracy)
        - If accuracy >= THRESHOLD: Loss = num_states (reward for simplification)
        
        Parameters
        ----------
        accuracy : float
            Training accuracy of the DFA
        num_states : int
            Number of states in the DFA
            
        Returns
        -------
        float
            Loss value to minimize
        """
        if accuracy < self.threshold:
            # Strong penalty if accuracy is below threshold
            penalty = 1000.0 + (self.threshold - accuracy) * 100.0
            return penalty
        else:
            # Reward for maintaining accuracy: minimize state count
            return float(num_states)
    
    def _update_gbest(self, dfa, accuracy: float, num_states: int, loss: float, val_accuracy: float = 0.0):
        """Update global best if current DFA is better."""
        if loss < self.gbest_fitness:
            self.gbest_dfa = deepcopy(dfa)
            self.gbest_fitness = loss
            self.gbest_accuracy = accuracy
            self.gbest_val_accuracy = val_accuracy
            self.gbest_states = num_states
            val_str = f", val_acc={val_accuracy:.4f}" if val_accuracy > 0 else ""
            if self.verbose:
                print(f"    [GBEST UPDATE] New best: {num_states} states, accuracy={accuracy:.4f}{val_str}, loss={loss:.4f}")
    
    def objective_function(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        PSO objective function (to minimize).
        
        Parameters
        ----------
        X : np.ndarray
            Particle positions with shape (n_particles, n_dimensions)
            Each row is a particle's position vector
        **kwargs : dict
            Additional keyword arguments (e.g., callback from pyswarms)
            
        Returns
        -------
        np.ndarray
            Loss values for each particle (to minimize)
        """
        n_particles = X.shape[0]
        losses = np.zeros(n_particles)
        
        for particle_id in range(n_particles):
            # Check budget before evaluating this particle
            if self.max_evaluations is not None and self.evaluations_count >= self.max_evaluations:
                # Signal that budget is exhausted by raising an exception to stop optimizer
                raise RuntimeError("PSO budget exhausted")
            # Get or initialize particle DFA
            if particle_id not in self.particle_dfas:
                self.particle_dfas[particle_id] = deepcopy(self.initial_dfa)
                self.particle_histories[particle_id] = []
            
            # Map continuous position to discrete DFA sequence
            dfa, operations = self._map_position_to_dfa(
                particle_id, X[particle_id]
            )
            
            # Ensure DFA is valid (no unreachable states)
            remove_unreachable_states(dfa)
            
            # Evaluate DFA
            accuracy, num_states, loss, val_acc = self._evaluate_and_cache(dfa)

            # Track in history
            self._add_to_history(dfa, accuracy, val_acc)
            
            # Update global best
            self._update_gbest(dfa, accuracy, num_states, loss, val_acc)
            
            losses[particle_id] = loss
        
        return losses
    
    def _map_position_to_dfa(self, particle_id: int, position: np.ndarray) -> Tuple[Any, List[str]]:
        """
        Map continuous particle position to a DFA through discrete operations.
        
        Strategy:
        1. Start from the particle's cached DFA (initial DFA on first visit)
        2. Divide dimensions into phases
        3. Each phase corresponds to ONE modification operation
        4. For each dimension, determine operation type and candidate index
        5. Apply modification, ensuring validity with remove_unreachable_states
        
        Parameters
        ----------
        particle_id : int
            Unique particle identifier
        position : np.ndarray
            Continuous position vector (typically in [-1, 1] range from PSO)
            
        Returns
        -------
        tuple
            (modified_dfa, list_of_operations)
        """
        # Start from the particle's latest DFA so edits accumulate across iterations
        base_dfa = self.particle_dfas.get(particle_id, self.initial_dfa)
        current_dfa = deepcopy(base_dfa)
        operations = []
        
        slot_vectors = self._extract_slot_vectors(position)
        if not slot_vectors:
            return current_dfa, operations

        # Softmax over slot probabilities so velocity updates change selection bias smoothly
        slot_probs = self._softmax([slot['prob_logit'] for slot in slot_vectors])
        for idx, prob in enumerate(slot_probs):
            slot_vectors[idx]['probability'] = prob

        # Execute top-k slots (highest probability first)
        slot_vectors.sort(key=lambda s: s.get('probability', 0.0), reverse=True)
        max_slots = min(self.max_ops_per_iteration, len(slot_vectors))

        for slot_idx in range(max_slots):
            slot = slot_vectors[slot_idx]
            op_type = self._decode_operation_type(slot['op_value'])
            if op_type == "SKIP":
                operations.append("SKIP:INVALID_OP")
                continue

            next_dfa, descriptor = self._apply_slot_operation(
                current_dfa,
                op_type,
                slot['target_value']
            )

            if next_dfa is None:
                operations.append(f"{descriptor}")
                continue

            current_dfa = next_dfa
            prob_value = slot.get('probability', 0.0)
            operations.append(f"{descriptor}:p={prob_value:.2f}")
        
        # Cache particle DFA so the next iteration keeps building on the latest edits
        self.particle_dfas[particle_id] = deepcopy(current_dfa)
        previous_ops = self.particle_histories.get(particle_id, [])
        self.particle_histories[particle_id] = previous_ops + operations
        
        return current_dfa, operations

    def _extract_slot_vectors(self, position: np.ndarray) -> List[Dict[str, float]]:
        slot_vectors: List[Dict[str, float]] = []
        for slot_idx in range(self.slot_count):
            base_idx = slot_idx * self.slot_width
            if base_idx + self.slot_width > len(position):
                break
            slot_vectors.append({
                'slot_idx': slot_idx,
                'op_value': float(position[base_idx]),
                'target_value': float(position[base_idx + 1]),
                'prob_logit': float(position[base_idx + 2]),
            })
        return slot_vectors

    def _decode_operation_type(self, raw_value: float) -> str:
        normalized = self._normalize_to_unit_interval(raw_value)
        if normalized < 1 / 3:
            return "DELETE"
        if normalized < 2 / 3:
            return "MERGE"
        if normalized <= 1.0:
            return "DELTA"
        return "SKIP"

    def _apply_slot_operation(self, current_dfa, op_type: str, target_value: float) -> Tuple[Optional[Any], str]:
        if self.learner is None:
            raise RuntimeError("Learner is required for PSO operations")

        seen_signatures = set()
        candidates: List[Any] = []

        try:
            if op_type == "DELETE":
                candidates = self.learner._propose_delete(
                    current_dfa,
                    self.state_metrics,
                    self.data,
                    self.labels,
                    seen_signatures,
                    beam_size=self.slot_beam_size
                )
                # Count candidates generated by propose_delete (aligned with SA/GA/Beam)
                self.evaluations_count += len(candidates)
            elif op_type == "MERGE":
                candidates = self.learner._propose_merge(
                    current_dfa,
                    self.state_metrics,
                    self.data,
                    self.labels,
                    seen_signatures,
                    beam_size=self.slot_beam_size
                )
                # Count candidates generated by propose_merge (aligned with SA/GA/Beam)
                self.evaluations_count += len(candidates)
            elif op_type == "DELTA":
                delta_candidates = self.learner._propose_delta(
                    current_dfa,
                    self.state_metrics,
                    self.data,
                    self.labels,
                    seen_signatures
                )
                candidates = delta_candidates[:self.slot_beam_size]
                # Count all delta candidates generated (aligned with SA/GA/Beam)
                self.evaluations_count += len(delta_candidates)
            else:
                return None, f"{op_type}:SKIP"
        except Exception as exc:
            if self.verbose:
                print(f"  [WARN] {op_type} proposal failed: {str(exc)[:50]}")
            return None, f"{op_type}:ERROR"

        if not candidates:
            return None, f"{op_type}:SKIP"

        idx = self._select_candidate_index(target_value, len(candidates))
        next_dfa = deepcopy(candidates[idx])
        descriptor = f"{op_type}:idx{idx}"

        for cand in candidates:
            del cand
        del candidates
        gc.collect()

        return next_dfa, descriptor

    def _select_candidate_index(self, raw_value: float, num_candidates: int) -> int:
        if num_candidates <= 0:
            return 0
        normalized = self._normalize_to_unit_interval(raw_value)
        scaled = int(normalized * num_candidates)
        return min(max(scaled, 0), num_candidates - 1)

    def _normalize_to_unit_interval(self, raw_value: float) -> float:
        bounded = math.tanh(raw_value)
        normalized = (bounded + 1.0) / 2.0
        return max(0.0, min(1.0, normalized))

    def _softmax(self, logits: List[float]) -> List[float]:
        if not logits:
            return []
        max_logit = max(logits)
        exp_vals = [math.exp(logit - max_logit) for logit in logits]
        total = sum(exp_vals)
        if total == 0.0:
            return [1.0 / len(logits)] * len(logits)
        return [val / total for val in exp_vals]
    
    def optimize(self, 
                 n_particles: Optional[int] = None,
                 n_iterations: Optional[int] = None,
                 save_trajectory: bool = False) -> Dict[str, Any]:
        """
        Run PSO optimization to find minimal DFA.
        
        Parameters
        ----------
        n_particles : int, optional
            Override default number of particles
        n_iterations : int, optional
            Override default number of iterations
        save_trajectory : bool
            If True, return fitness trajectory for each iteration
            
        Returns
        -------
        dict
            Contains:
            - 'best_dfa': Best DFA found (minimal states, accuracy >= threshold)
            - 'best_accuracy': Accuracy of best DFA
            - 'best_states': Number of states in best DFA
            - 'best_loss': Loss value of best DFA
            - 'iterations': Number of iterations completed
            - 'trajectory': List of best loss per iteration (if save_trajectory=True)
        """
        if not PSO_AVAILABLE:
            raise ImportError("pyswarms is required for optimization")
        
        n_particles = n_particles or self.n_particles
        n_iterations = n_iterations or self.n_iterations
        
        if self.verbose:
            print(f"\n[PSO] Starting optimization...")
            print(f"  Particles: {n_particles}")
            print(f"  Iterations: {n_iterations}")
            print(f"  Threshold: {self.threshold}")
            print(f"  Initial states: {len(self.initial_dfa.states)}")
        
        # PSO configuration
        options = {
            'c1': self.c1,
            'c2': self.c2,
            'w': self.w,
            'k': n_particles,
            'p': 2,  # 2D neighborhood topology
        }
        
        # Determine dimensionality from slot encoding (op type, target, prob)
        dimensions = self.dimensions
        
        # Initialize PSO optimizer
        optimizer = GlobalBestPSO(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            init_pos=np.random.uniform(-1, 1, size=(n_particles, dimensions))
        )
        
        trajectory = []
        
        # Run optimization with custom callback
        def logging_callback(current_cost, dimension, iteration):
            best_loss = np.min(current_cost)
            trajectory.append(best_loss)
            if self.verbose and iteration % 2 == 0:
                print(f"  [Iteration {iteration}] Best loss: {best_loss:.4f}")
        
        try:
            final_cost, final_pos = optimizer.optimize(
                self.objective_function,
                iters=n_iterations,
                verbose=False,
                callback=logging_callback
            )
        except Exception as e:
            if self.verbose:
                print(f"[PSO WARNING] Optimization interrupted: {e}")
        
        if self.verbose:
            print(f"\n[PSO] Optimization complete!")
            print(f"  Best loss: {self.gbest_fitness if not np.isinf(self.gbest_fitness) else 'No valid solution found'}")
            print(f"  Best accuracy: {self.gbest_accuracy:.4f}")
            val_str = f", val_acc={self.gbest_val_accuracy:.4f}" if self.gbest_val_accuracy > 0 else ""
            best_states_str = f"{int(self.gbest_states)}" if not np.isinf(self.gbest_states) else "No valid solution found"
            print(f"  Best states: {best_states_str}{val_str}")
        
        # Safe conversion handling infinity
        best_states = int(self.gbest_states) if not np.isinf(self.gbest_states) else -1
        best_loss = float(self.gbest_fitness) if not np.isinf(self.gbest_fitness) else -1.0
        
        # Determine success and reason based on whether a valid solution was found
        success = not np.isinf(self.gbest_fitness) and self.gbest_dfa is not None
        if success:
            reason = f"PSO found solution with accuracy {self.gbest_accuracy:.4f} and {int(self.gbest_states)} states"
        else:
            reason = "PSO failed to find valid solution (no automaton met validation criteria)"
        
        result = {
            'best_dfa': self.gbest_dfa,
            'best_accuracy': float(self.gbest_accuracy),
            'best_validation_accuracy': float(self.gbest_val_accuracy),
            'best_states': best_states,
            'best_loss': best_loss,
            'iterations': len(trajectory),
            'threshold': self.threshold,
            'all_history': self.all_history,  # Include all candidates for comparison
            'evaluations': int(self.evaluations_count),
            'max_evaluations': int(self.max_evaluations) if self.max_evaluations is not None else None,
            'success': success,                # Whether optimization found a valid solution
            'reason': reason,                  # Explanation of the result
        }
        
        if save_trajectory:
            result['trajectory'] = trajectory
        
        return result


# ============================================================================
# Convenience function for easy integration
# ============================================================================

def optimize_dfa_with_pso(initial_dfa,
                          learner,
                          data: List,
                          labels: np.ndarray,
                          threshold: float = 0.8,
                          n_particles: int = 10,
                          n_iterations: int = 20,
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to quickly optimize a DFA using PSO.
    
    Parameters
    ----------
    initial_dfa : Dfa
        Starting DFA for optimization
    learner : DFALearner
        Learner instance with propose methods
    data : list
        Training sequences
    labels : np.ndarray
        Training labels (binary)
    threshold : float
        Minimum required accuracy (default: 0.8)
    n_particles : int
        Number of particles (default: 10)
    n_iterations : int
        Number of iterations (default: 20)
    verbose : bool
        Enable logging (default: True)
        
    Returns
    -------
    dict
        Optimization results with 'best_dfa', 'best_accuracy', 'best_states', etc.
    """
    optimizer = PSOAutomataOptimizer(
        initial_dfa=initial_dfa,
        threshold=threshold,
        data=data,
        labels=labels,
        learner=learner,
        n_particles=n_particles,
        n_iterations=n_iterations,
        verbose=verbose
    )
    
    return optimizer.optimize(save_trajectory=True)


__all__ = [
    'PSOAutomataOptimizer',
    'optimize_dfa_with_pso',
]
