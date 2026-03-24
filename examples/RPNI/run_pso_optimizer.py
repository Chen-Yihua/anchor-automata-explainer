"""
PSO DFA Optimizer - Usage Example and Test Suite

This example demonstrates how to use the PSOAutomataOptimizer to minimize
DFA state count while maintaining accuracy above a threshold.

Usage:
    python -m examples.RPNI.run_pso_optimizer [dataset] [threshold]
    
Example:
    python -m examples.RPNI.run_pso_optimizer tomita 0.8
"""

import sys
import os
import numpy as np
from typing import List, Tuple
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.learner.dfa_learner import DFALearner, remove_unreachable_states
from src.learner.pso_optimizer import PSOAutomataOptimizer, optimize_dfa_with_pso


def generate_synthetic_data(n_samples: int = 200) -> Tuple[List, np.ndarray]:
    """
    Generate synthetic training data for testing.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
        
    Returns
    -------
    tuple
        (sequences, labels) where sequences is a list of tuples and labels is binary
    """
    sequences = []
    labels = []
    
    symbols = ['0', '1']
    
    for _ in range(n_samples):
        # Generate random sequence
        length = np.random.randint(1, 10)
        seq = tuple([np.random.choice(symbols) for _ in range(length)])
        sequences.append(seq)
        
        # Simple labeling rule: accept if sequence contains mostly '1's
        count_ones = sum(1 for s in seq if s == '1')
        label = 1 if count_ones >= len(seq) / 2 else 0
        labels.append(label)
    
    return sequences, np.array(labels)


def load_tomita_data(tomita_id: int = 1, n_samples: int = 200) -> Tuple[List, np.ndarray]:
    """
    Load Tomita grammar dataset.
    
    Parameters
    ----------
    tomita_id : int
        Tomita grammar ID (1-7)
    n_samples : int
        Number of samples to limit
        
    Returns
    -------
    tuple
        (sequences, labels)
    """
    try:
        from datasets.tomita_loader_dfa import load_tomita_dfa_dataset
        data, labels = load_tomita_dfa_dataset(tomita_id, n_samples)
        return data, labels
    except Exception as e:
        print(f"[WARNING] Could not load Tomita data: {e}")
        return generate_synthetic_data(n_samples)


def example_basic_optimization():
    """
    Basic example: Optimize a synthesized DFA.
    """
    print("\n" + "="*70)
    print("PSO DFA OPTIMIZER - BASIC EXAMPLE")
    print("="*70)
    
    # Load or generate data
    print("\n[1] Loading training data...")
    data, labels = generate_synthetic_data(n_samples=150)
    print(f"    Generated {len(data)} samples")
    print(f"    Positive samples: {np.sum(labels == 1)}")
    print(f"    Negative samples: {np.sum(labels == 0)}")
    
    # Create learner and initial DFA
    print("\n[2] Creating initial DFA...")
    learner = DFALearner()
    
    # Separate positive and negative samples
    pos_samples = [list(s) for s, l in zip(data, labels) if l == 1]
    neg_samples = [list(s) for s, l in zip(data, labels) if l == 0]
    
    initial_dfa = learner.create_init_automata(
        data_type='Tabular',
        positive_samples=pos_samples or [[]],
        negative_samples=neg_samples or [[]]
    )
    
    print(f"    Initial DFA states: {len(initial_dfa.states)}")
    
    # Evaluate initial DFA accuracy
    accepts = np.array([learner.check_path_accepted(initial_dfa, p) for p in data])
    initial_accuracy = np.mean(accepts == labels)
    print(f"    Initial accuracy: {initial_accuracy:.4f}")
    
    # Create PSO optimizer
    print("\n[3] Initializing PSO optimizer...")
    threshold = 0.85
    optimizer = PSOAutomataOptimizer(
        initial_dfa=initial_dfa,
        threshold=threshold,
        data=data,
        labels=labels,
        learner=learner,
        n_particles=8,
        n_iterations=15,
        verbose=True
    )
    
    # Run optimization
    print("\n[4] Running PSO optimization...")
    result = optimizer.optimize(save_trajectory=True)
    
    # Display results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Initial states:        {len(initial_dfa.states)}")
    print(f"Optimized states:      {result['best_states']}")
    print(f"State reduction:       {len(initial_dfa.states) - result['best_states']} "
          f"({100*(len(initial_dfa.states) - result['best_states'])/len(initial_dfa.states):.1f}%)")
    print(f"Accuracy (initial):    {initial_accuracy:.4f}")
    print(f"Accuracy (optimized):  {result['best_accuracy']:.4f}")
    print(f"Threshold:             {threshold:.4f}")
    print(f"Meets threshold:       {'✓ Yes' if result['best_accuracy'] >= threshold else '✗ No'}")
    print(f"Best loss:             {result['best_loss']:.4f}")
    print(f"Iterations completed:  {result['iterations']}")
    
    if 'trajectory' in result:
        print(f"\nOptimization trajectory (best loss per iteration):")
        for i, loss in enumerate(result['trajectory']):
            print(f"  Iter {i}: {loss:.4f}")
    
    return result


def example_with_tomita():
    """
    Example with Tomita grammar dataset.
    """
    print("\n" + "="*70)
    print("PSO DFA OPTIMIZER - TOMITA GRAMMAR EXAMPLE")
    print("="*70)
    
    # Load Tomita data
    print("\n[1] Loading Tomita grammar dataset...")
    tomita_id = 1
    try:
        data, labels = load_tomita_data(tomita_id=tomita_id, n_samples=200)
        print(f"    Loaded Tomita-{tomita_id} dataset: {len(data)} samples")
        print(f"    Positive samples: {np.sum(labels == 1)}")
        print(f"    Negative samples: {np.sum(labels == 0)}")
    except Exception as e:
        print(f"[ERROR] Could not load Tomita data: {e}")
        return None
    
    # Create learner and initial DFA
    print("\n[2] Creating initial DFA...")
    learner = DFALearner()
    
    pos_samples = [list(s) for s, l in zip(data, labels) if l == 1]
    neg_samples = [list(s) for s, l in zip(data, labels) if l == 0]
    
    initial_dfa = learner.create_init_automata(
        data_type='Tabular',
        positive_samples=pos_samples or [[]],
        negative_samples=neg_samples or [[]]
    )
    
    print(f"    Initial DFA states: {len(initial_dfa.states)}")
    
    # Optimize using convenience function
    print("\n[3] Running PSO optimization with convenience function...")
    result = optimize_dfa_with_pso(
        initial_dfa=initial_dfa,
        learner=learner,
        data=data,
        labels=labels,
        threshold=0.9,
        n_particles=10,
        n_iterations=20,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("TOMITA OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Initial states:        {len(initial_dfa.states)}")
    print(f"Optimized states:      {result['best_states']}")
    print(f"Accuracy (optimized):  {result['best_accuracy']:.4f}")
    print(f"Threshold:             0.9")
    
    return result


def benchmark_optimization_time():
    """
    Benchmark: Measure optimization time for different DFA sizes.
    """
    print("\n" + "="*70)
    print("BENCHMARK - OPTIMIZATION TIME")
    print("="*70)
    
    import time
    
    learner = DFALearner()
    
    sizes = [5, 10, 20]
    
    for size in sizes:
        print(f"\nEvaluating {size} particles × 10 iterations...")
        
        # Generate data
        data, labels = generate_synthetic_data(n_samples=100)
        
        # Create simple DFA
        pos_samples = [[str(i % 2)] * 5 for i in range(20)]
        neg_samples = [[str(1 - (i % 2))] * 5 for i in range(20)]
        
        initial_dfa = learner.create_init_automata('Tabular', pos_samples, neg_samples)
        
        # Time optimization
        start_time = time.time()
        result = optimize_dfa_with_pso(
            initial_dfa=initial_dfa,
            learner=learner,
            data=data,
            labels=labels,
            threshold=0.8,
            n_particles=size,
            n_iterations=10,
            verbose=False
        )
        elapsed = time.time() - start_time
        
        print(f"    Time: {elapsed:.2f}s")
        print(f"    Result: {result['best_states']} states, accuracy={result['best_accuracy']:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='PSO DFA Optimizer Examples'
    )
    parser.add_argument(
        '--example',
        choices=['basic', 'tomita', 'benchmark'],
        default='basic',
        help='Example to run'
    )
    parser.add_argument(
        '--tomita-id',
        type=int,
        default=1,
        help='Tomita grammar ID (1-7)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Accuracy threshold'
    )
    parser.add_argument(
        '--particles',
        type=int,
        default=8,
        help='Number of PSO particles'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=15,
        help='Number of PSO iterations'
    )
    
    args = parser.parse_args()
    
    if args.example == 'basic':
        result = example_basic_optimization()
    elif args.example == 'tomita':
        result = example_with_tomita()
    elif args.example == 'benchmark':
        benchmark_optimization_time()
    
    print("\n[Done]")


if __name__ == '__main__':
    main()
