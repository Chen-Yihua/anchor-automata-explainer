"""
Backward compatibility shim for automaton.learner

This module now re-exports all classes and functions from the new learner module.
Please use `from learner import ...` instead of `from automaton.learner import ...`
"""

# Re-export everything from the new learner module
from learner import (
    BaseAutomataLearner,
    DFALearner,
    RegisterAutomataLearner,
    LearnerFactory,
    get_learner,
    AUTO_INSTANCE,
)

# For backward compatibility
__all__ = [
    'BaseAutomataLearner',
    'DFALearner',
    'RegisterAutomataLearner',
    'LearnerFactory',
    'get_learner',
    'AUTO_INSTANCE',
]
